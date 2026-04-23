/*
 * @Author: Guyue
 * @Date: 2026-04-21 10:25:44
 * @LastEditTime: 2026-04-23 10:19:22
 * @LastEditors: Guyue
 * @FilePath: /Delta-PipeANN/src/search/pipe_search_blind_node.cpp
 */
#include "aligned_file_reader.h"
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>
#ifndef USE_AIO
#include "liburing.h"
#endif

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include "utils/timer.h"
#include "utils/tsl/robin_set.h"
#include "utils.h"
#include "utils/page_cache.h"

#include <unistd.h>
#include <sys/syscall.h>

namespace pipeann {
  // 候选队列结构体：管理未进行评估的读取操作
  struct Candidate {
    unsigned id;
    float dist;

    bool operator<(const Candidate &other) const {
      return dist > other.dist;
    }
  };

  // io_t 结构体：用于跟踪当前在途的异步IO请求状态
  struct io_t {
    Neighbor nbr;         // 发起该IO请求的目标邻居节点
    unsigned page_id;     // 目标节点所在的物理页面(sector)ID
    unsigned loc;         // 目标节点所在页面内的具体偏移/位置
    IORequest *read_req;  // 指向底层异步IO请求对象的指针

    bool operator>(const io_t &rhs) const {
      return nbr.distance > rhs.nbr.distance;
    }

    bool operator<(const io_t &rhs) const {
      return nbr.distance < rhs.nbr.distance;
    }

    bool finished() {
      return read_req->finished;
    }
  };

  template<typename T>
  struct node_buf {
    unsigned id;
    float dist;
    std::vector<char> node;

    bool operator<(const node_buf &other) const {
      return dist > other.dist;
    }
  };

  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::pipe_search_blind_node(const T *query1, const uint64_t k_search, const uint32_t mem_L,
                                                   const uint64_t l_search, TagT *res_tags, float *distances,
                                                   const uint64_t beam_width, QueryStats *stats, AbstractSelector *selector,
                                                   const void *filter_data, const uint64_t relaxed_monotonicity_l) {
    QueryBuffer<T> *query_buf = pop_query_buf(query1);
    void *ctx = reader->get_ctx();

    if (beam_width > MAX_N_SECTOR_READS) {
      LOG(ERROR) << "Beamwidth can not be higher than MAX_N_SECTOR_READS";
      crash();
    }

    const T *query = query_buf->aligned_query_T;

    query_buf->reset();
    T *data_buf = query_buf->coord_scratch;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);
    char *sector_scratch = query_buf->sector_scratch;

    Timer query_timer;
    std::vector<Neighbor> retset;                 // 记录已精确计算的top l_search节点
    std::vector<Neighbor> full_retset;            // 记录已精确计算过的节点
    std::priority_queue<Candidate> read_queue;    // 读取队列
    std::queue<io_t> on_flight_ios;               // 记录在途IO队列
    std::priority_queue<node_buf<T>> node_buf_queue; // 暂存从磁盘读取成功、但还尚未被处理的页面
    auto &visited = *(query_buf->visited);
    retset.reserve(l_search);
    full_retset.reserve(l_search * 10);

    unsigned cur_list_size = 0;
    uint64_t n_computes = 0;

    // Helper 计算从磁盘取回的向量的距离
    auto compute_dists_and_push = [&](const LVQDiskNode<T> &node, const unsigned id) -> float {
      T *node_fp_coords_copy = data_buf;

      std::vector<float> point(meta_.data_dim);
      for (size_t i = 0; i < meta_.data_dim; ++i) {
        uint8_t q_val = node.coords[i];
        point[i] = (static_cast<float>(q_val) * node.step) + node.minval;
      }
      memcpy(node_fp_coords_copy, point.data(), meta_.data_dim * sizeof(T));
      float cur_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);
      
      full_retset.push_back(Neighbor(id, cur_dist, true));
      return cur_dist;
    };
    
    // Helper 提取某节点的邻居，将有潜力的邻居推入候选
    auto process_and_push_nbrs = [&](const LVQDiskNode<T> &node, const unsigned id, float dist) {
      Neighbor nn(id, dist, false);
      auto r = InsertIntoPool(retset.data(), cur_list_size, nn);
      if (cur_list_size < l_search) {
        cur_list_size++;
      }

      if (r < l_search) {
        for (unsigned m = 0; m < node.nnbrs; ++m) {
          unsigned nbor_id = node.nbrs[m];
          if (visited.find(nbor_id) == visited.end()) {
            visited.insert(nbor_id);
            read_queue.push({nbor_id, dist});
          }
        }
      }
    };

    // Helper 构造并下发单一的异步读盘请求
    auto send_read_req = [&](Neighbor &item) -> bool {
      // 锁定底层索引结构以防止读盘期间发生并发冲突
      this->lock_idx(idx_lock_table, item.id, std::vector<uint32_t>(), true);
      const unsigned loc = id2loc(item.id), page_id = loc_sector_no(loc);
      uint64_t &cur_buf_idx = query_buf->sector_idx;
      auto buf = sector_scratch + cur_buf_idx * size_per_io;
      auto &req = query_buf->reqs[cur_buf_idx];

      // 装配IORequest并下发读取请求
      req = IORequest(static_cast<uint64_t>(page_id) * SECTOR_LEN, size_per_io, buf, 
                      u_loc_offset(loc), meta_.max_node_len, sector_scratch);
      reader->send_read_no_alloc(req, ctx);

      // 记录到在途IO队列
      on_flight_ios.push(io_t{item, page_id, loc, &req});
      cur_buf_idx = (cur_buf_idx + 1) % MAX_N_SECTOR_READS;

      if (stats != nullptr) {
        stats->n_ios++;
      }
      return true;
    };

    // Helper 从读取队列中挑选最优质的n个节点并发起异步请求
    auto send_best_read_req = [&](uint32_t n) -> bool {
      unsigned n_sent = 0;
      while (!read_queue.empty() && n_sent < n) {
        Candidate cand = read_queue.top();
        read_queue.pop();
        Neighbor item(cand.id, cand.dist, false);
        send_read_req(item);
        n_sent++;
      }
      return n_sent != 0;
    };

    // Helper 轮询底层IO完成事件
    auto pool_all = [&]() {
      reader->poll_all(ctx);

      while (!on_flight_ios.empty() && on_flight_ios.front().finished()) {
        io_t &io = on_flight_ios.front();
        // LVQDiskNode<T> node = lvqnode_from_page((char *) io.read_req->buf, io.loc);
        std::vector<char> node(meta_.max_node_len);
        memcpy(node.data(), 
               (char *) io.read_req->buf + (meta_.nnodes_per_sector == 0 ? 0 : (io.loc % meta_.nnodes_per_sector) * meta_.max_node_len), 
               meta_.max_node_len);
        node_buf_queue.push({io.nbr.id, io.nbr.distance, node});
        
        this->unlock_idx(idx_lock_table, io.nbr.id);
        on_flight_ios.pop();
      }
    };

    // Helper 从内存缓存区(node_buf_queue)中处理那些已经读回来的最佳节点
    auto calc_best_node = [&](uint32_t n) {
      unsigned n_calc = 0;
      while (!node_buf_queue.empty() && n_calc < n) {
        auto [id, dist, node_buf] = node_buf_queue.top();
        LVQDiskNode<T> node = lvqnode_from_page(node_buf.data(), 0);
        float cur_dist = compute_dists_and_push(node, id);
        process_and_push_nbrs(node, id, cur_dist);
        node_buf_queue.pop();
        n_calc++;
        n_computes++;
      }
    };

    if (stats != nullptr) {
      stats->io_us = 0;
      stats->io_us1 = 0;
      stats->cpu_us = 0;
      stats->cpu_us1 = 0;
      stats->cpu_us2 = 0;
    }
    if (mem_L) {
    } else {
      retset[cur_list_size++] = Neighbor(meta_.entry_point_id, 0.0f, false);
      visited.insert(meta_.entry_point_id);
      read_queue.push({meta_.entry_point_id, 0.0f});
    }
    auto cpu2_st = std::chrono::high_resolution_clock::now();

    send_best_read_req(beam_width - on_flight_ios.size());

    while (n_computes < 4000 && (!read_queue.empty() || !on_flight_ios.empty())) {
      pool_all();
      int64_t to_send = beam_width - on_flight_ios.size();
      if (to_send > 0) {
        send_best_read_req(to_send);
      }
      calc_best_node(5);
    }
    
    while (!on_flight_ios.empty() && !node_buf_queue.empty()) {
      reader->poll_all(ctx);
      while (!on_flight_ios.empty() && on_flight_ios.front().finished()) {
        io_t &io = on_flight_ios.front();
        this->unlock_idx(idx_lock_table, io.nbr.id);
        on_flight_ios.pop();
      }
      calc_best_node(5);
    }

    auto cpu2_ed = std::chrono::high_resolution_clock::now();
    if (stats != nullptr) {
      stats->cpu_us2 = std::chrono::duration_cast<std::chrono::microseconds>(cpu2_ed - cpu2_st).count();
      stats->cpu_us = n_computes;
    }

    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    uint64_t t = 0;
    for (uint64_t i = 0; i < full_retset.size() && t < k_search; i++) {
      if (i > 0 && full_retset[i].id == full_retset[i - 1].id) {
        continue;
      }
      res_tags[t] = id2tag(full_retset[i].id);
      if (distances != nullptr) {
        distances[t] = full_retset[i].distance;
      }
      t++;
    }
    
    push_query_buf(query_buf);

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
    return t;
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
} // namespace pipeann