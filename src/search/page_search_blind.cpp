/*
 * @Author: Guyue
 * @Date: 2026-04-17 00:02:38
 * @LastEditTime: 2026-04-20 14:43:34
 * @LastEditors: Guyue
 * @FilePath: /Delta-PipeANN/src/search/page_search_blind.cpp
 */
#include "aligned_file_reader.h"
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>
#include <queue>

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include "utils/timer.h"
#include "utils/tsl/robin_map.h"
#include "utils/tsl/robin_set.h"
#include "utils.h"
#include "utils/page_cache.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "linux_aligned_file_reader.h"

namespace pipeann {
  // 用于纯SSD盲搜的读取候选队列结构
  struct Candidate {
    unsigned id;
    float parent_dist;
    // 构造最小堆，使得父节点距离最小(最有希望)的节点被优先读取
    bool operator<(const Candidate& other) const {
      return parent_dist > other.parent_dist;
    }
  };
  
  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::page_search_blind(const T *query1, const uint64_t k_search, const uint32_t mem_L,
                                              const uint64_t l_search, TagT *res_tags, float *distances,
                                              const uint64_t beam_width, QueryStats *stats) {
    QueryBuffer<T> *query_buf = pop_query_buf(query1);
    void *ctx = reader->get_ctx();
    
    if (beam_width > MAX_N_SECTOR_READS) {
      LOG(ERROR) << "Beamwidth can not be higher than MAX_N_SECTOR_READS";
      crash();
    }
    const T *query = query_buf->aligned_query_T;

    // reset query
    query_buf->reset();

    // pointers to current vector for comparison
    T *data_buf = query_buf->coord_scratch;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_buf->sector_scratch;
    uint64_t &sector_scratch_idx = query_buf->sector_idx;

    Timer query_timer, io_timer, cpu_timer;
    // retset 仅保存精确评估过距离的节点
    std::vector<Neighbor> retset(4096);
    tsl::robin_set<uint64_t> &visited = *(query_buf->visited);
    tsl::robin_set<unsigned> &page_visited = *(query_buf->page_visited);
    unsigned cur_list_size = 0;

    std::vector<Neighbor> full_retset;
    full_retset.reserve(4096);

    // 待读队列
    std::priority_queue<Candidate> read_queue;

    // Helper Lambda 1：计算确切距离并存入全量返回集合
    auto compute_dists_and_push = [&](const LVQDiskNode<T> &node, const unsigned id) -> float {
      T *node_fp_coords_copy = data_buf;

      std::vector<float> point(meta_.data_dim);
      for (size_t i = 0; i < meta_.data_dim; ++i) {
        uint8_t q_val = node.coords[i];
        point[i] = (static_cast<float>(q_val) * node.step) + node.minval;
      }
      memcpy(node_fp_coords_copy, point.data(), meta_.data_dim * sizeof(T));

      float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);
      full_retset.push_back(Neighbor(id, cur_expanded_dist, true));
      return cur_expanded_dist;
    };

    // Helper Lambda 2：处理目标节点及其邻居，盲搜扩展
    auto process_node_and_neighbors = [&](LVQDiskNode<T> &node, const unsigned id, float dist) {
      Neighbor nn(id, dist, false);
      // 插入有序的确切距离池中
      auto r = InsertIntoPool(retset.data(), cur_list_size, nn);
      if (cur_list_size < l_search) {
        ++cur_list_size;
      }

      // 剪枝策略：只有当该节点能排进当前 top l_search 时，才值得去扩展它的邻居
      if (r < l_search) {
        unsigned *node_nbrs = node.nbrs;
        unsigned nnbrs = node.nnbrs;
        for (unsigned m = 0; m < nnbrs; ++m) {
          unsigned nbor_id = node_nbrs[m];
          if (visited.find(nbor_id) == visited.end()) {
            visited.insert(nbor_id);
            // 用父节点的精确距离作为探索子节点的启发式依据（Proxy Distance）
            read_queue.push({nbor_id, dist});
          }
        }
      }
    };

    // stats.
    stats->io_us = 0;
    stats->cpu_us = 0;

    // 初始化盲搜起始点
    if (mem_L) {
      std::vector<unsigned> mem_tags(mem_L);
      std::vector<float> mem_dists(mem_L);
      mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(), mem_dists.data());
      for (unsigned i = 0; i < std::min((unsigned)mem_L, (unsigned)l_search); ++i) {
        unsigned ep_id = mem_tags[i];
        if (visited.find(ep_id) == visited.end()) {
          visited.insert(ep_id);
          read_queue.push({ep_id, mem_dists[i]}); 
        }
      }
    } else {
      // Single entry point
      retset[cur_list_size].id = meta_.entry_point_id;
      retset[cur_list_size].distance = 0.0f;
      retset[cur_list_size++].flag = false;
      visited.insert(meta_.entry_point_id);
      read_queue.push({meta_.entry_point_id, 0.0f});
    }

    unsigned num_ios = 0;

    // 每次迭代清空
    std::vector<unsigned> frontier;
    frontier.reserve(2 * beam_width);
    using page_fnhood_t = std::tuple<unsigned, unsigned, PageArr, char *>;  // <node_id, page_id, page_layout, page_buf>
    std::vector<page_fnhood_t> frontier_nhoods;
    frontier_nhoods.reserve(2 * beam_width);
    std::vector<IORequest> frontier_read_reqs;
    frontier_read_reqs.reserve(2 * beam_width);

    using io_ss_t = std::tuple<unsigned, unsigned, PageArr>;  // <node_id, page_id, page_layout>
    std::vector<io_ss_t> last_io_snapshot;
    last_io_snapshot.reserve(2 * beam_width);

    std::vector<char> last_pages(SECTOR_LEN * beam_width * 2);

    // Search on Disk
    // 只要有待读节点，或者上一轮还留下缓存供overlap评估，就继续循环
    while (!read_queue.empty() || !last_io_snapshot.empty()) {
      // clear iteration state
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      sector_scratch_idx = 0;

      // 1. 构建新的 frontier 队列
      while (!read_queue.empty() && frontier.size() < beam_width) {
        Candidate cand = read_queue.top();
        read_queue.pop();

        const unsigned pid = id2page(cand.id);
        if (page_visited.find(pid) == page_visited.end()) {
          frontier.push_back(cand.id);
          page_visited.insert(pid);
        }
      }

      if (frontier.empty() && last_io_snapshot.empty()) {
        break; // 结束条件达到
      }

      // 2. 发起底层的磁盘异步 I/O 读取
      std::vector<uint32_t> locked, page_locked;
      int n_ios = 0;
      if (!frontier.empty()) {
        if (stats != nullptr)
          stats->n_hops++;

        locked = this->lock_idx(idx_lock_table, kInvalidID, frontier, true);
        page_locked = this->lock_page_idx(page_idx_lock_table, kInvalidID, frontier, true);

        for (uint64_t i = 0; i < frontier.size(); i++) {
          auto id = frontier[i];
          uint64_t page_id = id2page(id);
          auto buf = sector_scratch + sector_scratch_idx * size_per_io;
          PageArr layout = get_page_layout(page_id);
          page_fnhood_t fnhood = std::make_tuple(id, page_id, layout, buf);
          sector_scratch_idx++;
          frontier_nhoods.push_back(fnhood);
          // 构建读取请求
          frontier_read_reqs.emplace_back(
              IORequest(page_id * SECTOR_LEN, size_per_io, buf, page_id * SECTOR_LEN, size_per_io));
          if (stats != nullptr) {
            stats->n_4k++;
            stats->n_ios++;
          }
          num_ios++;
        }

        n_ios = reader->send_read_no_alloc(frontier_read_reqs, ctx);
      }

      // 3. CPU OVERLAP：处理上一轮取回来的页面里附带的其他邻居节点
      auto cpu1_st = std::chrono::high_resolution_clock::now();
      // for (size_t i = 0; i < last_io_snapshot.size(); ++i) {
      //   auto &[last_io_id, pid, page_layout] = last_io_snapshot[i];
      //   char *sector_buf = last_pages.data() + i * SECTOR_LEN;

      //   for (unsigned j = 0; j < meta_.nnodes_per_sector; ++j) {
      //     const unsigned id = page_layout[j];
      //     // 剔除上一轮已经处理过的主目标节点和非法节点
      //     if (id == last_io_id || id == kAllocatedID || id == kInvalidID) {
      //       continue;
      //     }
      //     LVQDiskNode<T> node = lvqnode_from_page(sector_buf, j);
      //     float dist = compute_dists_and_push(node, id);
          
      //     // 以纯盲搜方式进行处理（提取邻居、更新队列）
      //     process_node_and_neighbors(node, id, dist);
      //   }
      // }
      last_io_snapshot.clear();
      auto cpu1_ed = std::chrono::high_resolution_clock::now();
      stats->cpu_us1 += std::chrono::duration_cast<std::chrono::microseconds>(cpu1_ed - cpu1_st).count();

      // 4. 等待磁盘 I/O 结果
      auto io_time_st = std::chrono::high_resolution_clock::now();
      if (!frontier.empty()) {
        for (int i = 0; i < n_ios; ++i) {
          reader->poll_wait(ctx);
        }
        this->unlock_page_idx(page_idx_lock_table, page_locked);
        this->unlock_idx(idx_lock_table, locked);
      }
      auto io_time_ed = std::chrono::high_resolution_clock::now();
      stats->io_us += std::chrono::duration_cast<std::chrono::microseconds>(io_time_ed - io_time_st).count();

      // 5. 计算当前轮从磁盘取回的“目标节点”
      auto cpu_st = std::chrono::high_resolution_clock::now();
      for (auto &[id, pid, layout, sector_buf] : frontier_nhoods) {
        // 保存 Page 到 last_pages() / last_io_snapshot 供下一轮 CPU Overlap 使用
        memcpy(last_pages.data() + last_io_snapshot.size() * SECTOR_LEN, sector_buf, SECTOR_LEN);
        last_io_snapshot.emplace_back(std::make_tuple(id, pid, layout));

        for (unsigned j = 0; j < meta_.nnodes_per_sector; ++j) {
          unsigned cur_id = layout[j];
          if (cur_id == id) { // 仅处理主目标节点
            LVQDiskNode<T> node = lvqnode_from_page(sector_buf, j);
            float dist = compute_dists_and_push(node, id);
            process_node_and_neighbors(node, id, dist);
          }
        }
      }
      auto cpu_ed = std::chrono::high_resolution_clock::now();
      stats->cpu_us += std::chrono::duration_cast<std::chrono::microseconds>(cpu_ed - cpu_st).count();
    }

    // 后处理：将全量评测过的节点按准确距离排序以返回 k_search
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    // copy k_search values
    uint64_t t = 0;
    for (uint64_t i = 0; i < full_retset.size() && t < k_search; i++) {
      if (i > 0 && full_retset[i].id == full_retset[i - 1].id) {
        continue; // 去重
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