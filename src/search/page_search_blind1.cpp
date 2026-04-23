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
  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::page_search_blind1(const T *query1, const uint64_t k_search, const uint32_t mem_L,
                                               const uint64_t l_search, TagT *res_tags, float *distances,
                                               const uint64_t beam_width, QueryStats *stats) {
    QueryBuffer<T> *query_buf = pop_query_buf(query1);
    void *ctx = reader->get_ctx();

    const T *query = query_buf->aligned_query_T;
    query_buf->reset();

    T *data_buf = query_buf->coord_scratch;
    char *sector_scratch = query_buf->sector_scratch;
    uint64_t &sector_scratch_idx = query_buf->sector_idx;

    Timer query_timer;

    std::vector<Neighbor> retset;
    std::vector<Neighbor> full_retset;
    retset.reserve(4096);
    full_retset.reserve(4096);

    tsl::robin_set<uint64_t> &visited = *(query_buf->visited);
    tsl::robin_set<unsigned> &page_visited = *(query_buf->page_visited);

    unsigned cur_list_size = 0;

    // ===== 精确距离计算 =====
    auto compute_exact = [&](const LVQDiskNode<T> &node, unsigned id) {
      std::vector<float> point(meta_.data_dim);
      for (size_t i = 0; i < meta_.data_dim; ++i) {
        uint8_t q_val = node.coords[i];
        point[i] = (static_cast<float>(q_val) * node.step) + node.minval;
      }
      memcpy(data_buf, point.data(), meta_.data_dim * sizeof(T));

      float dist = dist_cmp->compare(query, data_buf, (unsigned) aligned_dim);
      full_retset.emplace_back(id, dist, true);
      return dist;
    };

    // ===== 初始化入口 =====
    {
      unsigned ep = meta_.entry_point_id;
      retset[cur_list_size++] = Neighbor(ep, 0.0f, true);
      visited.insert(ep);
    }

    std::sort(retset.begin(), retset.begin() + cur_list_size);

    unsigned k = 0;

    std::vector<unsigned> frontier;
    std::vector<IORequest> read_reqs;
    std::vector<std::tuple<unsigned, unsigned, PageArr, char *>> frontier_nodes;
    using page_fnhood_t = std::tuple<unsigned, unsigned, PageArr, char *>; 

    frontier.reserve(beam_width);
    read_reqs.reserve(beam_width);
    frontier_nodes.reserve(beam_width);

    stats->io_us = 0;
    stats->cpu_us = 0;

    while (k < cur_list_size) {
      unsigned nk = cur_list_size;

      frontier.clear();
      read_reqs.clear();
      frontier_nodes.clear();
      sector_scratch_idx = 0;

      // ===== 选beam =====
      unsigned marker = k;
      while (marker < cur_list_size && frontier.size() < beam_width) {
        unsigned id = retset[marker].id;
        unsigned pid = id2page(id);

        // && page_visited.find(pid) == page_visited.end()
        if (retset[marker].flag) {
          frontier.push_back(id);
          // page_visited.insert(pid);
          retset[marker].flag = false;
        }
        marker++;
      }

      // ===== 发起IO =====
      int n_ios = 0;
      if (!frontier.empty()) {
        for (auto id : frontier) {
          unsigned pid = id2page(id);

          char *buf = sector_scratch + sector_scratch_idx * SECTOR_LEN;
          PageArr layout = get_page_layout(pid);
          page_fnhood_t fnhood = std::make_tuple(id, pid, layout, buf);
          sector_scratch_idx++;
          frontier_nodes.push_back(fnhood);

          read_reqs.emplace_back(
              IORequest(pid * SECTOR_LEN, SECTOR_LEN, buf, pid * SECTOR_LEN, SECTOR_LEN));

          stats->n_ios++;
          stats->n_4k++;
        }
        
        n_ios = reader->send_read_no_alloc(read_reqs, ctx);
      }

      // ===== 等待IO =====
      auto io_st = std::chrono::high_resolution_clock::now();
      if (!frontier.empty()) {
        for (int i = 0; i < n_ios; i++) {
          reader->poll_wait(ctx);
        }
      }
      auto io_ed = std::chrono::high_resolution_clock::now();

      stats->io_us += std::chrono::duration_cast<std::chrono::microseconds>(io_ed - io_st).count();

      // ===== 处理page =====
      auto cpu_st = std::chrono::high_resolution_clock::now();
      for (auto &[id, pid, layout, buf] : frontier_nodes) {
        for (unsigned j = 0; j < meta_.nnodes_per_sector; j++) {
          unsigned nid = layout[j];

          if (nid != id || nid == kInvalidID || nid == kAllocatedID)
            continue;

          LVQDiskNode<T> node = lvqnode_from_page(buf, j);
          float dist = compute_exact(node, nid);

          // 插入候选
          // if (cur_list_size < l_search || dist < retset[cur_list_size - 1].distance) {
          //   Neighbor nn(nid, dist, true);
          //   auto r = InsertIntoPool(retset.data(), cur_list_size, nn);

          //   if (cur_list_size < l_search)
          //     cur_list_size++;

          //   if (r < nk)
          //     nk = r;
          // }

          // ===== 扩展邻居 =====
          for (unsigned m = 0; m < node.nnbrs; m++) {
            unsigned nnid = node.nbrs[m];

            if (visited.find(nnid) == visited.end()) {
              visited.insert(nnid);
              Neighbor nn(nnid, dist, true);
              auto r = InsertIntoPool(retset.data(), cur_list_size, nn);
              if (cur_list_size < l_search)
                cur_list_size++;

              if (r < nk)
                nk = r;
              // // 只加入retset，不算距离（延迟到读SSD）
              // if (cur_list_size < l_search) {
              //   retset[cur_list_size++] = Neighbor(nnid, dist, true);
              // }
            }
          }
        }
      }

      auto cpu_ed = std::chrono::high_resolution_clock::now();
      stats->cpu_us += std::chrono::duration_cast<std::chrono::microseconds>(cpu_ed - cpu_st).count();

      if (nk <= k)
        k = nk;
      else
        k++;
    }

    // ===== 排序输出 =====
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &a, const Neighbor &b) { return a < b; });

    uint64_t t = 0;
    for (size_t i = 0; i < full_retset.size() && t < k_search; i++) {
      if (i > 0 && full_retset[i].id == full_retset[i - 1].id)
        continue;

      res_tags[t] = id2tag(full_retset[i].id);
      if (distances)
        distances[t] = full_retset[i].distance;
      t++;
    }

    push_query_buf(query_buf);

    stats->total_us = query_timer.elapsed();

    return t;
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
} // namespace pipeann