/*
 * @Author: Guyue
 * @Date: 2026-04-17 15:25:15
 * @LastEditTime: 2026-04-24 16:47:23
 * @LastEditors: Guyue
 * @FilePath: /Delta-PipeANN/src/search/beam_search_blind.cpp
 */
#include "aligned_file_reader.h"
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include "utils/timer.h"
#include "utils/tsl/robin_map.h"
#include "utils.h"
#include "utils/page_cache.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "linux_aligned_file_reader.h"

namespace pipeann {
  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::do_beam_search_blind(const T *query1, uint32_t mem_L, uint32_t l_search, const uint32_t beam_width,
                                               std::vector<Neighbor> &expanded_nodes_info,
                                               tsl::robin_map<uint32_t, T *> *coord_map, T *coord_buf, QueryStats *stats,
                                               tsl::robin_set<uint32_t> *exclude_nodes /* tags */, bool dyn_search_l,
                                               std::vector<uint64_t> *passthrough_page_ref) {
    QueryBuffer<T> *query_buf = pop_query_buf(query1);
    void *ctx = reader->get_ctx();

    const T *query = query_buf->aligned_query_T;
    query_buf->reset();
    T *data_buf = query_buf->coord_scratch;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);
    char *sector_scratch = query_buf->sector_scratch;
    uint64_t &sector_scratch_idx = query_buf->sector_idx;

    Timer query_timer;
    std::vector<Neighbor> retset;                                 // 候选队列
    std::vector<Neighbor> &full_retset = expanded_nodes_info;     // 距离结果记录
    std::unordered_map<unsigned, std::vector<uint32_t>> nbr_buf;  // 记录某节点的邻居有哪些
    auto &visited = *(query_buf->visited);                        // 标记是否已算过距离
    retset.resize(l_search + 1);
    full_retset.reserve(l_search * 10);
    unsigned cur_list_size = 0;

    // cleared every iteration
    std::vector<unsigned> frontier;
    using fnhood_t = std::tuple<unsigned, unsigned, char *>;
    std::vector<fnhood_t> frontier_nhoods;
    std::vector<IORequest> frontier_read_reqs;
    std::vector<uint32_t> vec_rdlocks;

    std::vector<uint64_t> new_page_ref{};
    std::vector<uint64_t> &page_ref = passthrough_page_ref ? *passthrough_page_ref : new_page_ref;

    // Helper 计算从磁盘取回的向量距离
    auto compute_dist_and_push = [&](const LVQDiskNode<T> &node, const unsigned id) -> float {
      T *node_fp_coords_copy = data_buf;

      std::vector<float> tmp(meta_.data_dim);
      for (size_t i = 0; i < meta_.data_dim; ++i) {
        uint8_t q_val = node.coords[i];
        tmp[i] = (static_cast<float>(q_val) * node.step) + node.minval;
      }
      memcpy(node_fp_coords_copy, tmp.data(), meta_.data_dim * sizeof(T));
      float cur_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);

      full_retset.push_back(Neighbor(id, cur_dist, true));
      return cur_dist;
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
      retset[cur_list_size++] = Neighbor(meta_.entry_point_id, 0.0f, true);
      visited.insert(meta_.entry_point_id);
      frontier.push_back(meta_.entry_point_id);
    }

    unsigned k = 0;
    unsigned num_ios = 0;

    auto cpu2_st = std::chrono::high_resolution_clock::now();

    std::vector<uint32_t> locked;
    if (!frontier.empty()) {
      locked = this->lock_idx(idx_lock_table, kInvalidID, frontier, true);
      for (uint64_t i = 0; i < frontier.size(); ++i) {
        uint32_t id = frontier[i];
        uint32_t loc = this->id2loc(id);
        uint64_t offset = loc_sector_no(loc) * SECTOR_LEN;
        auto sector_buf = sector_scratch + sector_scratch_idx * size_per_io;
        fnhood_t fnhood = std::make_tuple(id, loc, sector_buf);
        sector_scratch_idx++;
        frontier_nhoods.push_back(fnhood);
        frontier_read_reqs.emplace_back(IORequest(offset, size_per_io, sector_buf, u_loc_offset(loc), meta_.max_node_len, sector_scratch));
      
        num_ios++;
      }
      reader->read_alloc(frontier_read_reqs, ctx, &page_ref);

      this->unlock_idx(idx_lock_table, locked);
    }
    for (auto &frontier_nhood : frontier_nhoods) {
      auto [id, loc, sector_buf] = frontier_nhood;
      LVQDiskNode<T> node = lvqnode_from_page(sector_buf, loc);

      std::vector<uint32_t> nbr(node.nnbrs);
      memcpy(nbr.data(), node.nbrs, node.nnbrs * sizeof(uint32_t));
      nbr_buf[id] = nbr;
    }

    while (k < cur_list_size) {
      auto nk = cur_list_size;
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      sector_scratch_idx = 0;

      // 1、取出一个待扩展节点，将其邻居纳入读取队列
      uint32_t marker = k;
      while (marker < cur_list_size) {
        if (retset[marker].flag) {
          if (nbr_buf.find(retset[marker].id) == nbr_buf.end()) {
            LOG(ERROR) << retset[marker].id << " " << " not found in nbr_buf_map";
            exit(-1);
          }
          for (uint64_t i = 0; i < nbr_buf[retset[marker].id].size(); ++i) {
            if (visited.find(nbr_buf[retset[marker].id][i]) == visited.end()) {
              frontier.push_back(nbr_buf[retset[marker].id][i]);
            }
          }
          retset[marker].flag = false;
          nbr_buf.erase(retset[marker].id);
          break;
        }
        marker++;
      }

      // 2、发起读取请求，读出读取队列中的节点
      // std::vector<uint32_t> locked;
      if (!frontier.empty()) {
        locked = this->lock_idx(idx_lock_table, kInvalidID, frontier, true);
        for (uint64_t i = 0; i < frontier.size(); ++i) {
          uint32_t id = frontier[i];
          uint32_t loc = this->id2loc(id);
          uint64_t offset = loc_sector_no(loc) * SECTOR_LEN;
          auto sector_buf = sector_scratch + sector_scratch_idx * size_per_io;
          fnhood_t fnhood = std::make_tuple(id, loc, sector_buf);
          sector_scratch_idx++;
          frontier_nhoods.push_back(fnhood);
          frontier_read_reqs.emplace_back(IORequest(offset, size_per_io, sector_buf, u_loc_offset(loc), meta_.max_node_len, sector_scratch));
          if (stats != nullptr) {
            stats->n_4k++;
            stats->n_ios++;
          }  
          num_ios++;
        }
        reader->read_alloc(frontier_read_reqs, ctx, &page_ref);

        this->unlock_idx(idx_lock_table, locked);
      }
      
      // 3、计算查询与扩展点邻居节点的距离
      for (auto &frontier_nhood : frontier_nhoods) {
        auto [id, loc, sector_buf] = frontier_nhood;
        LVQDiskNode<T> node = lvqnode_from_page(sector_buf, loc);

        float cur_dist = compute_dist_and_push(node, id);
        visited.insert(id);

        std::vector<uint32_t> nbr(node.nnbrs);
        memcpy(nbr.data(), node.nbrs, node.nnbrs * sizeof(uint32_t));
        nbr_buf[id] = nbr;

        Neighbor nn(id, cur_dist, true);
        auto r = InsertIntoPool(retset.data(), cur_list_size, nn);
        if (cur_list_size < l_search) {
          cur_list_size++;
        }
        if (r < nk)
          nk = r;
      }

      if (nk <= k)
        k = nk;
      else 
        ++k;
    }
    // re-sort by distance
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    if (passthrough_page_ref == nullptr) {
      reader->deref(&page_ref, ctx);
    }

    push_query_buf(query_buf);

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
  }

  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::beam_search_blind(const T *query, const uint64_t k_search, const uint32_t mem_L,
                                              const uint64_t l_search, TagT *res_tags, float *distances,
                                              const uint64_t beam_width, QueryStats *stats,
                                              tsl::robin_set<uint32_t> *deleted_nodes, bool dyn_search_l) {
    // iterate to fixed point
    std::shared_lock lk(merge_lock);
    std::vector<Neighbor> expanded_nodes_info;
    this->do_beam_search_blind(query, mem_L, (uint32_t) l_search, (uint32_t) beam_width, expanded_nodes_info, nullptr,
                               nullptr, stats, deleted_nodes, dyn_search_l);
    uint64_t res_count = 0;
    for (uint32_t i = 0; i < l_search && res_count < k_search && i < expanded_nodes_info.size(); i++) {
      res_tags[res_count] = id2tag(expanded_nodes_info[i].id);
      distances[res_count] = expanded_nodes_info[i].distance;
      res_count++;
    }
    return res_count;
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
