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
  void SSDIndex<T, TagT>::do_ssd_search_node(const T *query1, uint32_t mem_L, uint32_t l_search, const uint32_t beam_width,
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
    tsl::robin_set<uint64_t> visited(4096);                       // 标记是否已算过距离
    retset.resize(l_search + 1);
    full_retset.reserve(l_search * 10);
    unsigned cur_list_size = 0;
    uint64_t coord_buf_idx = 0;

    // cleared every iteration
    std::vector<unsigned> frontier;
    using fnhood_t = std::tuple<unsigned, unsigned, char *>;
    std::vector<fnhood_t> frontier_nhoods;
    std::vector<IORequest> frontier_read_reqs;

    std::vector<uint64_t> new_page_ref{};
    std::vector<uint64_t> &page_ref = passthrough_page_ref ? *passthrough_page_ref : new_page_ref;

    if (stats != nullptr) {
      stats->io_us = 0;
      stats->io_us1 = 0;
      stats->cpu_us = 0;
      stats->cpu_us1 = 0;
      stats->cpu_us2 = 0;
    }
    
    if (mem_L) {
      std::vector<unsigned> mem_tags(mem_L);
      std::vector<float> mem_dists(mem_L);
      mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(), mem_dists.data());
      for (unsigned i = 0; i < mem_L; ++i) {
        retset[cur_list_size++] = Neighbor(mem_tags[i], mem_dists[i], true);
        visited.insert(mem_tags[i]);
        frontier.push_back(mem_tags[i]);
        full_retset.push_back(Neighbor(mem_tags[i], mem_dists[i], true));
      }
    } else {
      retset[cur_list_size++] = Neighbor(meta_.entry_point_id, 0.0f, true);
      visited.insert(meta_.entry_point_id);
      frontier.push_back(meta_.entry_point_id);
    }
    std::sort(retset.begin(), retset.begin() + cur_list_size);

    unsigned k = 0;
    unsigned num_ios = 0;

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

    auto cpu_st = std::chrono::high_resolution_clock::now();
    while (k < cur_list_size) {
      auto io1_time_st = std::chrono::high_resolution_clock::now();
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
          // nbr_buf.erase(retset[marker].id);
          break;
        }
        marker++;
      }
      auto io1_time_ed = std::chrono::high_resolution_clock::now();
      stats->io_us1 += std::chrono::duration_cast<std::chrono::microseconds>(io1_time_ed - io1_time_st).count();


      // 2、发起读取请求，读出读取队列中的节点
      auto io_time_st = std::chrono::high_resolution_clock::now();
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
      auto io_time_ed = std::chrono::high_resolution_clock::now();
      stats->io_us += std::chrono::duration_cast<std::chrono::microseconds>(io_time_ed - io_time_st).count();

      // 3、计算查询与扩展点邻居节点的距离
      auto cpu2_st = std::chrono::high_resolution_clock::now();
      for (auto &frontier_nhood : frontier_nhoods) {
        auto [id, loc, sector_buf] = frontier_nhood;
        LVQDiskNode<T> node = lvqnode_from_page(sector_buf, loc);

        T *node_fp_coords_copy = data_buf;

        std::vector<float> tmp(meta_.data_dim);
        for (size_t i = 0; i < meta_.data_dim; ++i) {
          uint8_t q_val = node.coords[i];
          tmp[i] = (static_cast<float>(q_val) * node.step) + node.minval;
        }
        memcpy(node_fp_coords_copy, tmp.data(), meta_.data_dim * sizeof(T));
        float cur_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);

        if (coord_map != nullptr) {
          if (unlikely(coord_buf == nullptr || coord_buf_idx >= 4096)) {
            LOG(ERROR) << "Please allocate larger coord_buf.";
            crash();
          }
          T *coord_ptr = coord_buf + (coord_buf_idx * aligned_dim);
          memcpy(coord_ptr, tmp.data(), meta_.data_dim * sizeof(T));
          coord_map->insert(std::make_pair(id, coord_ptr));
          coord_buf_idx++;
        }

        full_retset.push_back(Neighbor(id, cur_dist, true));
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
      auto cpu2_ed = std::chrono::high_resolution_clock::now();
      stats->cpu_us2 += std::chrono::duration_cast<std::chrono::microseconds>(cpu2_ed - cpu2_st).count();

      if (nk <= k)
        k = nk;
      else 
        ++k;
    }
    auto cpu_ed = std::chrono::high_resolution_clock::now();
    stats->cpu_us += std::chrono::duration_cast<std::chrono::microseconds>(cpu_ed - cpu_st).count();
    
    
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
  size_t SSDIndex<T, TagT>::ssd_search_node(const T *query, const uint64_t k_search, const uint32_t mem_L,
                                            const uint64_t l_search, TagT *res_tags, float *distances,
                                            const uint64_t beam_width, QueryStats *stats,
                                            tsl::robin_set<uint32_t> *deleted_nodes, bool dyn_search_l) {
    // iterate to fixed point
    std::shared_lock lk(merge_lock);
    std::vector<Neighbor> expanded_nodes_info;
    this->do_ssd_search_node(query, mem_L, (uint32_t) l_search, (uint32_t) beam_width, expanded_nodes_info, nullptr,
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
