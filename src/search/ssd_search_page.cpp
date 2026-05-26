#include "aligned_file_reader.h"
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>
#include "liburing.h"

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
  void SSDIndex<T, TagT>::do_ssd_search_page(const T *query1, uint32_t mem_L, uint32_t l_search, const uint32_t beam_width,
                                             std::vector<Neighbor> &expanded_nodes_info,
                                             tsl::robin_map<uint32_t, T *> *coord_map, T *coord_buf, QueryStats *stats,
                                             tsl::robin_set<uint32_t> *exclude_nodes /* tags */, bool dyn_search_l,
                                             std::vector<uint64_t> *passthrough_page_ref) {
    QueryBuffer<T> *query_buf = pop_query_buf(query1);
    // void *ctx = reader->get_ctx();
    void *ctx = reader->get_ctx(IORING_SETUP_SQPOLL);

    const T *query = query_buf->aligned_query_T;
    query_buf->reset();
    T *data_buf = query_buf->coord_scratch;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);
    char *sector_scratch = query_buf->sector_scratch;
    uint64_t &sector_scratch_idx = query_buf->sector_idx;

    Timer query_timer;
    std::vector<Neighbor> retset;                                         // 候选队列
    std::vector<Neighbor> &full_retset = expanded_nodes_info;             // 距离结果记录
    std::unordered_map<unsigned, std::vector<uint32_t>> nbr_buf;          // 记录某节点的邻居有哪些
    std::unordered_map<unsigned, float> computed_dists;                   // 记录计算过距离的节点
    tsl::robin_set<uint64_t> &visited = *(query_buf->visited);            // 标记是否已算过距离
    tsl::robin_set<unsigned> &page_visited = *(query_buf->page_visited);  // 标记页面是否已算过
    retset.resize(l_search + 1);
    full_retset.reserve(l_search * 10);
    unsigned cur_list_size = 0;
    uint64_t coord_buf_idx = 0;

    // cleared every iteration
    std::vector<unsigned> frontier;
    using page_fnhood_t = std::tuple<unsigned, unsigned, PageArr, char *>;
    std::vector<page_fnhood_t> frontier_nhoods;
    std::vector<IORequest> frontier_read_reqs;
    std::vector<uint64_t> buf2page_map;
    std::vector<unsigned> frontier_computed;
    using io_ss_t = std::tuple<unsigned, unsigned, PageArr>;
    std::vector<io_ss_t> last_io_snapshot;
    last_io_snapshot.reserve(params.R + 1);
    std::vector<char> last_pages(SECTOR_LEN * (params.R + 1));
    frontier.reserve(params.R + 1);
    frontier_nhoods.reserve(params.R + 1);
    frontier_read_reqs.reserve(params.R + 1);
    buf2page_map.reserve(params.R + 1);
    frontier_computed.reserve(params.R + 1);

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
        computed_dists[mem_tags[i]] = mem_dists[i];
        visited.insert(mem_tags[i]);
        frontier.push_back(mem_tags[i]);
        full_retset.push_back(Neighbor(mem_tags[i], mem_dists[i], true));
      }
    } else {
      retset[cur_list_size++] = Neighbor(meta_.entry_point_id, 0.0f, true);
      computed_dists[meta_.entry_point_id] = 0.0f;
      visited.insert(meta_.entry_point_id);
      frontier.push_back(meta_.entry_point_id);
    }
    std::sort(retset.begin(), retset.begin() + cur_list_size);
    
    unsigned k = 0;
    unsigned num_ios = 0;

    std::vector<uint32_t> locked, page_locked;
    int n_ios1 = 0;
    if (!frontier.empty()) {
      locked = this->lock_idx(idx_lock_table, kInvalidID, frontier, true);
      page_locked = this->lock_page_idx(page_idx_lock_table, kInvalidID, frontier, true);
      for (uint64_t i = 0; i < frontier.size(); ++i) {
        auto id = frontier[i];
        uint64_t page_id = id2page(id);

        auto buf = sector_scratch + sector_scratch_idx * size_per_io;

        PageArr layout = get_page_layout(page_id);
        page_fnhood_t fnhood = std::make_tuple(id, page_id, layout, buf);
        sector_scratch_idx++;
        frontier_nhoods.push_back(fnhood);

        frontier_read_reqs.emplace_back(
          IORequest(page_id * SECTOR_LEN, size_per_io, buf, page_id * SECTOR_LEN, size_per_io));
        num_ios++;
      }
      n_ios1 = reader->send_read_no_alloc(frontier_read_reqs, ctx);
      if (stats != nullptr) {
        stats->n_4k += n_ios1;
        stats->n_ios += n_ios1;
      }
    }

    if (!frontier.empty()) {
      for (int i = 0; i < n_ios1; ++i) {
        reader->poll_wait(ctx, true);
      }
      this->unlock_page_idx(page_idx_lock_table, page_locked);
      this->unlock_idx(idx_lock_table, locked);
    }

    for (auto &[id, pid, layout, sector_buf] : frontier_nhoods) {
      memcpy(last_pages.data() + last_io_snapshot.size() * SECTOR_LEN, sector_buf, SECTOR_LEN);
      last_io_snapshot.emplace_back(std::make_tuple(id, pid, layout));

      for (unsigned j = 0; j < meta_.nnodes_per_sector; ++j) {
        unsigned cur_id = layout[j];
        if (cur_id == id) {
          LVQDiskNode<T> node = lvqnode_from_page(sector_buf, j);
          std::vector<uint32_t> nbr(node.nnbrs);
          memcpy(nbr.data(), node.nbrs, node.nnbrs * sizeof(uint32_t));
          nbr_buf[id] = nbr;
        }
        computed_dists[cur_id] = -1;
      }
    }

    unsigned hop = 16;
    auto cpu_st = std::chrono::high_resolution_clock::now();
    while (k < cur_list_size) {
      auto io1_time_st = std::chrono::high_resolution_clock::now();
      auto nk = cur_list_size;
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      sector_scratch_idx = 0;
      buf2page_map.clear();
      frontier_computed.clear();

      // 1、取出一个待扩展节点，将其邻居纳入读取队列
      uint32_t marker = k;
      while (marker < cur_list_size) {
        // if (page_visited.find(pid) == page_visited.end() && retset[marker].flag) {
        if (retset[marker].flag) {
          if (nbr_buf.find(retset[marker].id) == nbr_buf.end()) {
            LOG(ERROR) << retset[marker].id << " " << " not found in nbr_buf_map";
            exit(-1);
          }
          for (uint64_t i = 0; i < nbr_buf[retset[marker].id].size(); ++i) {
            auto id = nbr_buf[retset[marker].id][i];
            if (visited.find(id) == visited.end()) {
              if (computed_dists.find(id) == computed_dists.end()) {
                frontier.push_back(id);
              } else {
                frontier_computed.push_back(id);
              }
            }
          }
          retset[marker].flag = false;
          nbr_buf.erase(retset[marker].id);
          break;
        }
        marker++;
      }

      // 2、发起异步IO读取请求，读出读取队列中的节点(扩展节点的邻居)
      int n_ios = 0;
      if (!frontier.empty()) {
        locked = this->lock_idx(idx_lock_table, kInvalidID, frontier, true);
        page_locked = this->lock_page_idx(page_idx_lock_table, kInvalidID, frontier, true);
        
        for (uint64_t i = 0; i < frontier.size(); ++i) {
          auto id = frontier[i];
          uint64_t page_id = id2page(id);
          
          if (std::find(buf2page_map.begin(), buf2page_map.end(), page_id) != buf2page_map.end()) {
            std::vector<uint64_t>::iterator result = find(buf2page_map.begin(), buf2page_map.end(), page_id); //查找target
	          int idx = distance(buf2page_map.begin(), result);
            auto buf = sector_scratch + idx * size_per_io;
            PageArr layout = get_page_layout(page_id);
            page_fnhood_t fnhood = std::make_tuple(id, page_id, layout, buf);
            frontier_nhoods.push_back(fnhood);
            continue;
          }

          auto buf = sector_scratch + sector_scratch_idx * size_per_io;
          buf2page_map[sector_scratch_idx] = page_id;
          
          PageArr layout = get_page_layout(page_id);
          page_fnhood_t fnhood = std::make_tuple(id, page_id, layout, buf);
          sector_scratch_idx++;
          frontier_nhoods.push_back(fnhood);

          frontier_read_reqs.emplace_back(
            IORequest(page_id * SECTOR_LEN, size_per_io, buf, page_id * SECTOR_LEN, size_per_io)); 
          num_ios++;
        }
        n_ios = reader->send_read_no_alloc(frontier_read_reqs, ctx);
        if (stats != nullptr) {
          stats->n_4k += n_ios;
          stats->n_ios += n_ios;
        }
      }
      auto io1_time_ed = std::chrono::high_resolution_clock::now();
      stats->io_us1 += std::chrono::duration_cast<std::chrono::microseconds>(io1_time_ed - io1_time_st).count();

      // 3、CPU/IO流水线
      auto cpu1_st = std::chrono::high_resolution_clock::now();
      for (size_t i = 0; i < last_io_snapshot.size(); ++i) {
        auto &[last_io_id, pid, page_layout] = last_io_snapshot[i];
        char *sector_buf = last_pages.data() + i * SECTOR_LEN;

        for (unsigned j = 0; j < meta_.nnodes_per_sector; ++j) {
          const unsigned id = page_layout[j];
          if ((computed_dists.find(id) != computed_dists.end() && computed_dists[id] != -1) || id == last_io_id || id == kAllocatedID || id == kInvalidID) {
            continue;
          }
          LVQDiskNode<T> node = lvqnode_from_page(sector_buf, j);
          
          T *node_fp_coords_copy = data_buf;
          std::vector<float> tmp(meta_.data_dim);
          for (size_t d = 0; d < meta_.data_dim; ++d) {
            uint8_t q_val = node.coords[d];
            tmp[d] = (static_cast<float>(q_val) * node.step) + node.minval;
          }
          memcpy(node_fp_coords_copy, tmp.data(), meta_.data_dim * sizeof(T));
          float cur_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);

          computed_dists[id] = cur_dist;
          // visited.insert(id);
          // full_retset.push_back(Neighbor(id, cur_dist, true));

          std::vector<uint32_t> nbr(node.nnbrs);
          memcpy(nbr.data(), node.nbrs, node.nnbrs * sizeof(uint32_t));
          nbr_buf[id] = nbr;
        }
      }
      last_io_snapshot.clear();
      auto cpu1_ed = std::chrono::high_resolution_clock::now();
      stats->cpu_us1 += std::chrono::duration_cast<std::chrono::microseconds>(cpu1_ed - cpu1_st).count();

      
      // 4、阻塞等待本轮发起的IO全部完成
      auto io_time_st = std::chrono::high_resolution_clock::now();
      if (!frontier.empty()) {
        if (hop > 0) {
          for (int i = 0; i < n_ios; ++i) {
            reader->poll_wait(ctx, true);
          }  
          hop--;
        } else {
          for (int i = 0; i < n_ios; ++i) {
            reader->poll_wait(ctx);
          }
        }
        this->unlock_page_idx(page_idx_lock_table, page_locked);
        this->unlock_idx(idx_lock_table, locked);
      }
      auto io_time_ed = std::chrono::high_resolution_clock::now();
      stats->io_us += std::chrono::duration_cast<std::chrono::microseconds>(io_time_ed - io_time_st).count();

      // 5、处理读取回来的新页面中的明确目标节点
      auto cpu2_st = std::chrono::high_resolution_clock::now();
      for (auto id : frontier_computed) {
        float cur_dist = computed_dists[id];
        visited.insert(id);
        full_retset.push_back(Neighbor(id, cur_dist, true));
        Neighbor nn(id, cur_dist, true);
        auto r = InsertIntoPool(retset.data(), cur_list_size, nn);
        if (cur_list_size < l_search) {
          cur_list_size++;
        }
        if (r < nk)
          nk = r;
      }
      for (auto &[id, pid, layout, sector_buf] : frontier_nhoods) {
        memcpy(last_pages.data() + last_io_snapshot.size() * SECTOR_LEN, sector_buf, SECTOR_LEN);
        last_io_snapshot.emplace_back(std::make_tuple(id, pid, layout));

        for (unsigned j = 0; j < meta_.nnodes_per_sector; ++j) {
          unsigned cur_id = layout[j];
          if (cur_id == id) {
            LVQDiskNode<T> node = lvqnode_from_page(sector_buf, j);
            
            T *node_fp_coords_copy = data_buf;
            std::vector<float> tmp(meta_.data_dim);
            for (size_t i = 0; i < meta_.data_dim; ++i) {
              uint8_t q_val = node.coords[i];
              tmp[i] = (static_cast<float>(q_val) * node.step) + node.minval;
            }
            memcpy(node_fp_coords_copy, tmp.data(), meta_.data_dim * sizeof(T));
            float cur_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);
            full_retset.push_back(Neighbor(id, cur_dist, true));
            computed_dists[id] = cur_dist;
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
          computed_dists[cur_id] = -1;
        }
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

    // if (passthrough_page_ref == nullptr) {
    //   reader->deref(&page_ref, ctx);
    // }

    push_query_buf(query_buf);

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
  }

  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::ssd_search_page(const T *query, const uint64_t k_search, const uint32_t mem_L,
                                            const uint64_t l_search, TagT *res_tags, float *distances,
                                            const uint64_t beam_width, QueryStats *stats,
                                            tsl::robin_set<uint32_t> *deleted_nodes, bool dyn_search_l) {
    // iterate to fixed point
    std::shared_lock lk(merge_lock);
    std::vector<Neighbor> expanded_nodes_info;
    this->do_ssd_search_page(query, mem_L, (uint32_t) l_search, (uint32_t) beam_width, expanded_nodes_info, nullptr,
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
