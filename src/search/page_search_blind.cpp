/*
 * @Author: Guyue
 * @Date: 2026-04-16 16:51:55
 * @LastEditTime: 2026-04-16 19:40:50
 * @LastEditors: Guyue
 * @FilePath: /Delta-PipeANN/src/search/page_search_blind.cpp
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
#include "utils/tsl/robin_set.h"
#include "utils.h"
#include "utils/page_cache.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "linux_aligned_file_reader.h"

namespace pipeann {
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
    std::vector<Neighbor> retset(4096);     // candidate pool for beam search
    tsl::robin_set<uint64_t> &visited = *(query_buf->visited);
    tsl::robin_set<unsigned> &page_visited = *(query_buf->page_visited);
    unsigned cur_list_size = 0;

    std::vector<Neighbor> full_retset;      // stores all exactly computed distances
    full_retset.reserve(4096);

    // Helper:compute dist between query and a node
    // update retset (if present) and append to full_retset
    auto compute_dists_and_update = [&](const LVQDiskNode<T> &node, const unsigned id) -> float {
      T *node_fp_coords_copy = data_buf;

      std::vector<float> point(meta_.data_dim);
      for (size_t i = 0; i < meta_.data_dim; ++i) {
        uint8_t q_val = node.coords[i];
        point[i] = (static_cast<float>(q_val) * node.step) + node.minval;
      }
      memcpy(node_fp_coords_copy, point.data(), meta_.data_dim * sizeof(T));
      
      float dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);

      // Add to full result set
      full_retset.push_back(Neighbor(id, dist, true));

      // Update retset if this id already exists (from a previous placeholder)
      for (unsigned i = 0; i < cur_list_size; ++i) {
        if (retset[i].id == id) {
          retset[i].distance = dist;
          retset[i].flag = true;
          return dist;
        }
      }

      // Not in retset yet, insert as a new candidate (should not happend often)
      retset[cur_list_size] = Neighbor(id, dist, true);
      ++cur_list_size;
      return dist;
    };

    // Helper:add neighbors of a node to retset as placeholders (distance = INF, flag = false)
    // They will be exactly evaluated when their page is loaded
    auto add_neighbors_as_placeholders = [&](LVQDiskNode<T> &node) -> unsigned {
      unsigned *node_nbrs = node.nbrs;
      unsigned nnbrs = node.nnbrs;
      unsigned best_insert_pos = cur_list_size; // track smallest index where a neighbor is inserted
      for (unsigned m = 0; m < nnbrs; ++m) {
        unsigned nid = node_nbrs[m];
        if (visited.find(nid) == visited.end()) {
          visited.insert(nid);
          // Insert placeholder with infinite distance
          Neighbor nn(nid, std::numeric_limits<float>::max(), false);
          auto r = InsertIntoPool(retset.data(), cur_list_size, nn);
          if (cur_list_size < l_search)
            ++cur_list_size;
          if (r < best_insert_pos)
            best_insert_pos = r;
        }
      }
      return best_insert_pos;
    };

    // stats
    stats->io_us = 0;
    stats->cpu_us = 0;

    // Initialize candidate pool with entry point(s)
    if (mem_L) {

    } else {
      // Single entry point
      retset[0].id = meta_.entry_point_id;
      retset[0].distance = 0.0f;
      retset[0].flag = false;
      visited.insert(meta_.entry_point_id);
      cur_list_size = 1;
    }

    std::sort(retset.begin(), retset.begin() + cur_list_size);

    unsigned num_ios = 0;
    unsigned k = 0;

    // cleared every iteration
    std::vector<unsigned> frontier;
    frontier.reserve(2 * beam_width);
    // <node_id, page_id, page_layout, page_buf>
    using page_fnhood_t = std::tuple<unsigned, unsigned, PageArr, char*>;
    std::vector<page_fnhood_t> frontier_nhoods;
    frontier_nhoods.reserve(2 * beam_width);
    std::vector<IORequest> frontier_read_reqs;
    frontier_read_reqs.reserve(2 * beam_width);

    // <node_id, page_id, page_layout>
    using io_ss_t = std::tuple<unsigned, unsigned, PageArr>;
    std::vector<io_ss_t> last_io_snapshot;
    last_io_snapshot.reserve(2 * beam_width);

    std::vector<char> last_pages(SECTOR_LEN * beam_width * 2);

    // Main search loop
    while (k < cur_list_size) {
      unsigned nk = cur_list_size;
      // clear iteration state
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      sector_scratch_idx = 0;

      // Select next beam of nodes whose pages will be read.
      // Only nodes with flag == true (distance already computed) are eligible.
      uint32_t marker = k;
      uint32_t num_seen = 0;

      while (marker < cur_list_size && frontier.size() < beam_width && num_seen < beam_width) {
        const unsigned pid = id2page(retset[marker].id);
        if (page_visited.find(pid) == page_visited.end()) {
          num_seen++;
          frontier.push_back(retset[marker].id);
          page_visited.insert(pid);
          // Do not change flag here? it remains true.
        }
        marker++;
      }

      // Issue read requests for the selected frontier pages
      std::vector<uint32_t> locked, page_locked;
      int n_ios = 0;
      if (!frontier.empty()) {
        if (stats != nullptr) {
          stats->n_hops++;
        }

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
          frontier_read_reqs.emplace_back(IORequest(page_id * SECTOR_LEN, size_per_io, buf, page_id * SECTOR_LEN, size_per_io));
          
          if (stats != nullptr) {
            stats->n_4k++;
            stats->n_ios++;
          }
          num_ios++;
        }

        n_ios = reader->send_read_no_alloc(frontier_read_reqs, ctx);
      }

      // --- PIPELINE: While I/O is an flight, process nodes from last round's pages ---
      auto cpu1_st = std::chrono::high_resolution_clock::now();
      for (size_t i = 0; i < last_io_snapshot.size(); ++i) {
        auto &[last_io_id, pid, page_layout] = last_io_snapshot[i];
        char *sector_buf = last_pages.data() + i * SECTOR_LEN;

        // Process all other nodes in the same page (excluding the one that was the frontier)
        for (unsigned j = 0; j < meta_.nnodes_per_sector; ++j) {
          const unsigned id = page_layout[j];
          if (id == last_io_id || id == kAllocatedID || id == kInvalidID) {
            continue;
          }
          LVQDiskNode<T> node = lvqnode_from_page(sector_buf, j);
          compute_dists_and_update(node, id);
          unsigned best_nbor_pos = add_neighbors_as_placeholders(node);
          if (best_nbor_pos < nk)
            nk = best_nbor_pos;
        }
      }

      last_io_snapshot.clear();
      auto cpu1_ed = std::chrono::high_resolution_clock::now();
      stats->cpu_us1 += std::chrono::duration_cast<std::chrono::microseconds>(cpu1_ed - cpu1_st).count();

      // Wait for I/O to complete
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

      // Process the newly read frontier nodes
      auto cpu_st = std::chrono::high_resolution_clock::now();
      for (auto &[id, pid, layout, sector_buf] : frontier_nhoods) {
        // Save page buffer for next round's pipeline processing
        memcpy(last_pages.data() + last_io_snapshot.size() * SECTOR_LEN, sector_buf, SECTOR_LEN);
        last_io_snapshot.emplace_back(std::make_tuple(id, pid, layout));

        // Process the exact frontier node itself
        for (unsigned j = 0; j < meta_.nnodes_per_sector; ++j) {
          unsigned cur_id = layout[j];
          if (cur_id == id) {
            LVQDiskNode<T> node = lvqnode_from_page(sector_buf, j);
            compute_dists_and_update(node, id);
            unsigned best_nbor_pos = add_neighbors_as_placeholders(node);
            if (best_nbor_pos < nk)
              nk = best_nbor_pos;
            break;
          }
        }
      }
      auto cpu_ed = std::chrono::high_resolution_clock::now();
      stats->cpu_us += std::chrono::duration_cast<std::chrono::microseconds>(cpu_ed - cpu_st).count();

      // Sort retset to keep the best candidate at the front
      // Placeholder (INF distance) will sink to the bottom
      std::sort(retset.begin(), retset.begin() + cur_list_size);

      // Advance k: if a new neighbor inserted before current k, backtrack
      if (nk <= k)
        k = nk;
      else
        ++k;

      // Ensure k points to a candidate (skip any that might still be placeholder at front?)
      // Actually we want k to eventually process all reachable nodes.
    }

    // Final sorting of distances
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    // Deduplicate and return top k_search results
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