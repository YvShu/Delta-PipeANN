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
#include "utils/prune_neighbors.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "linux_aligned_file_reader.h"

namespace pipeann {
  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::insert_lvq_in_place(const T *point1, const TagT &tag, tsl::robin_set<uint32_t> *deletion_set) {
    // LOG(INFO) << "insert start";
    QueryBuffer<T> *read_data = this->pop_query_buf(point1);
    T *point = read_data->aligned_query_T;
    void *ctx = reader->get_ctx();

    uint32_t target_id = cur_id++;

    std::vector<Neighbor> exp_node_info;
    tsl::robin_map<uint32_t, T *> coord_map;
    coord_map.reserve(2 * this->params.L);

    // T *coord_buf = nullptr;
    // alloc_aligned((void **) &coord_buf, 10 * this->params.L * this->aligned_dim, 256);
    std::vector<uint64_t> page_ref{};
    
    // change start 修改为lvq SSD量化搜索函数
    // LOG(INFO) << "insert search start";
    // this->do_pipe_search_blind(point1, 0, params.L, params.beam_width, exp_node_info, &coord_map, coord_buf, nullptr,
    //                            deletion_set, false, &page_ref);
    this->do_beam_search_blind1(point1, 0, params.L, params.beam_width, exp_node_info, &coord_map, nullptr, deletion_set, false,
                                &page_ref, read_data);
    // this->do_pipe_search_blind(point1, 0, params.L, params.beam_width, exp_node_info, &coord_map, nullptr, deletion_set, false,
    //                            &page_ref, read_data);
    // LOG(INFO) << "insert search finish";
    // change end

    std::vector<uint32_t> new_nhood;

    // change start 修改为lvq页面对应的剪枝方法
    pipeann::prune_neighbors(exp_node_info, new_nhood, params, metric, [this, &coord_map](uint32_t a, uint32_t b) {
      return this->dist_cmp->compare(coord_map[a], coord_map[b], this->meta_.data_dim);
    });
    // change end

    // locs[new_nhood.size()] is the target, locs[0:new_nhood.size() - 1] are the neighbors.
    // lock the pages to write
    // aligned_free(coord_buf);

    std::set<uint64_t> pages_need_to_read;

#ifdef IN_PLACE_RECORD_UPDATE
    std::vector<uint64_t> locs;
    for (auto &nbr : new_nhood) {
      auto loc = id2loc(nbr);
      locs.emplace_back(loc);
      pages_need_to_read.insert(loc_sector_no(loc));
    }
    locs.push_back(target_id);
    pages_need_to_read.insert(loc_sector_no(target_id));
    set_id2loc(target_id, target_id);

    // update loc2id, target_id <-> target_id.
    cur_loc++;  // for target ID, atomic update.
    set_loc2id(target_id, target_id);
#else
    std::vector<uint64_t> locs = this->alloc_loc(new_nhood.size() + 1, page_ref, pages_need_to_read);
#endif

    std::set<uint64_t> pages_to_rmw_set;
    for (auto &loc : locs) {
      pages_to_rmw_set.insert(loc_sector_no(loc));
    }
    std::vector<IORequest> pages_to_rmw;
    // ordered because of std::set
    for (auto &page_no : pages_to_rmw_set) {
      pages_to_rmw.push_back(IORequest(page_no * SECTOR_LEN, size_per_io, nullptr, 0, 0));
    }
    
    // lock the target and the neighbor ids (ensure that sector_no does not change).
    auto pages_locked = pipeann::lockReqs(this->page_lock_table, pages_to_rmw);
    lock_vec(vec_lock_table, target_id, new_nhood);

    // re-read the candidate pages (mostly in the cache).
    std::unordered_map<uint32_t, char *> page_buf_map;

    // dynamically allocate update_buf to reduce memory footprint.
    // 2x MAX_N_EDGES for read + write, the update_buf is freed in bg_io_thread.
    assert(read_data->update_buf == nullptr);
    pipeann::alloc_aligned((void **) &read_data->update_buf, (2 * MAX_N_EDGES + 1) * size_per_io, SECTOR_LEN);
    auto &update_buf = read_data->update_buf;

    std::vector<IORequest> reads, writes_4k, writes;
    assert(new_nhood.size() < MAX_N_EDGES);
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      auto loc = id2loc(new_nhood[i]);
      reads.push_back(IORequest(loc_sector_no(loc) * SECTOR_LEN, size_per_io, update_buf + i * size_per_io, 0, 0));
      page_buf_map[loc_sector_no(loc)] = update_buf + i * size_per_io;
    }

    for (uint32_t i = new_nhood.size(); i < new_nhood.size() + pages_to_rmw.size(); ++i) {
      auto off = pages_to_rmw[i - new_nhood.size()].offset;
      writes_4k.push_back(IORequest(off, size_per_io, update_buf + i * size_per_io, 0, 0));
      // LOG(INFO) << off / SECTOR_LEN;
      uint64_t page = off / SECTOR_LEN;
      if (pages_need_to_read.find(page) != pages_need_to_read.end()) {
        reads.push_back(IORequest(off, size_per_io, update_buf + i * size_per_io, 0, 0));
      }
      page_buf_map[off / SECTOR_LEN] = update_buf + i * size_per_io;
    }

    // generate continuous writes from 4k writes.
    // dummy one.
    writes_4k.push_back(IORequest(std::numeric_limits<uint64_t>::max(), 0, nullptr, 0, 0));
    uint64_t start_idx = 0;
    uint64_t cur_off = writes_4k[0].offset;
    for (uint32_t i = 1; i < writes_4k.size(); ++i) {
      if (writes_4k[i].offset != cur_off + size_per_io) {
        writes.push_back(
            IORequest(writes_4k[start_idx].offset, size_per_io * (i - start_idx), writes_4k[start_idx].buf, 0, 0));
        start_idx = i;
      }
      cur_off = writes_4k[i].offset;
    }
    writes_4k.pop_back();

    reader->read_alloc(reads, ctx, &page_ref);

    // update the target node.
    auto sector = loc_sector_no(locs[new_nhood.size()]);
    LVQDiskNode<T> target_node = lvqnode_from_page(page_buf_map[sector], locs[new_nhood.size()]);
    
    // 对数据进行量化，得到lvq向量、最小值、步长
    std::vector<uint8_t> out(meta_.data_dim);
    float min_val = point[0];
    float max_val = point[0];
    for (size_t d = 1; d < meta_.data_dim; ++d) {
      if (point[d] < min_val) min_val = point[d];
      if (point[d] > max_val) max_val = point[d];
    }
    float step = 0.0f;
    step = (max_val - min_val) / 255.0f;
    for (size_t d = 0; d < meta_.data_dim; ++d) {
      int p_val = std::round((point[d] - min_val) / step);
      p_val = std::max(0, std::min(255, p_val));
      out[d] = static_cast<uint8_t>(p_val);
    }
    memcpy(target_node.coords, out.data(), meta_.data_dim * sizeof(uint8_t));
    
    target_node.minval = min_val;
    target_node.step = step;

    target_node.nnbrs = new_nhood.size();
    *(target_node.nbrs - 1) = target_node.nnbrs;
    memcpy(target_node.nbrs, new_nhood.data(), new_nhood.size() * sizeof(uint32_t));
    tags.insert_or_assign(target_id, tag);

    // 向量缓存：存储ID到向量的映射
    std::set<uint64_t> node_to_fetch;
    std::unordered_map<uint32_t, const T*> node_coords_cache;
    std::vector<std::vector<T>> coord_pool;
    std::vector<T> target_point(meta_.data_dim);
    for (size_t d = 0; d < meta_.data_dim; ++d) {
      target_point[d] = point[d];
    }
    coord_pool.push_back(std::move(target_point));
    node_coords_cache[target_id] = coord_pool.back().data();

    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      auto loc = id2loc(new_nhood[i]);
      auto r_sector = loc_sector_no(loc);

      LVQDiskNode<T> r_nbr_node = lvqnode_from_page(page_buf_map[r_sector], loc);

      for (uint32_t j = 0; j < r_nbr_node.nnbrs; ++j) {
        uint32_t nnbr_id = r_nbr_node.nbrs[j];
        auto nnbr_loc = id2loc(nnbr_id);
        auto nnbr_sector = loc_sector_no(nnbr_loc);

        // 邻居不在页面缓存中
        if (page_buf_map.find(nnbr_sector) == page_buf_map.end()) {
          node_to_fetch.insert(nnbr_id);
        } else {
          LVQDiskNode<T> node = lvqnode_from_page(page_buf_map[nnbr_sector], nnbr_loc);
          std::vector<T> point(meta_.data_dim);
          for (size_t d = 0; d < meta_.data_dim; ++d) {
            uint8_t q_val = node.coords[d];
            point[d] = (static_cast<T>(q_val) * node.step) + node.minval;
          }
          coord_pool.push_back(std::move(point));
          node_coords_cache[nnbr_id] = coord_pool.back().data();
        }
      }
    }
    
    if (!node_to_fetch.empty()) {
      char *buf = nullptr;
      pipeann::alloc_aligned((void **)&buf, size_per_io, SECTOR_LEN);
      for (auto id : node_to_fetch) {
        if (node_coords_cache.find(id) != node_coords_cache.end()) {
          continue;
        }
        uint64_t page_id = id2page(id);
        PageArr layout = get_page_layout(page_id);
        std::vector<IORequest> reqs;
        auto req = IORequest(page_id * SECTOR_LEN, size_per_io, buf, page_id * SECTOR_LEN, size_per_io);
        reqs.emplace_back(req);
        // reader->send_read_no_alloc(req, ctx);
        reader->read_alloc(reqs, ctx, &page_ref);        

        for (unsigned j = 0; j < meta_.nnodes_per_sector; ++j) {
          unsigned cur_id = layout[j];
          if (node_coords_cache.find(cur_id) == node_coords_cache.end() &&
              node_to_fetch.find(cur_id) != node_to_fetch.end()) {
            LVQDiskNode<T> node = lvqnode_from_page(buf, j);
            std::vector<T> point(meta_.data_dim);
            for (size_t d = 0; d < meta_.data_dim; ++d) {
              uint8_t q_val = node.coords[d];
              point[d] = (static_cast<T>(q_val) * node.step) + node.minval;
            }
            coord_pool.push_back(std::move(point));
            node_coords_cache[cur_id] = coord_pool.back().data();
          }
        }
      }
      aligned_free(buf);
      buf = nullptr;
    }

    // update the neighbors
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      auto loc = id2loc(new_nhood[i]);
      auto r_sector = loc_sector_no(loc);
      if (page_buf_map.find(r_sector) == page_buf_map.end()) {
        LOG(ERROR) << new_nhood[i] << " " << "Sector " << r_sector << " not found in page_buf_map";
        exit(-1);
      }

      LVQDiskNode<T> r_nbr_node = lvqnode_from_page(page_buf_map[r_sector], loc);
      std::vector<uint32_t> nhood(r_nbr_node.nnbrs + 1);
      nhood.assign(r_nbr_node.nbrs, r_nbr_node.nbrs + r_nbr_node.nnbrs);
      nhood.emplace_back(target_id);  // attention: we do not reuse IDs.

      if (node_coords_cache.find(new_nhood[i]) == node_coords_cache.end()) {
        std::vector<T> point(meta_.data_dim);
        for (size_t d = 0; d < meta_.data_dim; ++d) {
          uint8_t q_val = r_nbr_node.coords[d];
          point[d] = (static_cast<T>(q_val) * r_nbr_node.step) + r_nbr_node.minval;
        }
        coord_pool.push_back(std::move(point));
        node_coords_cache[new_nhood[i]] = coord_pool.back().data();
      }

      if (nhood.size() > this->params.R) {  // delta prune neighbors
        pipeann::delta_prune_neighbors(
            nhood, new_nhood[i], target_id, params, metric,
            [&](uint32_t center, const uint32_t *ids, uint32_t n, float *dists_out) {
              if (node_coords_cache.find(center) == node_coords_cache.end()) {
                LOG(ERROR) << center << " not found in node_coords_cache";
                exit(-1);
              }
              std::memset(dists_out, 0, n);
              for (uint32_t j = 0; j < n; ++j) {
                if (node_coords_cache.find(ids[j]) == node_coords_cache.end()) {
                  LOG(ERROR) << ids[j] << " not found in node_coords_cache";
                  exit(-1);
                }
                dists_out[j] = this->dist_cmp->compare(node_coords_cache[center],
                                                 node_coords_cache[ids[j]], 
                                                 (unsigned) aligned_dim);
              }
            });
        // nhood.resize(this->params.R);
      }

      auto w_sector = loc_sector_no(locs[i]);
      LVQDiskNode<T> w_nbr_node = lvqnode_from_page(page_buf_map[w_sector], locs[i]);
      w_nbr_node.nnbrs = (uint32_t) nhood.size();  // write to nnbrs reference.
      w_nbr_node.minval = (float) r_nbr_node.minval;
      w_nbr_node.step = (float) r_nbr_node.step;
      // memcpy(w_nbr_node.coords, r_nbr_node.coords, meta_.data_dim * sizeof(T));
      memcpy(w_nbr_node.coords, r_nbr_node.coords, meta_.data_dim * sizeof(uint8_t));
      memcpy(w_nbr_node.nbrs, nhood.data(), w_nbr_node.nnbrs * sizeof(uint32_t));
    }
    // LOG(INFO) << "neighbor finish";

    std::vector<uint64_t> write_page_ref;
    reader->wbc_write(writes, ctx, &write_page_ref);

#ifndef IN_PLACE_RECORD_UPDATE
    // update locs
    // no concurrency issue for target_id (as it can be only inserted).
    set_id2loc(target_id, locs[new_nhood.size()]);
    auto locked = lock_idx(idx_lock_table, target_id, new_nhood);
    auto page_locked = lock_page_idx(page_idx_lock_table, target_id, new_nhood);
    std::vector<uint64_t> orig_locs;
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      orig_locs.emplace_back(id2loc(new_nhood[i]));
      set_id2loc(new_nhood[i], locs[i]);
    }

    // with lock, for simple concurrency with alloc_loc.
    // Only for convenience, note that locs[new_nhood.size()] -> target.
    new_nhood.push_back(target_id);
    erase_and_set_loc(orig_locs, locs, new_nhood);
    unlock_page_idx(page_idx_lock_table, page_locked);
    unlock_idx(idx_lock_table, locked);
#endif
    unlock_vec(vec_lock_table, target_id, new_nhood);

    // LOG(INFO) << "ID " << target_id << " Target loc " << id2loc(target_id);

    // commit writes (in the background thread.)
#ifdef BG_IO_THREAD
    if (!page_ref.empty()) {
      auto bg_task = new BgTask{.thread_data = read_data,
                                .writes = std::move(writes),
                                .pages_to_unlock = std::move(pages_locked),
                                .pages_to_deref = std::move(write_page_ref),
                                .terminate = false};
      bg_tasks.push(bg_task);
      bg_tasks.push_notify_all();
    } else {
      pipeann::unlockReqs(this->page_lock_table, pages_locked);
    }
    reader->deref(&page_ref, ctx);
#else
    reader->write(writes, ctx);
    aligned_free(read_data->update_buf);
    read_data->update_buf = nullptr;

    pipeann::unlockReqs(this->page_lock_table, pages_locked);
    reader->deref(&write_page_ref, ctx);

    reader->deref(&page_ref, ctx);
    this->push_query_buf(read_data);
#endif
    return target_id;
  }

  template<class T, class TagT>
  void SSDIndex<T, TagT>::bg_io_thread() {
    auto ctx = reader->get_ctx();
    auto timer = pipeann::Timer();
    uint64_t n_tasks = 0;

    while (true) {
      auto task = bg_tasks.pop();
      while (task == nullptr) {
        this->bg_tasks.wait_for_push_notify();
        task = bg_tasks.pop();
      }

      if (unlikely(task->terminate)) {
        delete task;
        break;
      }

      reader->write(task->writes, ctx);
      aligned_free(task->thread_data->update_buf);
      task->thread_data->update_buf = nullptr;

      pipeann::unlockReqs(this->page_lock_table, task->pages_to_unlock);
      reader->deref(&task->pages_to_deref, ctx);
      this->push_query_buf(task->thread_data);
      delete task;
      ++n_tasks;

      if (timer.elapsed() >= 5000000) {
        LOG(INFO) << "Processed " << n_tasks << " tasks, throughput: " << (double) n_tasks * 1e6 / timer.elapsed()
                  << " tasks/sec.";
        timer.reset();
        n_tasks = 0;
      }
    }
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
