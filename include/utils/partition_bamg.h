/*
 * @Author: Guyue
 * @Date: 2026-05-20 10:51:24
 * @LastEditTime: 2026-05-21 12:32:24
 * @LastEditors: Guyue
 * @FilePath: /Delta-PipeANN/include/utils/partition_bamg.h
 */
#pragma once

#include <omp.h>
#include <algorithm>
#include <atomic>
#include <bitset>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <ostream>
#include <queue>
#include <random>
#include <set>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "filesystem"
#include "ssd_index_defs.h"
#include "ssd_index.h"
#include "linux_aligned_file_reader.h"

#define READ_SECTOR_OFFSET(node_id) \
  ((uint64_t) node_id / nnodes_per_sector  + 1) * SECTOR_LEN + ((uint64_t) node_id % nnodes_per_sector) * max_node_len;
#ifndef INF
#define INF 0xffffffff
#endif  // INF

namespace GP {
  class bamg_partitioner {
  public:
    size_t _nd = 0;
    size_t _dim = 0;
    uint64_t _C = 0;
    uint64_t _max_node_len = 0;
    uint64_t _partition_num = 0;
    std::vector<std::vector<unsigned>> full_graph;
    std::vector<std::vector<unsigned>> _layout;
    std::vector<unsigned> block_assign;
    std::vector<float> raw_data;
    pipeann::Distance<float> *_distance;

    bamg_partitioner() = default;

    /**
     * @brief: 加载原始向量数据
     * @param {string&} data_file
     * @param {size_t} nd
     * @param {size_t} dim
     * @return {*}
     */    
    void load_raw_data(const std::string& data_file) {
      int num_points, dim;
      std::ifstream reader(data_file, std::ios::binary | std::ios::ate);
      reader.seekg(0, std::ios::beg);
      reader.read((char *) &num_points, sizeof(int));
      reader.read((char *) &dim, sizeof(int));
      LOG(INFO) << "Num points: " << num_points << " Dim: " << dim;
      raw_data.resize(num_points * dim);
      reader.read((char *) raw_data.data(), num_points * dim * sizeof(float));
    }

    template <typename T>
    void load_partitioned_graph(const std::string& in_file, const std::string& part_file) {
      std::cout << "loading disk index file: " << in_file << "..." << std::endl;
      _distance = pipeann::get_distance_function<T>(pipeann::L2);

      uint64_t C;
      uint64_t partition_nums;
      uint64_t nd;
      uint64_t max_node_len;

      std::ifstream part(part_file);
      part.read((char*) &C, sizeof(uint64_t));
      part.read((char*) &partition_nums, sizeof(uint64_t));
      part.read((char*) &nd, sizeof(uint64_t));
      std::cout << "C: " << C << " partition_nums: " << partition_nums << " nd: " << nd << std::endl;
      _layout.resize(partition_nums);
      for (unsigned i = 0; i < partition_nums; i++) {
        unsigned s;
        part.read((char*)&s, sizeof(unsigned));
        _layout[i].resize(s);
        part.read((char*) _layout[i].data(), sizeof(unsigned) * s);
      }
      block_assign.resize(nd);
      part.read((char*)block_assign.data(), sizeof(unsigned) * nd);
      part.close();

      std::ifstream in(in_file, std::ios::binary);
      pipeann::SSDIndexMetadata<T> meta;
      meta.load_from_disk_index(in);
      in.close();

      _nd = meta.npoints;
      _dim = meta.data_dim;
      _max_node_len = meta.max_node_len;
      _C = meta.nnodes_per_sector;
      _partition_num = ROUND_UP(_nd, _C) / _C;

      in.open(in_file, std::ios::binary);
      std::unique_ptr<char[]> mem_index = std::make_unique<char[]>(_partition_num * SECTOR_LEN);
      in.seekg(SECTOR_LEN, std::ios::beg);
      in.read(mem_index.get(), _partition_num * SECTOR_LEN);
      in.close();

      full_graph.resize(_nd);
      uint64_t des = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : des)
      for (unsigned i = 0; i < _partition_num; ++i) {
        char* sector_buf;
        pipeann::alloc_aligned((void **) &sector_buf, SECTOR_LEN, SECTOR_LEN);
        memcpy(sector_buf, mem_index.get() + i * SECTOR_LEN, SECTOR_LEN);
        
        for (unsigned j = 0; j < _layout[i].size(); ++j) {
          unsigned id = _layout[i][j];
          pipeann::LVQDiskNode<T> node(sector_buf, j, meta);
          
          std::vector<unsigned> tmp(node.nnbrs);
          des += node.nnbrs;
          memcpy(tmp.data(), node.nbrs, node.nnbrs * sizeof(uint32_t));
          full_graph[id].assign(tmp.begin(), tmp.end());
        }
        pipeann::aligned_free(sector_buf);
      }
      std::cout << "avg degree: " << (double)des / _nd << std::endl;
      mem_index.reset();
      _C = SECTOR_LEN / _max_node_len;
      _partition_num = ROUND_UP(_nd, _C) / _C;
      std::cout << "_nd: " << _nd << " _dim:" << _dim << " _C:" << _C << " pn:" << _partition_num << std::endl;
      std::cout << "load index over." << std::endl;

      std::cout << "Done. Loaded " << partition_nums << " blocks, capacity " << C << "." << std::endl;
    }

    std::vector<std::vector<unsigned>> build_BAMG(float alpha = 1.2f, int beta = 4, const std::string& output_file = "bamg.bin") {
      std::vector<std::vector<unsigned>> bamg_graph(_nd);

      for (unsigned u = 0; u < _nd; ++u) {
        std::vector<unsigned> C_out = full_graph[u];  // 候选
        std::vector<unsigned> R_out;                  // 块内边
        std::vector<unsigned> inter_candidates;

        // Step 1: 保留所有块内边
        for (unsigned v : C_out) {
          if (block_assign[u] == block_assign[v]) {
            bamg_graph[u].push_back(v);
          } else {
            inter_candidates.push_back(v);
          }
        }

        // Step 2: 剪枝
        for (unsigned q : inter_candidates) {
          bool occlude = false;

          for (unsigned v : R_out) {
            // 块内查找发现距离目标q最近的节点
            // 从v所在的块开始，通过beta限制跳数
            float C0_dist = search_within_block(block_assign[v], v, q, beta);
            float dist_vq = dist(v, q);
            
            // v所在块中如果发现距离q较近的节点
            if (C0_dist * alpha < dist_vq) {
              occlude = true;
              break;
            }

            // 对同一块内的节点添加双向边
            // if (block_assign[v] == block_assign[q]) {
            //   bamg_graph[v].push_back(q);
            //   bamg_graph[q].push_back(v);
            //   break;
            // }
          }

          if (!occlude) {
            R_out.push_back(q);
            // bamg_graph[u].push_back(q);
          }
        }

        // Step 3: Add retained inter-block edges to the final graph
        for (unsigned v : R_out) {
          bamg_graph[u].push_back(v);
        }
      }
      

      int degree = 0;
      size_t min_degree = 99;
      size_t max_degree = 0;
      for (unsigned u = 0; u < _nd; ++u) {
        std::sort(bamg_graph[u].begin(), bamg_graph[u].end());
        bamg_graph[u].erase(
          std::unique(bamg_graph[u].begin(), bamg_graph[u].end()), 
          bamg_graph[u].end()
        );
        degree += bamg_graph[u].size();
        min_degree = std::min(bamg_graph[u].size(), min_degree);
        max_degree = std::max(bamg_graph[u].size(), max_degree);
      }

      std::cout << "[BAMG] Save completed. Total edges in BAMG: \n" 
                << "Avg degree: " << (double)degree / _nd << "\n"
                << "Min degree: " << min_degree << "\n" 
                << "Max degree: " << max_degree << "\n"<< std::endl;

      return bamg_graph;
    }

    inline float dist(unsigned i, unsigned j) const {
      return _distance->compare(raw_data.data() + _dim * i, raw_data.data() + _dim * j, _dim);
    }

    // 在指定块内向目标节点贪心搜索(最多beta跳)
    float search_within_block(unsigned block_id, unsigned start_node, unsigned target_node, int beta) {
      std::queue<std::pair<unsigned, int>> queue;
      std::unordered_set<unsigned> explored;

      queue.push({start_node, 0});
      explored.insert(start_node);

      float min_dist = dist(start_node, target_node);
      float closest_dist_found = min_dist;

      while (!queue.empty()) {
        auto [u, depth] = queue.front();
        queue.pop();

        if (depth >= beta) {
          continue;
        }

        for (unsigned nbr : full_graph[u]) {
          bool is_intra_block = (block_assign[nbr] == block_id);
          if (!is_intra_block) {
            continue;
          }
          
          if (explored.count(nbr)) continue;

          float current_dist = dist(nbr, target_node);
          closest_dist_found = std::min(closest_dist_found, current_dist);

          if (current_dist < min_dist) {
            queue.push({nbr, depth + 1});
            min_dist = current_dist;
          }
          explored.insert(nbr);
        }
      }

      return closest_dist_found;
    }

    // 保存 BAMG
    template <typename T>
    void save_graph(const std::string& input_file, std::vector<std::vector<unsigned>> bamg_graph, const std::string& output_file) {
      std::ifstream in(input_file, std::ios::binary);
      pipeann::SSDIndexMetadata<T> meta;
      meta.load_from_disk_index(in);
      in.close();
      
      unsigned max_node_len = meta.max_node_len;
      unsigned nnodes_per_sector = meta.nnodes_per_sector;

      in.open(input_file, std::ios::binary);
      std::unique_ptr<char[]> mem_index = std::make_unique<char[]>(_partition_num * SECTOR_LEN);
      in.seekg(SECTOR_LEN, std::ios::beg);
      in.read(mem_index.get(), _partition_num * SECTOR_LEN);
      in.close();

      constexpr uint64_t kBlkSize = 64 * 1024 * 1024;
      uint64_t bytes_per_write = meta.nnodes_per_sector > 0 ? SECTOR_LEN : ROUND_UP(_max_node_len, SECTOR_LEN);
      uint64_t nodes_per_write = meta.nnodes_per_sector > 0 ? meta.nnodes_per_sector : 1;
      std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(SECTOR_LEN);
      std::unique_ptr<char[]> node_buf = std::make_unique<char[]>(meta.max_node_len);

      std::remove(output_file.c_str());
      cached_ofstream diskann_writer;
      diskann_writer.open(output_file, kBlkSize);
      memset(sector_buf.get(), 0, SECTOR_LEN);
      diskann_writer.write(sector_buf.get(), SECTOR_LEN);
      
      for (unsigned i = 0; i < _partition_num; ++i) {
        memset(sector_buf.get(), 0, SECTOR_LEN);
        for (unsigned j = 0; j < _layout[i].size(); ++j) {
          unsigned id = _layout[i][j];
          memset(node_buf.get(), 0, meta.max_node_len);
          // uint64_t index_offset = READ_SECTOR_OFFSET(id);
          uint64_t index_offset = i * SECTOR_LEN + j * max_node_len;
          memcpy((char*) node_buf.get(), 
                 (char*) mem_index.get() + index_offset, meta.max_node_len);
          pipeann::LVQDiskNode<T> node(node_buf.get(), 0, meta);
          node.nnbrs = bamg_graph[id].size();
          memcpy(node.nbrs, bamg_graph[id].data(), node.nnbrs * sizeof(uint32_t));

          uint64_t buf_offset = (uint64_t)j * meta.max_node_len;
          memcpy((char*) sector_buf.get() + buf_offset,
                 (char*) node_buf.get(), meta.max_node_len);
        }
        diskann_writer.write(sector_buf.get(), SECTOR_LEN);
      }
      mem_index.reset();
      diskann_writer.close();
      meta.save_to_disk_index(output_file);
    }
  };
} // namespace GP

