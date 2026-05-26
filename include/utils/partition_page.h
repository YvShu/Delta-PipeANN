/*
 * @Author: Guyue
 * @Date: 2026-05-08 17:01:55
 * @LastEditTime: 2026-05-20 10:28:15
 * @LastEditors: Guyue
 * @FilePath: /Delta-PipeANN/include/utils/partition_page.h
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

#ifndef INF
#define INF 0xffffffff
#endif  // INF

#ifndef READ_U64
#define READ_U64(stream, val) stream.read((char *)&val, sizeof(_u64))
#endif  // !READ_U64

#ifndef ROUND_UP
#define ROUND_UP(X, Y) (((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0)) * (Y)
#endif  // !ROUND_UP 向上取整，将 X 按 Y 对齐(整数倍)

namespace GP {

  namespace fs = std::filesystem;

  inline size_t get_file_size(const std::string &filename) {
    std::ifstream reader(filename, std::ios::binary | std::ios::ate);
    if (!reader.fail() && reader.is_open()) {
      size_t end_pos = reader.tellg();
      reader.close();
      return end_pos;
    } else {
      std::cout << "Could not open file: " << filename << std::endl;
      return 0;
    }
  }

  class graph_partitioner {
  public:
    graph_partitioner(const char *indexName, const char *data_type = "float", 
                      bool load_disk = true, unsigned BS = 1, bool visual = false) {
      _visual = visual;
      std::srand(static_cast<unsigned int>(std::time(nullptr)));

      _rd = new std::random_device();
      _gen = new std::mt19937((*_rd)());
      _dis = new std::uniform_real_distribution<>(0, 1);

      if (load_disk) {
        if (std::string(data_type) == std::string("uint8")) {
          load_disk_index<uint8_t>(indexName, BS);
        } else if (std::string(data_type) == std::string("float")) {
          load_disk_index<float>(indexName, BS);
        } else {
          std::cout << "not support type" << std::endl;
          exit(-1);
        }
      } else {
        std::cout << "Could not load index from memory!" << std::endl;
        return;
      }
      
      // copy to direct_graph
      direct_graph.clear();
      direct_graph.resize(full_graph.size());
#pragma omp parallel for
      for (unsigned i = 0; i < _nd; ++i) {
        direct_graph[i].assign(full_graph[i].begin(), full_graph[i].end());
      }

      // reverse graph
      std::vector<std::mutex> ms(_nd);
      reverse_graph.resize(_nd);
#pragma omp parallel for shared(reverse_graph, direct_graph)
      for (unsigned i = 0; i < _nd; i++) {
        for (unsigned j = 0; j < direct_graph[i].size(); j++) {
          std::lock_guard<std::mutex> lock(ms[direct_graph[i][j]]);
          reverse_graph[direct_graph[i][j]].emplace_back(i);
        }
      }
      std::cout << "reverse graph done." << std::endl;
      for (unsigned i = 0; i < _partition_number; i++) {
        pmutex.push_back(std::make_unique<std::mutex>());
      }
    }

    template <typename T>
    void load_disk_index(const char *index_name, int BS = 1) {
      std::cout << "loading disk index file: " << index_name << "... " << std::flush;
      std::ifstream in;
      in.exceptions(std::ifstream::failbit | std::ifstream::badbit);

      try {
        in.open(index_name, std::ios::binary);
        pipeann::SSDIndexMetadata<T> meta;
        meta.load_from_disk_index(in);
        in.close();

        _nd = meta.npoints;
        _dim = meta.data_dim;
        _max_node_len = meta.max_node_len;
        C = meta.nnodes_per_sector;

        _partition_number = ROUND_UP(_nd, C) / C;

        in.open(index_name, std::ios::binary);
        std::unique_ptr<char[]> mem_index = std::make_unique<char[]>(_partition_number * SECTOR_LEN);
        in.seekg(SECTOR_LEN, std::ios::beg);
        in.read(mem_index.get(), _partition_number * SECTOR_LEN);
        in.close();

        full_graph.resize(_nd);
        uint64_t des = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : des)
        for (unsigned i = 0; i < _partition_number; ++i) {
          char* sector_buf;
          pipeann::alloc_aligned((void **) &sector_buf, SECTOR_LEN, SECTOR_LEN);
          memcpy(sector_buf, mem_index.get() + i * SECTOR_LEN, SECTOR_LEN);
          
          for (unsigned j = 0; j < C && i * C + j < _nd; ++j) {
            pipeann::LVQDiskNode<T> node(sector_buf, j, meta);
            
            std::vector<unsigned> tmp(node.nnbrs);
            des += node.nnbrs;
            memcpy(tmp.data(), node.nbrs, node.nnbrs * sizeof(uint32_t));
            full_graph[i * C + j].assign(tmp.begin(), tmp.end());
          }
          pipeann::aligned_free(sector_buf);
        }
        std::cout << "avg degree: " << (double)des / _nd << std::endl;
        mem_index.reset();
        C = (SECTOR_LEN * BS) / _max_node_len;
        _partition_number = ROUND_UP(_nd, C) / C;
        std::cout << "_nd: " << _nd << " _dim:" << _dim << " C:" << C << " pn:" << _partition_number << std::endl;
        std::cout << "load index over." << std::endl;
      } catch (std::system_error &e) {
        std::cout << "open file " << index_name << " error!" << std::endl;
        exit(-1);
      }
    }

    void save_partition(const char *filename) {
      // re_id2pid();
      std::ofstream writer(filename, std::ios::binary | std::ios::out);
      std::cout << "writing bin: " << filename << std::endl;
      writer.write((char *)&C, sizeof(uint64_t));
      writer.write((char *)&_partition_number, sizeof(uint64_t));
      writer.write((char *)&_nd, sizeof(uint64_t));
      std::cout << "_partition_num: " << _partition_number << " C: " << C << " _nd: " << _nd << std::endl;
      for (unsigned i = 0; i < _partition_number; i++) {
        auto p = _partition[i];
        unsigned s = p.size();
        writer.write((char *)&s, sizeof(unsigned));
        writer.write((char *)p.data(), sizeof(unsigned) * s);
      }
      std::vector<unsigned> id2pidv(_nd);
      for (auto n : id2pid) {
        id2pidv[n.first] = n.second;
      }
      writer.write((char *)id2pidv.data(), sizeof(unsigned) * _nd);
    }

    /**
     * count the id overlap according to the graph partitioning
     */
    void partition_statistic() {
      std::vector<unsigned> overlap(_nd, 0);
      std::vector<unsigned> blk_neighbor_overlap(_partition_number, 0);
      double overlap_ratio = 0;

#pragma omp parallel for schedule(dynamic, 100) reduction(+ : overlap_ratio)
      for (size_t i = 0; i < _partition_number; i++) {
        std::unordered_set<unsigned> neighbors;
        unsigned blk_neighbor_num = 0;
        for (size_t j = 0; j < _partition[i].size(); j++) {
          blk_neighbor_num += full_graph[_partition[i][j]].size();
          std::unordered_set<unsigned> ne;
          for (unsigned &x : full_graph[_partition[i][j]]) {
            neighbors.insert(x);
            ne.insert(x);
          }
          blk_neighbor_overlap[i] = blk_neighbor_num - neighbors.size();
          for (size_t z = 0; z < _partition[i].size(); z++) {
            if (_partition[i][j] == _partition[i][z]) continue;
            if (ne.find(_partition[i][z]) != ne.end()) {
              overlap[_partition[i][j]]++;
            }
          }
          overlap_ratio +=
              (_partition[i].size() == 1 ? 0 : (1.0 * overlap[_partition[i][j]] / (_partition[i].size() - 1)));
        }
      }
      unsigned max_overlaps = 0;
      unsigned min_overlaps = std::numeric_limits<unsigned>::max();
      double ave_overlap_ratio = 0;
      std::map<unsigned, unsigned> overlap_count;
      for (size_t i = 0; i < _nd; i++) {
        if (overlap_count.count(overlap[i])) {
          overlap_count[overlap[i]]++;
        } else {
          overlap_count[overlap[i]] = 1;
        }
        if (overlap[i] > max_overlaps) max_overlaps = overlap[i];
        if (overlap[i] < min_overlaps) min_overlaps = overlap[i];
      }
      ave_overlap_ratio = overlap_ratio / (double)_nd;
      for (auto &it : overlap_count) {
        std::cout << "each id, overlap number " << it.first << ", count: " << it.second << std::endl;
      }
      std::cout << "each id, max overlaps: " << max_overlaps << std::endl;
      std::cout << "each id, min overlaps: " << min_overlaps << std::endl;
      std::cout << "each id, average overlap ratio: " << ave_overlap_ratio << std::endl;
    }

    unsigned select_partition(unsigned i) {
// #pragma omp atomic
      select_nums++;

      float maxn = 0.0;
      unsigned res = INF;
      std::unordered_map<unsigned, unsigned> pcount;
      unsigned tpid = 0;
      for (auto n : direct_graph[i]) {
        unsigned pid = id2pid[n];
        if (pid == INF) continue;
        pcount[pid] = pcount[pid] + 1;
        if (tpid < pid) {
          tpid = pid;
        }
      }
      for (auto n : reverse_graph[i]) {
        unsigned pid = id2pid[n];
        if (pid == INF) continue;
        pcount[pid] = pcount[pid] + 1;
        if (tpid < pid) {
          tpid = pid;
        }
      }
      for (auto c : pcount) {
        unsigned pid = c.first;
        float cnt = c.second;
        std::lock_guard<std::mutex> lock(*pmutex[pid]);
        double s = _partition[pid].size();
        cnt *= (1 - s / C);
        if (cnt > maxn && _partition[pid].size() < C) {
          res = pid;
          maxn = cnt;
        }
      }
      pcount.clear();
      if (res == INF) {
  // #pragma omp atomic
        select_free++;
        res = getUnfilled();
      }
      return res;
    }

    unsigned getUnfilled() {
// #pragma omp atomic
      getUnfilled_nums++;
      unsigned res;
      do {
        res = free_q.pop();
      } while (_partition[res].size() == C);
      return res;
    }

    // graph partition
    void graph_partition(const char *filename, int k, int lock_nums = 0) {
      for (unsigned i = 0; i < _nd; i++) {
        id2pid[i] = INF;
      }
      _partition.clear();
      _partition.resize(_partition_number);
      std::unordered_set<unsigned> vis;
      std::vector<unsigned> init_stream;
      init_stream.resize(_nd);
      std::iota(init_stream.begin(), init_stream.end(), 0);

      _lock_nodes.clear();
      _lock_pids.clear();
      _lock_nodes.resize(_nd, false);
      _lock_pids.resize(_partition_number, false);
      unsigned pid = 0;
      vis.clear();
      if (lock_nums) {
        std::cout << "lock first " << lock_nums << " nodes at init stage." << std::endl;
      }
      for (auto i : init_stream) {
        if (vis.count(i)) {
          lock_nums--;
          continue;  // has insert into partition
        }
        if (_partition[pid].size() == C) {
          ++pid;
        }
        vis.insert(i);
        _partition[pid].push_back(i);
        id2pid[i] = pid;
        if (lock_nums > 0) {
          _lock_pids[pid] = true;
        }
        for (unsigned s : full_graph[i]) {
          if (vis.count(s)) continue;
          if (_partition[pid].size() == C) {
            ++pid;
            break;
          }
          _partition[pid].push_back(s);
          id2pid[s] = pid;
          vis.insert(s);
        }
        if (lock_nums) --lock_nums;
      }
      int s = 0;
      for (unsigned i = 0; i < _partition_number; i++) {
        if (!_lock_pids[i]) break;
        for (unsigned s : _partition[i]) {
          _lock_nodes[s] = true;
        }
        s++;
      }
      if (_lock_pids[0]) {
        std::cout << "finally, it locks partition nums: " << s << " locks nodes num: " << s * C << std::endl;
      }

      std::cout << "init over." << std::endl;

      for (int i = 0; i < k; i++) {
        select_free = 0;
        graph_partition_LDG();
        std::cout << "select free: " << (double)select_free / _partition_number << std::endl;
        partition_statistic();
        auto ivf_file_name = std::string(filename) + std::string(".ivf") + std::to_string(i + 1);
        std::cout << "total ivf time: " << ivf_time << std::endl;
        save_partition(ivf_file_name.c_str());
      }
      save_partition(filename);
      std::cout << "select pid nums" << select_nums << " get unfilled partition nums: " << getUnfilled_nums << std::endl;
      std::cout << "total ivf time: " << ivf_time << std::endl;
    }

    void graph_partition_LDG() {
      while (!free_q.empty()) {
        free_q.pop();
      }
      
  #pragma omp parallel for
      for (unsigned i = 0; i < _partition_number; i++) {
        if (_lock_pids[i]) continue;
        _partition[i].clear();
        free_q.push(i);
      }

      cur = 0;
      std::cout << "start" << std::endl;
      std::vector<unsigned> stream(_nd);
      std::iota(stream.begin(), stream.end(), 0);
      auto rng = std::default_random_engine{};
      std::shuffle(std::begin(stream), std::end(stream), rng);
      auto start = omp_get_wtime();
  #pragma omp parallel for schedule(dynamic)
      for (unsigned i = 0; i < _nd; i++) {
        size_t n = stream[i];
        if (_lock_nodes[n]) continue;
        sync(n);
        cout_step();
      }
      auto end = omp_get_wtime();
      std::cout << "ivf time: " << end - start << " round: " << round << std::endl;
      ivf_time += end - start;
      round++;
    }

    unsigned sync(unsigned i) {
      unsigned pid = select_partition(i);
      pmutex[pid]->lock();

      while (_partition[pid].size() == C) {
        pmutex[pid]->unlock();
        pid = select_partition(i);
        pmutex[pid]->lock();
      }
      _partition[pid].emplace_back(i);
      id2pid[i] = pid;
      unsigned s = _partition[pid].size();
      pmutex[pid]->unlock();

      if (s != C) {
        free_q.push(pid);
      }

      return pid;
    }

    void cout_step() {
      if (!_visual) {
        return;
      }

// #pragma omp atomic
      cur++;
      if ((cur + 0) % cursize == 0) {
        std::cout << (double)(cur + 0) / _nd * 100 << "%    \r";
        std::cout.flush();
      }
    }
    
  private:
    size_t _dim;
    uint64_t _nd;
    uint64_t _max_node_len;
    unsigned _width;
    unsigned _ep;
    std::vector<std::vector<unsigned>> direct_graph;
    std::vector<std::vector<unsigned>> full_graph;
    unsigned select_free;
    uint64_t C;
    uint64_t _partition_number = 0;
    std::vector<std::vector<unsigned>> _partition{1000000};
    std::vector<std::unique_ptr<std::mutex>> pmutex;
    int cur = 0;
    std::vector<std::vector<unsigned>> reverse_graph;
    std::vector<std::vector<unsigned>> undirect_graph;
    std::unordered_map<unsigned, unsigned> id2pid;
    std::unordered_map<unsigned, unsigned> id2ratio;
    int round = 0;
    double ivf_time = 0.0;
    bool _visual = false;
    unsigned cursize = 10000;
    uint64_t select_nums = 0;
    uint64_t getUnfilled_nums = 0;
    uint64_t E;
    std::uniform_real_distribution<> *_dis;
    std::mt19937 *_gen;
    std::random_device *_rd;
    pipeann::ConcurrentQueue<unsigned> free_q;

    std::vector<bool> _lock_nodes;
    std::vector<bool> _lock_pids;
  };

} // namespace GP