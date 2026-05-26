/*
 * @Author: Guyue
 * @Date: 2026-05-09 10:15:56
 * @LastEditTime: 2026-05-09 11:14:07
 * @LastEditors: Guyue
 * @FilePath: /Delta-PipeANN/tests/utils/index_relayout.cpp
 */
#include <chrono>
#include <string>
#include <utils.h>
#include <memory>
#include <set>
#include <vector>
#include <iostream>
#include <fstream>
#include <limits>
#include <cstring>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <utility>
#include <omp.h>
#include <cmath>
#include <mutex>
#include <queue>
#include <random>
#include "ssd_index.h"

#define READ_SECTOR_OFFSET(node_id) \
  ((uint64_t) node_id / nnodes_per_sector  + 1) * SECTOR_LEN + ((uint64_t) node_id % nnodes_per_sector) * max_node_len;
#define INF 0xffffffff

const std::string partition_index_filename = "_tmp.index";

template<typename T>
void relayout(const char* index_name, const char* partition_name) {
  uint64_t C;
  uint64_t _partition_nums;
  uint64_t _nd;
  uint64_t max_node_len;
  std::vector<std::vector<unsigned>> layout;
  std::vector<std::vector<unsigned>> _partition;

  std::ifstream part(partition_name);
  part.read((char*) &C, sizeof(uint64_t));
  part.read((char*) &_partition_nums, sizeof(uint64_t));
  part.read((char*) &_nd, sizeof(uint64_t));
  std::cout << "C: " << C << " partition_nums: " << _partition_nums
            << " nd: " << _nd << std::endl;

  std::ifstream in;
  in.open(index_name, std::ios::binary);
  pipeann::SSDIndexMetadata<T> meta;
  meta.load_from_disk_index(in);
  in.close();

  max_node_len = meta.max_node_len;
  unsigned nnodes_per_sector = meta.nnodes_per_sector;

  layout.resize(_partition_nums);
  for (unsigned i = 0; i < _partition_nums; ++i) {
    unsigned s;
    part.read((char*) &s, sizeof(unsigned));
    layout[i].resize(s);
    part.read((char*) layout[i].data(), sizeof(unsigned) * s);
  }
  part.close();

  uint64_t read_blk_size = 64 * 1024 * 1024;
  uint64_t write_blk_size = read_blk_size;
  std::string partition_path(partition_name);
  partition_path = partition_path.substr(0, partition_path.find_last_of('.')) + partition_index_filename;
  cached_ofstream diskann_writer(partition_path, write_blk_size);

  std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(SECTOR_LEN);
  std::unique_ptr<char[]> node_buf = std::make_unique<char[]>(max_node_len);

  std::cout << "nnodes per sector" << nnodes_per_sector << std::endl;
  
  uint64_t file_size = SECTOR_LEN + SECTOR_LEN * ((_nd + nnodes_per_sector - 1) / nnodes_per_sector);
  std::unique_ptr<char[]> mem_index = std::make_unique<char[]>(file_size);
  std::ifstream diskann_reader(index_name);
  diskann_reader.read(mem_index.get(), file_size);
  std::cout << "C: " << " partition_nums: " << _partition_nums
            << " nd: " << _nd << std::endl;
  
  const uint64_t disk_file_size = _partition_nums * SECTOR_LEN + SECTOR_LEN;
  std::cout << "size: " << disk_file_size << std::endl;
  
  memset(sector_buf.get(), 0, SECTOR_LEN);
  diskann_writer.write(sector_buf.get(), SECTOR_LEN);

  for (unsigned i = 0; i < _partition_nums; ++i) {
    if (i % 100000 == 0) {
      std::cout << "relayout has done " << (float) i / _partition_nums
                    << std::endl;
      std::cout.flush();
    }

    memset(sector_buf.get(), 0, SECTOR_LEN);
    for (unsigned j = 0; j < layout[i].size(); ++j) {
      unsigned id = layout[i][j];
      memset(node_buf.get(), 0, max_node_len);
      uint64_t index_offset = READ_SECTOR_OFFSET(id);
      uint64_t buf_offset = (uint64_t)j * max_node_len;
      memcpy((char*) sector_buf.get() + buf_offset,
              (char*) mem_index.get() + index_offset, max_node_len);
    }
    diskann_writer.write(sector_buf.get(), SECTOR_LEN);
  }
  diskann_writer.close();
  meta.save_to_disk_index(partition_path);
  std::cout << "Relayout index." << std::endl;
}

int main(int argc, char** argv) {
  char* indexName = argv[1];
  char* partitionName = argv[2];

  relayout<float>(indexName, partitionName);

  return 0;
}