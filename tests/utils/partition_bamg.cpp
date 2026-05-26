/*
 * @Author: Guyue
 * @Date: 2026-05-20 11:16:19
 * @LastEditTime: 2026-05-21 10:32:18
 * @LastEditors: Guyue
 * @FilePath: /Delta-PipeANN/tests/utils/partition_bamg.cpp
 */
#include <utils/partition_bamg.h>

int main(int argc, char** argv) {
  char* base_data_file = argv[1];
  char* index_file = argv[2];
  char* partition_file = argv[3];
  char* output_file = argv[4];
  int beta = std::atoi(argv[5]);

  GP::bamg_partitioner bamg_builder;

  bamg_builder.load_raw_data(base_data_file);

  bamg_builder.load_partitioned_graph<float>(index_file, partition_file);

  std::vector<std::vector<unsigned>> bamg = bamg_builder.build_BAMG(1.2, beta, "bamg.bin");

  bamg_builder.save_graph<float>(index_file, bamg, output_file);

  return 0;
}