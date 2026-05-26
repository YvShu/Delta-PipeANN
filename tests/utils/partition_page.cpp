/*
 * @Author: Guyue
 * @Date: 2026-05-08 17:55:46
 * @LastEditTime: 2026-05-12 18:00:05
 * @LastEditors: Guyue
 * @FilePath: /Delta-PipeANN/tests/utils/partition_page.cpp
 */
#include <utils/partition_page.h>

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "Correct usage: " << argv[0] << " <type[uint8/int8/float]> <file> <to_pts>" << std::endl;
    exit(-1);
  }

  int arg_no = 1;
  char *type = argv[arg_no++];
  char *index_file = argv[arg_no++];
  char *output_file = argv[arg_no++];

  unsigned ldg_times = 100;
  unsigned lock_nums = 0,
  omp_set_num_threads(8);
  GP::graph_partitioner partitioner(index_file, type);
  partitioner.graph_partition(output_file, ldg_times, lock_nums);
  
  return 0;
}