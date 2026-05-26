for L in 1 5 10
do
  for Q in 4 10 16
  do
    build/tests/overall_performance float /mnt/hgfs/DataSet/sift-2M/sift_base.bin 60 \
    /mnt/hgfs/DataSet/sift-2M/Vamana48_page/Vamana \
    /mnt/hgfs/DataSet/sift-2M/sift_query.bin$Q \
    /mnt/hgfs/DataSet/sift-2M/gt$Q \
    10 $L 1000 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 \
    |& tee /home/guyue/Delta-PipeANN/output/1M_Q${Q}_R48_+memgraph0.01L${L}_-cache_+relayout_+pagese.txt
  done
done

# for R in 16 24 32 48
# do
#   for beta in 1 2 3 4
#   do
#     for Q in 4 10 16
#     do
#       build/tests/overall_performance float /mnt/hgfs/DataSet/sift-2M/sift_base.bin 60 \
#       /mnt/hgfs/DataSet/sift-2M/Vamana${R}_bamg${beta}/Vamana \
#       /mnt/hgfs/DataSet/sift-2M/sift_query.bin$Q \
#       /mnt/hgfs/DataSet/sift-2M/gt$Q \
#       10 10 1000 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250 260 270 280 290 300 \
#       |& tee /home/guyue/Delta-PipeANN/output/1M_Q${Q}_R${R}_+memgraph0.01L10_-cache_+relayout${beta}bamg_+pagese.txt
#     done
#   done
# done

# for R in 16 24 32 48
# do 
#   for beta in 1 2 3 4
#   do
#     build/tests/utils/partition_bamg /mnt/hgfs/DataSet/sift-2M/sift_base.bin1000000 \
#     /mnt/hgfs/DataSet/sift-2M/Vamana${R}_page/Vamana_disk.index \
#     /mnt/hgfs/DataSet/sift-2M/Vamana${R}_page/Vamana_partition.bin \
#     /mnt/hgfs/DataSet/sift-2M/Vamana${R}_bamg${beta}/Vamana_disk.index $beta
#     cp /mnt/hgfs/DataSet/sift-2M/Vamana${R}_page/Vamana_mem* /mnt/hgfs/DataSet/sift-2M/Vamana${R}_bamg${beta}/
#     cp /mnt/hgfs/DataSet/sift-2M/Vamana${R}_page/Vamana_partition.bin /mnt/hgfs/DataSet/sift-2M/Vamana${R}_bamg${beta}/
#     cp /mnt/hgfs/DataSet/sift-2M/Vamana${R}_page/Vamana_partition.bin.aligned /mnt/hgfs/DataSet/sift-2M/Vamana${R}_bamg${beta}/
#   done
# done