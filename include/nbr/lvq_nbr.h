/*
 * @Author: Guyue
 * @Date: 2026-04-15 17:09:30
 * @LastEditTime: 2026-04-15 18:55:56
 * @LastEditors: Guyue
 * @FilePath: /Delta-PipeANN/include/nbr/lvq_nbr.h
 */
#pragma once

#include "utils/libcuckoo/cuckoohash_map.hh"
#include "utils.h"
#include <immintrin.h>
#include "nbr/pq_table.h"
#include "ssd_index_defs.h"
#include "nbr/abstract_nbr.h"
#include "utils/lock_table.h"
#include "utils/tsl/robin_map.h"
#include "utils/cached_io.h"
#include "utils/partition.h"
#include "utils/kmeans_utils.h"

namespace pipeann {
    template<typename T>
    class LVQNeighbor : public AbstractNeighbor<T> {
        public:
            LVQNeighbor(pipeann::Metric metric) : AbstractNeighbor<T>(metric) {
        }
        
        std::string get_name() {
           return "LVQNeighbor";
        }
        
        void quantize_single(float* point, unsigned dim, uint8_t* out, float& min_val, float& step) {
            // 1、记录最大最小值
            min_val = point[0];
            float max_val = point[0];
            for (size_t i = 1; i < dim; ++i)
            {
                if (point[i] < min_val) min_val = point[i];
                if (point[i] > max_val) max_val = point[i];          
            }
            
            // 2、计算步长
            step = 0.0f;
            step = (max_val - min_val) / 255.0f;

            // 3、量化每个维度的数据
            for (size_t i = 0; i < dim; ++i)
            {
                int p_val = std::round((point[i] - min_val) / step);
                p_val = std::max(0, std::min(255, p_val));
                out[i] = static_cast<uint8_t>(p_val);
            }

            // 4、将min_val和step写入内存布局的尾部
            size_t offset = dim;
            std::memcpy(out + offset, &min_val, sizeof(float));
            offset += sizeof(float);
            std::memcpy(out + offset, &step, sizeof(float));
        }  
    };
} // namespace pipeann