/*
 * @Author: Guyue
 * @Date: 2026-04-16 14:09:59
 * @LastEditTime: 2026-04-16 16:32:40
 * @LastEditors: Guyue
 * @FilePath: /Delta-PipeANN/src/search/pipe_search_blind.cpp
 */
#include "aligned_file_reader.h"
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>
#ifndef USE_AIO
#include "liburing.h"
#endif

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include "utils/timer.h"
#include "utils/tsl/robin_set.h"
#include "utils.h"
#include "utils/page_cache.h"

#include <unistd.h>
#include <sys/syscall.h>

namespace pipeann {
    // ==========================================
    // io_t 结构体: 纯盲搜版本不需要在 IO 阶段比较距离
    // ==========================================
    struct io_t {
        unsigned id;            // 发起该 IO 请求的目标节点 ID
        unsigned page_id;       // 物理页面 ID
        unsigned loc;           // 页内偏移
        IORequest *read_req;    // 异步 IO 请求指针

        bool finished() {
            return read_req->finished;
        }
    };

    template<typename T, typename TagT>
    size_t SSDIndex<T, TagT>::pipe_search_blind(const T *query1, const uint64_t k_search, const uint32_t mem_L,
                                                const uint64_t l_search, TagT *res_tags, float *distances,
                                                const uint64_t beam_width, QueryStats *stats, AbstractSelector *selector,
                                                const void *filter_data, const uint64_t relaxed_monotonicity_l) {
        QueryBuffer<T> *query_buf = pop_query_buf(query1);
    // #ifdef USE_AIO
        void *ctx = reader->get_ctx();
    // #else
    //     void *ctx = reader->get_ctx(IORING_SETUP_SQPOLL);
    // #endif
    
        if (beam_width > MAX_N_SECTOR_READS) {
            LOG(ERROR) << "Beamwidth can not be higher than MAX_N_SECTOR_READS";
            crash();
        }
        
        const T *query = query_buf->aligned_query_T;
        query_buf->reset();
        T *data_buf = query_buf->coord_scratch;
        _mm_prefetch((char *) data_buf, _MM_HINT_T1);
        char *sector_scratch = query_buf->sector_scratch;
        
        Timer query_timer;
        
        // 主候选集：存放【已经读取并算出距离】的节点
        std::vector<Neighbor> retset(l_search + 100);
        auto &visited = *(query_buf->visited);
        unsigned cur_list_size = 0;

        // 记录所有实际计算过距离节点，用于最终输出
        std::vector<Neighbor> full_retset;
        full_retset.reserve(l_search * 10);

        // 缓存已评估节点的邻居列表，避免将整个页面驻留内存
        std::unordered_map<unsigned, std::vector<unsigned>> nbr_cache;

        // 待抓取队列：存放已经被发现，但尚未发起 SSD IO 请求的节点 ID
        std::deque<unsigned> fetch_queue;
        // 进行中的 I/O 请求队列
        std::queue<io_t> on_flight_ios;
        // 全 IO 带宽
        int64_t cur_beam_width = beam_width;

        // 初始化统计指标
        if (stats != nullptr) {
            stats->io_us = 0;
            stats->cpu_us = 0;
            stats->n_ios = 0;
            stats->n_cmps = 0;
        }

        // Lambda 1: 距离计算
        auto compute_dists_and_push = [&](const LVQDiskNode<T> &node, const unsigned id) -> float {
            T *node_fp_coords_copy = data_buf;
            
            // 解码
            std::vector<float> point(meta_.data_dim);
            for (size_t i = 0; i < meta_.data_dim; ++i) {
                uint8_t q_val = node.coords[i];
                point[i] = (static_cast<float>(q_val) * node.step) + node.minval;
            }
            memcpy(node_fp_coords_copy, point.data(), meta_.data_dim * sizeof(T));
            float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);

            if (selector == nullptr || selector->is_member(id, filter_data, node.labels)) {
                full_retset.push_back(Neighbor(id, cur_expanded_dist, true));
            }
            if (stats != nullptr) stats->n_cmps++;
            return cur_expanded_dist;
        };

        // Lambda 2: 发送异步读取请求
        auto send_read_req = [&](unsigned id) -> bool {
            this->lock_idx(idx_lock_table, id, std::vector<uint32_t>(), true);
            const unsigned loc = id2loc(id), pid = loc_sector_no(loc);

            uint64_t &cur_buf_idx = query_buf->sector_idx;
            auto buf = sector_scratch + cur_buf_idx * size_per_io;
            auto &req = query_buf->reqs[cur_buf_idx];

            req = IORequest(static_cast<uint64_t>(pid) * SECTOR_LEN, size_per_io, buf, u_loc_offset(loc), meta_.max_node_len, sector_scratch);
            reader->send_read_no_alloc(req, ctx);

            on_flight_ios.push(io_t{id, pid, loc, &req});
            cur_buf_idx = (cur_buf_idx + 1) % MAX_N_SECTOR_READS;

            if (stats != nullptr) stats->n_ios++;
            return true;
        };

        // Lambda 3: 轮询 IO 结果，计算真实距离，并将邻居关系暂存
        auto poll_all = [&]() -> int {
            reader->poll_all(ctx);
            int n_finished = 0;

            while (!on_flight_ios.empty() && on_flight_ios.front().finished()) {
                io_t &io = on_flight_ios.front();
                LVQDiskNode<T> node = lvqnode_from_page((char *) io.read_req->buf, io.loc);
                unsigned id = io.id;

                // 1、进行距离计算
                float dist = compute_dists_and_push(node, id);
                LOG(INFO) << dist;

                // 2、将节点放入已评估候选集(Neighbor.flag = true表示已评估但尚未展开)
                Neighbor nn(id, dist, true);
                InsertIntoPool(retset.data(), cur_list_size, nn);
                if (cur_list_size < l_search) {
                    ++cur_list_size;
                }

                // 3、提取并缓存其邻居数组，以供未来展开使用
                std::vector<unsigned> nbrs(node.nbrs, node.nbrs + node.nnbrs);
                nbr_cache[id] = std::move(nbrs);

                // IO读取处理完毕，解锁
                this->unlock_idx(idx_lock_table, id);
                on_flight_ios.pop();
                n_finished++;
            }
            return n_finished;
        };

        // Lambda 4: 获取候选集中最好的且尚未展开的节点
        auto get_best_unexpanded = [&]() -> int {
            for (unsigned i = 0; i < cur_list_size; ++i) {
                if (retset[i].flag) {   // flag 为 true 代表该节点尚未作为跳板扩散
                    return i;
                }
            }
            return -1;
        };

        // ======== 起步阶段 ========
        // 盲搜模式下废弃了基于内存的 PQ 入口节点，直接将原始入口点塞入待抓取队列
        visited.insert(meta_.entry_point_id);
        fetch_queue.push_back(meta_.entry_point_id);

        // ======== 核心盲搜流水线 ========
        while (true) {
            // 1、主动填充流水线
            // 如果待抓取队列没满，主动从已知结果中挑选最好的节点进行扩展
            while (fetch_queue.size() < (long unsigned) cur_beam_width) {
                int expand_idx = get_best_unexpanded();

                // 终止条件：没有可以展开的节点或者最优的节点已经在搜索深度L之外
                if (expand_idx == -1 || expand_idx >= (int) l_search) {
                    break;
                }
                retset[expand_idx].flag = false; // 标记为已展开
                unsigned expand_id = retset[expand_idx].id;

                // 将该节点的邻居推入抓取队列(等待SSD读取并验证)
                for (unsigned nbr : nbr_cache[expand_id]) {
                    if (visited.find(nbr) == visited.end()) {
                        visited.insert(nbr);
                        fetch_queue.push_back(nbr);
                    }
                }
                // 释放该节点的缓存以节省内存
                nbr_cache.erase(expand_id);
            }
            LOG(INFO) << "填充流水";
            
            // 2、下发 I/O 请求
            // 只要抓取队列未满且有节点待抓取，就下发SSD读取请求
            while (on_flight_ios.size() < cur_beam_width && !fetch_queue.empty()) {
                unsigned next_id = fetch_queue.front();
                fetch_queue.pop_front();
                send_read_req(next_id);
            }
            LOG(INFO) << "I/O请求";

            // 3、退出判定
            // 如果没有任何正在进行的 I/O，且抓取队列为空，说明搜索彻底收敛
            if (on_flight_ios.empty()) {
                break;
            }

            // 4、轮询并回收 I/O 结果
            // 异步处理完毕的节点会被放入 retset, 重新参与下一轮的排序与展开判定
            poll_all();
            LOG(INFO) << "轮询处理";
        }

        // ======== 清尾阶段 ========
        // 将所有获取过距离的节点按距离升序排序
        std::sort(full_retset.begin(), full_retset.end(), [](const Neighbor &left, const Neighbor &right) {
            return left < right;
        });

        // 提取最终的 Top-K 结果
        uint64_t t = 0;
        for (uint64_t i = 0; i < full_retset.size() && t < k_search; ++i) {
            if (i > 0 && full_retset[i].id == full_retset[i - 1].id) {
                continue; // 去重
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