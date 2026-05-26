// /*
//  * @Author: Guyue
//  * @Date: 2026-04-27 11:18:38
//  * @LastEditTime: 2026-04-27 16:55:14
//  * @LastEditors: Guyue
//  * @FilePath: /Delta-PipeANN/src/update/direct_delete_lvq.cpp
//  */
// #include "aligned_file_reader.h"
// #include "utils/libcuckoo/cuckoohash_map.hh"
// #include "ssd_index.h"
// #include <malloc.h>
// #include <algorithm>

// #include <omp.h>
// #include <chrono>
// #include <cmath>
// #include <cstdint>
// #include <limits>
// #include <tuple>
// #include "utils/timer.h"
// #include "utils/tsl/robin_map.h"
// #include "utils.h"
// #include "utils/page_cache.h"
// #include "utils/prune_neighbors.h"

// #include <unistd.h>
// #include <sys/syscall.h>
// #include "linux_aligned_file_reader.h"

// namespace pipeann {
//   template<typename T, typename TagT>
//   int SSDIndex<T, TagT>::delete_lvq_in_place(const TagT &tag, const uint32_t l, const uint32_t k, const uint32_t c,  tsl::robin_set<uint32_t> *deletion_set) {
//     // ===================================================================
//     // 1、定位待删除点 p
//     // ===================================================================

//     // 查找tag对应的ID
//     uint32_t p_id = std::numeric_limits<uint32_t>::max();
//     auto lt = tags.lock_table();
//     for (const auto &kv : lt) {
//       if (kv.second == tag) {
//         p_id = kv.first;
//         break;
//       }
//     }
//     if (p_id == std::numeric_limits<uint32_t>::max()) {
//       LOG(ERROR) << "Delete failed: tag not found.";
//       return -1;
//     }
//     if (deletion_set && deletion_set->find(p_id) != deletion_set->end()) {
//       LOG(INFO) << "Point" << p_id << "already deleted.";
//       return -1;
//     }

//     // 获取p的向量
//     QueryBuffer<T> *read_data = this->pop_query_buf(nullptr);
//     void *ctx = reader->get_ctx();
//     T *point = read_data->aligned_query_T;
//     uint64_t p_loc = id2loc(p_id);
//     uint64_t p_sector = loc_sector_no(p_loc);
//     char *buf = nullptr;
//     alloc_aligned((void **)&buf, size_per_io, SECTOR_LEN);
//     std::vector<IORequest> reqs;
//     reqs.push_back(IORequest(p_sector * SECTOR_LEN, size_per_io, buf, 0, 0));    
//     reader->read(reqs, ctx, false);

//     LVQDiskNode<T> pnode = lvqnode_from_page(buf, p_loc);
//     for (size_t d = 0; d < meta_.data_dim; ++d) {
//       point[d] = static_cast<T>(pnode.coords[d]) * pnode.step + pnode.minval;
//     }

//     // 保存p的原始出边邻居(N_out(p))
//     std::vector<uint32_t> out_neighbors(pnode.nbrs, pnode.nbrs + pnode.nnbrs);
//     aligned_free(buf);

//     // ===================================================================
//     // 2、搜索获得Candidates和Visited集
//     //    do_ssd_search 回填 exp_node_info(候选)和coord_map(访问过的节点坐标)
//     // ===================================================================
//     std::vector<Neighbor> exp_node_info;
//     tsl::robin_map<uint32_t, T *> coord_map;
//     coord_map.reserve(2 * this->params.L);
//     std::vector<uint64_t> page_ref{};

//     // 执行SSD搜索以搜集candidates和coord_map(相当于visited集合的坐标缓存)
//     this->do_ssd_search(point, 0, l, this->params.beam_width, exp_node_info, &coord_map, 
//                         nullptr, deletion_set, false, &page_ref, read_data);
//     std::vector<uint32_t> candidates;
//     for (size_t i = 0; i < std::min((size_t)k, exp_node_info.size()); ++i) {
//       uint32_t cid = exp_node_info[i].id;
//       if (cid != p_id && (!deletion_set || deletion_set->find(cid) == deletion_set->end())) {
//         candidates.push_back(cid);
//       }
//     }

//     std::vector<uint32_t> visited_ids;
//     for (auto &kv : coord_map) {
//       if (kv.first != p_id && (!deletion_set || deletion_set->find(kv.first) == deletion_set->end())) {
//         visited_ids.push_back(kv.first);
//       }
//     }

//     // ===================================================================
//     // 3、确定近似入邻居 N‘_in(p)
//     //    遍历 visited 节点，检查其邻居列表是否包含 p
//     // ===================================================================
//     std::set<uint64_t> pages_to_read;

//     // 收集需要读取的扇区(visited 节点所在的页面)
//     for (uint32_t vid : visited_ids) {
//       pages_to_read.insert(loc_sector_no(id2loc(vid)));
//     }

//     // p 自己所在的页
//     pages_to_read.insert(loc_sector_no(id2loc(p_id)));

//     // 读取页面并建立映射
//     std::unordered_map<uint64_t, char *> page_buf_map;
//     std::vector<IORequest> read_reqs;
//     for (auto &sector : pages_to_read) {
//       char *b = nullptr;
//       alloc_aligned((void **)&b, size_per_io, SECTOR_LEN);
//       read_reqs.push_back(IORequest(sector * SECTOR_LEN, size_per_io, b, 0, 0));
//       page_buf_map[sector] = b;
//     }
//     reader->read(read_reqs, ctx, false);

//     // 从页面中提取邻居信息，找出真实入邻居
//     std::vector<uint32_t> approx_in_neighbors;
//     for (uint32_t vid : visited_ids) {
//       uint64_t vloc = id2loc(vid);
//       uint64_t vsector = loc_sector_no(vloc);
//       LVQDiskNode<T> vnode = lvqnode_from_page(page_buf_map[vsector], vloc);
//       for (uint32_t j = 0; j < vnode.nnbrs; ++j) {
//         if (vnode.nbrs[j] == p_id) {
//           approx_in_neighbors.push_back(vid);
//           break;
//         }
//       }
//     }

//     // ===================================================================
//     // 4、图修复——更新入邻居：z ∈ N'_in(p) 
//     //    C_z ← closest-c 的 Candidates 点加到 z 的邻居，移除 p
//     // ===================================================================

//     // 为了后续剪枝，需要缓存坐标
//     std::unordered_map<uint32_t, const T *> node_coords_cache;
//     std::vector<std::vector<T>> coord_pool;

//     // 记录所有需要写回的页面(修改过的)
//     std::set<uint64_t> dirty_sectors;
//     for (uint32_t z : approx_in_neighbors) {
//       if (deletion_set && deletion_set->find(z) != deletion_set->end()) {
//         continue;
//       }

//       // 计算 z 到各候选点的距离
//       std::vector<Neighbor> z_candidates;
//       for (uint32_t cid : candidates) {
//         if (cid == z) continue;
//         float dist = this->dist_cmp->compare(coord_map[z], coord_map[cid], (unsigned) aligned_dim);
//         z_candidates.emplace_back(cid, dist);
//       }
//       std::sort(z_candidates.begin(), z_candidates.end());
//       size_t num_add = std::min((size_t)c, z_candidates.size());

//       // 读取z的邻居列表
//       uint64_t zloc = id2loc(z);
//       uint64_t zsector = loc_sector_no(zloc);
//       LVQDiskNode<T> znode = lvqnode_from_page(page_buf_map[zsector], zloc);
//       std::vector<uint32_t> new_nbrs(znode.nbrs, znode.nbrs + znode.nnbrs);

//       // 移除 p
//       new_nbrs.erase(std::remove(new_nbrs.begin(), new_nbrs.end(), p_id), new_nbrs.end());

//       // 添加C_z中的点(不去重，后续prune会处理)
//       for (size_t i = 0; i < num_add; ++i) {
//         uint32_t cid = z_candidates[i].id;
//         if (std::find(new_nbrs.begin(), new_nbrs.end(), cid) == new_nbrs.end()) {
//           new_nbrs.push_back(cid);
//         }
//       }

//       // 写回邻居
//       znode.nnbrs = (uint32_t)new_nbrs.size();
//       memcpy(znode.nbrs, new_nbrs.data(), new_nbrs.size() * sizeof(uint32_t));
//       dirty_sectors.insert(zsector);
//     }

//     // ===================================================================
//     // 5、图修复——更新p指向的节点(出邻居w)，添加反向边
//     // ===================================================================
//     for (uint32_t w : out_neighbors) {
//       if (deletion_set && deletion_set->find(w) != deletion_set->end()) {
//         continue;
//       }

//       // 计算w到各候选点的距离
//       std::vector<Neighbor> w_candidates;
//       for (uint32_t cid : candidates) {
//         if (cid == w) {
//           continue;
//         }
//         float dist = this->dist_cmp->compare(coord_map[w], coord_map[cid], (unsigned) aligned_dim);
//         w_candidates.emplace_back(cid, dist);
//       }
//       std::sort(w_candidates.begin(), w_candidates.end());
//       size_t num_add = std::min((size_t)c, w_candidates.size());

//       // 对于每个 y ∈ C_w，将 w 添加到 y 的邻居
//       for (size_t i = 0; i < num_add; ++i) {
//         uint32_t y = w_candidates[i].id;
//         uint64_t yloc = id2loc(y);
//         uint64_t ysector = loc_sector_no(yloc);
//         LVQDiskNode<T> ynode = lvqnode_from_page(page_buf_map[ysector], yloc);
//         std::vector<uint32_t> y_nbrs(ynode.nbrs, ynode.nbrs + ynode.nnbrs);
//         if (std::find(y_nbrs.begin(), y_nbrs.end(), w) == y_nbrs.end()) {
//           y_nbrs.push_back(w);
//           ynode.nnbrs = (uint32_t)y_nbrs.size();
//           memcpy(ynode.nbrs, y_nbrs.data(), y_nbrs.size() * sizeof(uint32_t));

//           dirty_sectors.insert(ysector);
//         }
//       }
//     }

//     // ===================================================================
//     // 6、删除p：清空邻居，加入delete_set, 移除tag映射
//     // ===================================================================    
//     pnode.nnbrs = 0;
//     memcpy(pnode.nbrs, &pnode.nnbrs, sizeof(uint32_t));
//     dirty_sectors.insert(p_sector);
//     if (deletion_set) {
//       deletion_set->insert(p_id);
//     }
//     tags.erase(p_id);

//     // ===================================================================
//     // 7、对度数超过限制的节点执行剪枝
//     //    R为最大度数
//     // ===================================================================  
//     std::set<uint32_t> vertices_to_prune;
//     for (uint32_t z : approx_in_neighbors) {
//       vertices_to_prune.insert(z);
//     }
//     for (uint32_t cid : approx_in_neighbors) {
//       vertices_to_prune.insert(cid);
//     }

//     for (uint32_t v : vertices_to_prune) {
//       if (deletion_set && deletion_set->find(v) != deletion_set->end()) {
//         continue;
//       }
//       uint64_t vloc = id2loc(v);
//       uint64_t vsector = loc_sector_no(vloc);
//       LVQDiskNode<T> vnode = lvqnode_from_page(page_buf_map[vsector], vloc);
//       std::vector<uint32_t> vnbrs(vnode.nbrs, vnode.nbrs + vnode.nnbrs);
//       if (vnbrs.size() > params.R) {
//         std::vector<Neighbor> pool;
//         for (uint32_t nid : vnbrs) {
//           if (deletion_set && deletion_set->find(nid) != deletion_set->end()) {
//             continue;
//           }
//           if (coord_map.find(nid) == coord_map.end()) {
//             continue;
//           }

//           float dist = this->dist_cmp->compare(coord_map[v], coord_map[nid], (unsigned) aligned_dim);
//           pool.emplace_back(nid, dist);
//         }
//         std::vector<uint32_t> pruned;
//         pipeann::prune_neighbors(pool, pruned, params, metric, [&](uint32_t a, uint32_t b) {
//           return this->dist_cmp->compare(coord_map[a], coord_map[b], (unsigned) aligned_dim);
//         });
//         memcpy(vnode.nbrs, pruned.data(), pruned.size() * sizeof(uint32_t));
//         vnode.nnbrs = (uint32_t) pruned.size();
//         dirty_sectors.insert(vsector);
//       }
//     }

//     // ===================================================================
//     // 8、将所有脏页写回磁盘
//     // ===================================================================  
//     std::vector<IORequest> writes;
//     for (auto &sec : dirty_sectors) {
//         writes.push_back(IORequest(sec * SECTOR_LEN, size_per_io, page_buf_map[sec], 0, 0));
//     }
//     std::vector<uint64_t> write_ref;
//     reader->write(writes, ctx);
//     reader->deref(&write_ref, ctx);

//     // 清理临时缓冲区
//     for (auto &kv : page_buf_map) {
//         aligned_free(kv.second);
//     }
//     reader->deref(&page_ref, ctx);
//     this->push_query_buf(read_data);

//     LOG(INFO) << "In-place delete finished for tag " << tag
//               << " (ID: " << p_id << "), "
//               << approx_in_neighbors.size() << " in-neighbors repaired.";

//     return 0;
//   }
  
//   template class SSDIndex<float>;
//   template class SSDIndex<int8_t>;
//   template class SSDIndex<uint8_t>;
// } // namespace pipeann