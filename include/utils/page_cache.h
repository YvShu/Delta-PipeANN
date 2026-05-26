#ifndef PAGE_CACHE_H_
#define PAGE_CACHE_H_

#include <cstring>
#include <cstdint>
#include <queue>
#include "ssd_index_defs.h"
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "utils/lock_table.h"

// namespace pipeann {
//   struct PageCache {
//     size_t capacity_;
//     // 预分配的连续内存块，消除 new/delete
//     std::vector<uint8_t> buffer_pool;
//     // 跟踪空闲的内存块索引
//     std::vector<uint32_t> free_indices;
    
//     struct CacheEntry {
//       uint32_t pool_idx;
//       uint64_t ref_cnt;
//     };
    
//     std::unordered_map<uint64_t, CacheEntry> cache;
//     std::queue<uint64_t> fifo_queue;

//     PageCache(size_t capacity = 1024) : capacity_(capacity) {
//       buffer_pool.resize(capacity * SECTOR_LEN);
//       for (uint32_t i = 0; i < capacity; ++i) {
//           free_indices.push_back(i);
//       }
//       cache.reserve(capacity); // 减少哈希表 rehash
//     }

//     bool get(uint64_t block_no, uint8_t *value, bool ref = false) {
//       auto it = cache.find(block_no);
//       if (it == cache.end()) return false;
      
//       // 直接内存拷贝，无回调，无锁
//       memcpy(value, &buffer_pool[it->second.pool_idx * SECTOR_LEN], SECTOR_LEN);
//       return true;
//     }

//     void put(uint64_t block_no, uint8_t *value, bool ref = false) {
//       auto it = cache.find(block_no);
//       if (it != cache.end()) {
//           memcpy(&buffer_pool[it->second.pool_idx * SECTOR_LEN], value, SECTOR_LEN);
//           return;
//       }

//       // 达到容量上限，执行淘汰
//       if (cache.size() >= capacity_) {
//           uint64_t victim = fifo_queue.front();
//           fifo_queue.pop();
//           auto victim_it = cache.find(victim);
//           if (victim_it != cache.end()) {
//               free_indices.push_back(victim_it->second.pool_idx);
//               cache.erase(victim_it);
//           }
//       }

//       // 使用预分配的内存
//       uint32_t idx = free_indices.back();
//       free_indices.pop_back();
      
//       memcpy(&buffer_pool[idx * SECTOR_LEN], value, SECTOR_LEN);
//       cache[block_no] = {idx, 0};
//       fifo_queue.push(block_no);
//     }

//     bool deref(uint64_t block_no) {
//       return true;
//     }

//     void clear() {
//     }

//   };
//   inline PageCache cache(4096);
// }
// #endif  // PAGE_CACHE_H_


// namespace pipeann {
//   struct PageCache {
//     size_t capacity_;
//     // 预分配连续内存，避免运行时的 new/delete 
//     std::vector<uint8_t> buffer_pool;
    
//     // 结合了元数据与双向链表指针的条目结构
//     struct CacheEntry {
//       uint32_t pool_idx;   // 指向 buffer_pool 块的索引 [cite: 23]
//       uint64_t block_no;   // 存储对应的 SSD 块号
//       int32_t prev = -1;   // 链表前驱索引
//       int32_t next = -1;   // 链表后继索引
//     };

//     std::vector<CacheEntry> entries;
//     std::vector<uint32_t> free_indices; // 管理可用的空闲条目 [cite: 22]
//     std::unordered_map<uint64_t, uint32_t> cache;
    
//     int32_t head = -1; // 链表头：指向最久未访问的数据 (LRU)
//     int32_t tail = -1; // 链表尾：指向最近刚访问的数据 (MRU)

//     PageCache(size_t capacity = 1024) : capacity_(capacity) {
//       buffer_pool.resize(capacity * SECTOR_LEN); // 
//       entries.resize(capacity);
//       cache.reserve(capacity); // 预留哈希表空间以减少 rehash 
      
//       for (uint32_t i = 0; i < capacity; ++i) { // [cite: 24]
//         entries[i].pool_idx = i;
//         free_indices.push_back(i); // [cite: 25]
//       }
//     }

//     // 内部私有辅助：将指定索引的条目移动到双向链表尾部（表示最新访问）
//     void touch(uint32_t idx) {
//       if (static_cast<int32_t>(idx) == tail) return;

//       // 1. 从当前位置移除
//       if (entries[idx].prev != -1) {
//         entries[entries[idx].prev].next = entries[idx].next;
//       } else if (static_cast<int32_t>(idx) == head) {
//         head = entries[idx].next;
//       }

//       if (entries[idx].next != -1) {
//         entries[entries[idx].next].prev = entries[idx].prev;
//       }

//       // 2. 插入到尾部 (tail)
//       entries[idx].prev = tail;
//       entries[idx].next = -1;
//       if (tail != -1) {
//         entries[tail].next = idx;
//       }
//       tail = idx;
//       if (head == -1) head = idx;
//     }

//     bool get(uint64_t block_no, uint8_t *value, bool ref = false) {
//       auto it = cache.find(block_no);
//       if (it == cache.end()) return false; // [cite: 26]

//       uint32_t entry_idx = it->second;
//       touch(entry_idx); // 提升优先级
      
//       // 直接内存拷贝，性能无损 
//       memcpy(value, &buffer_pool[entries[entry_idx].pool_idx * SECTOR_LEN], SECTOR_LEN);
//       return true; // [cite: 27]
//     }

//     void put(uint64_t block_no, uint8_t *value, bool ref = false) {
//       auto it = cache.find(block_no);
//       if (it != cache.end()) { // [cite: 28]
//         uint32_t entry_idx = it->second;
//         memcpy(&buffer_pool[entries[entry_idx].pool_idx * SECTOR_LEN], value, SECTOR_LEN); // [cite: 29]
//         touch(entry_idx);
//         return;
//       }

//       uint32_t entry_idx;
//       if (!free_indices.empty()) { // 如果还有未使用的内存块 [cite: 32]
//         entry_idx = free_indices.back();
//         free_indices.pop_back();
//       } else {
//         // 缓存已满，执行 LRU 淘汰：移除链表头部 (head) 对应的块 
//         entry_idx = head;
//         cache.erase(entries[entry_idx].block_no); // [cite: 31]
        
//         // 更新头指针
//         head = entries[entry_idx].next;
//         if (head != -1) entries[head].prev = -1;
//         else tail = -1;
//       }

//       // 填充新数据并映射
//       entries[entry_idx].block_no = block_no;
//       memcpy(&buffer_pool[entries[entry_idx].pool_idx * SECTOR_LEN], value, SECTOR_LEN); // 
//       cache[block_no] = entry_idx;
      
//       // 将新条目挂载到链表尾部
//       entries[entry_idx].prev = tail;
//       entries[entry_idx].next = -1;
//       if (tail != -1) entries[tail].next = entry_idx;
//       tail = entry_idx;
//       if (head == -1) head = entry_idx;
//     }

//     bool deref(uint64_t block_no) {
//       return true;
//     }

//     void clear() {
//     }
//   };
//   inline PageCache cache(4096);
// } // namespace pipeann
// #endif  // PAGE_CACHE_H_

namespace pipeann {
  // User-space page cache for update acceleration (in fact it's a buffer)
  // only used for write-write, ensure that disk has a consistent state
  // expect a lock-free read

  struct PageCacheItem {
    uint8_t *buf;
    uint64_t ref_cnt;

    // use lock!
    uint64_t ref() {
      return ++ref_cnt;
    }

    // use lock!
    uint64_t deref() {
      return --ref_cnt;
    }
  };

  struct PageCache {
    bool get(uint64_t block_no, uint8_t *value, bool ref = false) {
      bool ret = cache.update_fn(block_no, [&](PageCacheItem &v) {
        memcpy(value, v.buf, SECTOR_LEN);
        if (ref) {
          v.ref();
        }
      });
      return ret;
    }

    bool put(uint64_t block_no, uint8_t *value, bool ref = false) {
      return cache.upsert(block_no, [&](PageCacheItem &v, libcuckoo::UpsertContext ctx) {
        if (ctx == libcuckoo::UpsertContext::NEWLY_INSERTED) {
          v = PageCacheItem{.buf = new uint8_t[SECTOR_LEN], .ref_cnt = 0};
        }
        if (ref) {
          v.ref();
        }
        memcpy(v.buf, value, SECTOR_LEN);
      });
    }

    bool deref(uint64_t block_no) {
      bool ret = cache.uprase_fn(block_no, [&](PageCacheItem &v, libcuckoo::UpsertContext ctx) {
        if (ctx == libcuckoo::UpsertContext::NEWLY_INSERTED) {
          LOG(ERROR) << "PageCache: deref a non-exist block_no: " << block_no;
          return true;
          __builtin_trap();
        }
        uint64_t refs = v.deref();
        if (refs == 0) {
          delete[] v.buf;
        }
        return refs == 0;
      });
      return ret;
    }

    void clear() {
      cache.clear();
    }

    SparseLockTable<uint64_t> lock_table;
    libcuckoo::cuckoohash_map<uint64_t, PageCacheItem> cache;
  };

  inline PageCache cache;
}  // namespace pipeann

#endif  // PAGE_CACHE_H_