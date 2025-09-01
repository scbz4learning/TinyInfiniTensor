#pragma once
#include "core/runtime.h"
#include "core/tensor.h"
#ifdef BUILD_TEST
#include "gtest/gtest.h"
#endif
#include <cstddef>
#include <map>
#include <unordered_set>

namespace infini {
  class Allocator
  {
  private:
    Runtime runtime;

    size_t used;

    size_t peak;

    size_t alignment;

    // pointer to the memory actually allocated
    void *ptr;

    // =================================== 作业 ===================================
    // TODO：可能需要设计一个数据结构来存储free block，以便于管理和合并
    // HINT: 可以使用一个 map 来存储 free block，key 为 block 的起始/结尾地址，value 为 block 的大小
    // =================================== 作业 ===================================

    // 尝试了 first-fit，一个 map 的 best-fit 
    // 最后还是参考了 InfiniTensor 的实现。
    // 笔记见 https://scbz4learning.github.io/Infinitensor/Stage_1/5_TinyInfiniTensor.md
    
    // Finding a suitable block need O(n), as only linear search is available.
    // std::map<size_t, size_t> free_blocks; 
    
    // However, referring to the original project
    // Using an ordered set to sort the blocksize from smallest to largest
    //   can reduce the time in O(log n)
    // And the blockAddr can use the hash table with doubly linked list 
    //   to avoid maintain multiple datastructures

    // 这样也不行
    // 一个是必须要维护尾指针块信息，不然就没法知道扩容要不要merge
    // 一个是unorder_map 存储addrInfo是不能找到临近块的
    // map能，但是O(n)了
    // struct BlockInfo {
    //     size_t addr;       // 当前块的起始地址
    //     size_t size;       // 当前块的大小
    //     size_t prevAddr;   // 前一个块的起始地址，如果没有则设为 SIZE_MAX
    //     size_t nextAddr;   // 后一个块的起始地址，如果没有则设为 SIZE_MAX

    //     // unordered_map needs default constructer
    //     BlockInfo() 
    //     : addr(0), size(0), prevAddr(SIZE_MAX), nextAddr(SIZE_MAX) {}

    //     BlockInfo(size_t a, size_t s, size_t prev = SIZE_MAX, size_t next = SIZE_MAX)
    //         : addr(a), size(s), prevAddr(prev), nextAddr(next) {}

    //     // overload <
    //     bool operator<(const BlockInfo &other) const {
    //         if (size != other.size) return size < other.size;
    //         return addr < other.addr;
    //     }
    // };

    // // balanced tree ordered by size in descending order
    // std::set<BlockInfo> freeBlocks;
    // // hash map to find block info by addr
    // // key addr
    // // value blockInfo
    // std::unordered_map<size_t, BlockInfo> addrInfo;

    // // add a tailblock pointer for merging when increasing peak
    // BlockInfo* tailBlock = nullptr;

    struct BlockInfo {
        size_t addr_;
        size_t size_;

        BlockInfo() : addr_(0), size_(0){}
        BlockInfo(size_t s) : addr_(0), size_(s){}
        BlockInfo(size_t a, size_t s): addr_(a), size_(s){}

        bool operator<(const BlockInfo &other) const {
            return (size_ != other.size_) ? (size_ < other.size_) : (addr_ < other.addr_);
        }
    };

    // Balanced tree for free memory blocks, sorted by size and address
    std::set<BlockInfo> freeBlocks;

    // Key: Starting address of the free memory block
    // Value: Size of the block
    std::unordered_map<size_t, size_t> blockStartToSize;

    // Key: Ending address of the free memory block
    // Value: Size of the block
    std::unordered_map<size_t, size_t> blockEndToSize;

  public:
    Allocator(Runtime runtime);

    virtual ~Allocator();

    // function: simulate memory allocation
    // arguments：
    //     size: size of memory block to be allocated
    // return: head address offset of the allocated memory block
    size_t alloc(size_t size);

    // function: simulate memory free
    // arguments:
    //     addr: head address offset of memory block to be free
    //     size: size of memory block to be freed
    void free(size_t addr, size_t size);

    // function: perform actual memory allocation
    // return: pointer to the head address of the allocated memory
    void *getPtr();

    void info();

  private:
    // function: memory alignment, rouned up
    // return: size of the aligned memory block
    size_t getAlignedSize(size_t size);
  };
}
