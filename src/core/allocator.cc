#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    // size_t Allocator::alloc(size_t size)
    // {
    //     IT_ASSERT(this->ptr == nullptr);
    //     // pad the size to the multiple of alignment
    //     size = this->getAlignedSize(size);

    //     // =================================== 作业 ===================================
    //     // TODO: 设计一个算法来分配内存，返回起始地址偏移量
    //     // =================================== 作业 ===================================
    //     size_t offset = 0;

    //     // iterate through free blocks (first-fit / best-fit)
    //     auto it = freeBlocks.lower_bound(BlockInfo(0, size)); // find first block >= size
    //     if (it != freeBlocks.end()) {
    //         BlockInfo block = *it;
    //         offset = block.addr;

    //         // remove old block
    //         freeBlocks.erase(it);
    //         addrInfo.erase(block.addr);

    //         // split if block is larger than needed
    //         if (block.size > size) {
    //             BlockInfo newBlock(block.addr + size, block.size - size, offset, block.nextAddr);
    //             if (block.nextAddr != SIZE_MAX) {
    //                 addrInfo[block.nextAddr].prevAddr = newBlock.addr;
    //             } else {
    //                 tailBlock = &newBlock;
    //             }
    //             freeBlocks.insert(newBlock);
    //             addrInfo[newBlock.addr] = newBlock;
    //         } else {
    //             // update prev/next of neighbors if the block is removed
    //             if (block.prevAddr != SIZE_MAX) {
    //                 addrInfo[block.prevAddr].nextAddr = block.nextAddr;
    //             }
    //             if (block.nextAddr != SIZE_MAX) {
    //                 addrInfo[block.nextAddr].prevAddr = block.prevAddr;
    //             } else {
    //                 tailBlock = &addrInfo[block.prevAddr];
    //             }
    //         }            
    //     } else {
    //         // no suitable free block found in freeBlocks
    //         if (tailBlock && tailBlock->addr + tailBlock->size == peak) {
    //             // Merge new allocation with the current tail block
    //             freeBlocks.erase(*tailBlock);
    //             tailBlock->size += size;
    //             freeBlocks.insert(*tailBlock);
    //             offset = tailBlock->addr;
    //             // peak doesn't need to be increased since it's already included
    //         } else {
    //             // Extend peak: create a new block at the end
    //             offset = peak;
    //             BlockInfo newBlock(peak, size, tailBlock ? tailBlock->addr : SIZE_MAX, SIZE_MAX);
    //             if (tailBlock) {
    //                 tailBlock->nextAddr = newBlock.addr;  // link previous tail
    //             }
    //             addrInfo[newBlock.addr] = newBlock;
    //             tailBlock = &addrInfo[newBlock.addr];    // update tailBlock pointer
    //             peak += size;                             // move peak to include new allocation
    //         }
    //     }

    //     used += size;
    //     return offset;     
    // }

    // void Allocator::free(size_t addr, size_t size)
    // {
    //     IT_ASSERT(this->ptr == nullptr);
    //     size = getAlignedSize(size);

    //     // =================================== 作业 ===================================
    //     // TODO: 设计一个算法来回收内存
    //     // =================================== 作业 ===================================
    //     BlockInfo newBlock(addr, size, SIZE_MAX, SIZE_MAX);

    //     auto it = freeBlocks.lower_bound(addr);

    // }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================

        size_t retAddr;
        auto it = freeBlocks.lower_bound(BlockInfo(size));

        // 如果找到符合条件的空闲块
        if (it != freeBlocks.end()) {
            BlockInfo bestFitBlock = *it;
            retAddr = bestFitBlock.addr_;

            // 删除旧块  
            freeBlocks.erase(it);
            blockStartToSize.erase(bestFitBlock.addr_);
            blockEndToSize.erase(bestFitBlock.addr_ + bestFitBlock.size_);

            // 如果剩下碎片
            if (bestFitBlock.size_ > size){
                // 更新空闲块列表
                BlockInfo remainBlock = {bestFitBlock.addr_ + size, bestFitBlock.size_ - size};
                freeBlocks.insert(remainBlock);
                blockStartToSize[remainBlock.addr_] = remainBlock.size_;
                blockEndToSize[remainBlock.addr_ + remainBlock.size_] = remainBlock.size_;
            }

            // 更新已使用内存
            used += size;

            // 返回分配的起始地址
            return retAddr;
        }
        else {
            // 没有找到大小够的空闲块，尝试扩展内存池
            // 如果尾块存在，要合并
            auto itEnd = blockEndToSize.find(peak);
            if (itEnd != blockEndToSize.end()) {
                size_t endBlockSize = itEnd->second;

                // 更新freeBlocks
                BlockInfo endBlock(peak - endBlockSize, endBlockSize);
                freeBlocks.erase(endBlock);

                // 更新 hash map
                blockStartToSize.erase(endBlock.addr_);
                blockEndToSize.erase(peak);

                // Update used & peak, just margin
                used += size - endBlockSize;
                peak += size - endBlockSize;

                // retAddr
                retAddr = endBlock.addr_;
            } else {
                // retAddr
                retAddr = peak;

                // Update used & peak 
                used += size;
                peak += size;
            }
        }
        return retAddr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================

        BlockInfo newBlock(addr, size);

        // 如果有邻接前块，合并并删除前块
        auto itAdjBlock = blockEndToSize.find(addr);
        if (itAdjBlock != blockEndToSize.end()) {
            // 拓展 newBlock 空间
            newBlock.addr_ -= itAdjBlock->second;
            newBlock.size_ += itAdjBlock->second;

            // 删除 prev block
            freeBlocks.erase(BlockInfo(itAdjBlock->first - itAdjBlock->second,
                                             itAdjBlock->second));
            blockStartToSize.erase(itAdjBlock->first - itAdjBlock->second);
            blockEndToSize.erase(itAdjBlock->first);
        }

        // 如果有邻接后块，合并并删除后块
        itAdjBlock = blockStartToSize.find(newBlock.addr_ + newBlock.size_);
        if (itAdjBlock != blockStartToSize.end()) {
            newBlock.size_ += itAdjBlock->second;

            freeBlocks.erase(BlockInfo(itAdjBlock->first, itAdjBlock->second));
            blockStartToSize.erase(itAdjBlock->first);
            blockEndToSize.erase(itAdjBlock->first + itAdjBlock->second);
        }

        freeBlocks.emplace(newBlock);
        blockStartToSize.emplace(newBlock.addr_, newBlock.size_);
        blockEndToSize.emplace(newBlock.addr_ + newBlock.size_, newBlock.size_);
    }


    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
