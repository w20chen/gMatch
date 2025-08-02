#ifndef MEM_POOL_H
#define MEM_POOL_H


#include <cstdint>
#include <assert.h>
#include <cstdlib>
#include "helper.h"
#include "params.h"


#define memPoolBlockIntNum (memPoolBlockSize / sizeof(int))         // # of integers within a block


class MemPool {
    int *head;              // starting address of memory pool
    int *nextAddrBound;     // upper bound of nextAddr
    int *nextAddr;          // address of next available block

public:
    MemPool() {
        head = nullptr;
        nextAddrBound = nullptr;
        nextAddr = nullptr;
    }

    void init() {
        cudaCheck(cudaMalloc(&head, memPoolBlockNum * memPoolBlockSize));
        printf("--- Allocating %d bytes (%d int) for head @ %p\n", memPoolBlockNum * memPoolBlockSize, memPoolBlockNum * memPoolBlockSize, head);
        cudaCheck(cudaMemset(head, -1, memPoolBlockNum * memPoolBlockSize));
        nextAddr = head;
        nextAddrBound = head + memPoolBlockNum * memPoolBlockIntNum;
        print_meta(memPoolBlockNum);
    }

    void init(int partial_matching_cnt) {
        if (partial_matching_cnt == 0) {
            head = nullptr;
            nextAddr = nullptr;
            nextAddrBound = nullptr;
            return;
        }

        // Note: a partial matching is represented by two integers
        int block_partial_matching_num = memPoolBlockSize / (2 * sizeof(int));
        int block_num = ceil_div(partial_matching_cnt, block_partial_matching_num);
        cudaCheck(cudaMalloc(&head, block_num * memPoolBlockSize));
        printf("--- Allocating %d bytes (%d int) for head @ %p\n", block_num * memPoolBlockSize, block_num * memPoolBlockSize, head);
        cudaCheck(cudaMemset(head, -1, block_num * memPoolBlockSize));
        nextAddr = head;
        nextAddrBound = head + block_num * memPoolBlockIntNum;
        print_meta(block_num);
    }

    __device__ __forceinline__ int *
    alloc() {
        int *current;
        int *next;
        do {
            current = (int *)atomicAdd((ull *)&nextAddr, 0);
            next = current + memPoolBlockIntNum;
            if (next >= nextAddrBound) {
                // printf("No more available block in the memory pool. The maximum address allowed for a block is %p, yet the requested address is %p\n",
                //        nextAddrBound, nextAddr);
                return nullptr;
            }
        }
        while (atomicCAS((ull *)&nextAddr, (ull)current, (ull)next) != (ull)current);

        return current;
    }

    __host__ void
    freeAll() {
        nextAddr = head;
    }

    __host__ void
    deallocate() {
        if (head != nullptr) {
            cudaCheck(cudaFree(head));
        }
        head = nullptr;
    }

    void
    print_meta(int block_num) {
        printf("+-----------------------------------------+\n");
        printf("| Memory Pool Information                 |\n");
        printf("+-----------------------+-----------------+\n");
        printf("| Head Address          | %-16p|\n", head);
        printf("| Maximum Block Address | %-16p|\n", nextAddrBound);
        printf("| Block Number          | %-16lld|\n", block_num);
        printf("| Block Size (bytes)    | %-16lld|\n", memPoolBlockSize);
        printf("+-----------------------+-----------------+\n");
    }
};

#endif