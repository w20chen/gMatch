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
        cudaCheck(cudaMalloc(&head, memPoolBlockNum * memPoolBlockSize));
        cudaCheck(cudaMemset(head, -1, memPoolBlockNum * memPoolBlockSize));
        nextAddr = head;
        nextAddrBound = head + memPoolBlockNum * memPoolBlockIntNum;
    }

    __device__ __forceinline__ int *
    alloc(int flag = 0) {
        if (nextAddr >= nextAddrBound) {
            printf("No more available block in the memory pool. Next address bound: %p. Next address: %p\n",
                   nextAddrBound, nextAddr);
            return nullptr;
        }

        unsigned long long oldNextAddr = atomicAdd((unsigned long long *)&nextAddr,
                                         (unsigned long long)memPoolBlockSize);
        return (int *)oldNextAddr;
    }

    __host__ int *
    h_alloc() {
        if (nextAddr >= nextAddrBound) {
            printf("No more available block in the memory pool. Next address bound: %p. Next address: %p\n",
                   nextAddrBound, nextAddr);
            return nullptr;
        }

        int *oldNextAddr = nextAddr;
        nextAddr += memPoolBlockIntNum;
        return oldNextAddr;
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
    print_meta() {
        printf("head=%p, memPoolBlockSize=%lld, memPoolBlockNum=%lld, memPoolBlockIntNum=%lld.\n",
               head, memPoolBlockSize, memPoolBlockNum, memPoolBlockIntNum);
    }
};

#endif