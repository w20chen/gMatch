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
        static_assert(sizeof(char) == 1);
        static_assert(sizeof(int) == 4);
        static_assert(sizeof(void *) == 8);
        static_assert(sizeof(unsigned long long) == 8);
        static_assert(sizeof(int *) == sizeof(unsigned long long));

        cudaCheck(cudaMalloc(&head, memPoolBlockNum * memPoolBlockSize));
        cudaCheck(cudaMemset(head, -1, memPoolBlockNum * memPoolBlockSize));

        nextAddr = head;
        nextAddrBound = head + memPoolBlockNum * memPoolBlockIntNum;
        // printf("head: %p, nextAddrBound: %p\n", head, nextAddrBound);
    }

    __device__ __forceinline__ int *
    alloc(int flag = 0) {
        if (nextAddr >= nextAddrBound) {
            printf("No more available block in mempool. nextAddrBound: %p, nextAddr: %p\n",
                   nextAddrBound, nextAddr);
            assert(0);
            return nullptr;
        }

        unsigned long long oldNextAddr = atomicAdd((unsigned long long *)&nextAddr,
                                         (unsigned long long)memPoolBlockSize);

        // printf("flag: %d, alloc at %p, len %d int\n", flag, oldNextAddr, memPoolBlockIntNum);

        return (int *)oldNextAddr;
    }

    __host__ int *
    h_alloc() {
        if (nextAddr >= nextAddrBound) {
            printf("No more available block in mempool. nextAddrBound: %p, nextAddr: %p\n",
                   nextAddrBound, nextAddr);
            assert(0);
            return nullptr;
        }

        int *oldNextAddr = nextAddr;
        nextAddr += memPoolBlockIntNum;
        return oldNextAddr;
    }

    __host__ void
    freeAll() {
        nextAddr = head;
        // cudaCheck(cudaMemset(head, -1, memPoolBlockNum * memPoolBlockSize));
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