#pragma once

// #define IDLE_CNT
// #define BALANCE_CNT

#define CUCKOO_HASH

#define SYMMETRY_BREAKING

#define STK_ELEM_CAND_ON_SHARED

#define UNROLL_MIN 4
#define UNROLL_MAX 32

#define QVMAX 16

#define maxBlocks 1024
#define threadsPerBlock 256
#define warpsPerBlock (threadsPerBlock / 32)
#define warpNum (maxBlocks * warpsPerBlock)

#define memPoolBlockSize (32LL * QVMAX * sizeof(int))                 // # of bytes within a block
#define memPoolBlockNum (1024LL * 1024LL * 2)                         // # of memory blocks within a mempool

#define FULL_MASK 0xffffffff
