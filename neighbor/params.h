#pragma once

// #define IDLE_CNT
// #define BALANCE_CNT

#define UNLABELED    // comment this out if the graph is labeled

// #define NLF_FILTER   // comment this out if you do not want to use NLF filter

// #define DEGREE_FILTER_ON_THE_FLY

#define SYMMETRY_BREAKING

#define EXPECT_BEGIN_NUM 1e7

#define UNROLL_MIN 1
#define UNROLL_MAX 32

#define maxBlocks 1024
#define threadsPerBlock 512
#define warpsPerBlock (threadsPerBlock / 32)
#define warpNum (maxBlocks * warpsPerBlock)

#define memPoolBlockSize (128LL * 64LL)                               // # of bytes within a block
#define memPoolBlockNum (1024LL * 1024LL)                             // # of memory blocks within a mempool

#define FULL_MASK 0xffffffff
