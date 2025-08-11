#pragma once

#define maxBlocks 1024
#define threadsPerBlock 512
#define warpsPerBlock (threadsPerBlock / 32)
#define warpNum (maxBlocks * warpsPerBlock)

#define memPoolBlockSize (128LL * 64LL)                               // # of bytes within a block
#define memPoolBlockNum (1024LL * 1024LL)                             // # of memory blocks within a mempool

#define FULL_MASK 0xffffffff

#define ull unsigned long long