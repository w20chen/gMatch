#pragma once

#define IDLE_CNT
// #define BALANCE_CNT
// #define WORK_STEALING

#define UNROLL_MIN 1
#define UNROLL_MAX 32

#define QVMAX 16

#define INF 0x7fffffff

#define maxBlocks 1024
#define threadsPerBlock 512
#define warpsPerBlock (threadsPerBlock / 32)
#define warpNum (maxBlocks * warpsPerBlock)

#define memPoolBlockSize (128LL * 64LL)                               // # of bytes within a block
#define memPoolBlockNum (1024LL * 1024LL)                             // # of memory blocks within a mempool

#define BFS_BEGIN_SIZE 128

#define FULL_MASK 0xffffffff

#define LOCAL_TASK_QUEUE_SIZE 128
#define GLOBAL_TASK_QUEUE_SIZE (1024 * 128)
