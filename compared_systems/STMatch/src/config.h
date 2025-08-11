#pragma once

#define ull unsigned long long

#define cudaCheck(call)                               \
do {                                                  \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess) {                  \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

static void
check_gpu_memory() {
    size_t free_byte, total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);

    size_t used_byte = total_byte - free_byte;
    printf("Device memory: total %ld MB, available %ld MB, used %ld MB\n",
           total_byte / 1024 / 1024, free_byte / 1024 / 1024, used_byte / 1024 / 1024);
}

namespace STMatch {

  typedef int graph_node_t;
  typedef long long graph_edge_t;
  typedef int pattern_node_t;
  typedef char set_op_t;
  typedef unsigned int bitarray32;

  inline constexpr size_t PAT_SIZE = 5;
  inline constexpr size_t GRAPH_DEGREE = 4096;
  inline constexpr size_t MAX_SLOT_NUM = 15;

#include "config_for_ae/fig_local_global_unroll.h" 

  inline constexpr int GRID_DIM = 82;
  inline constexpr int BLOCK_DIM = 1024;
  inline constexpr int WARP_SIZE = 32;
  inline constexpr int NWARPS_PER_BLOCK = (BLOCK_DIM / WARP_SIZE);
  inline constexpr int NWARPS_TOTAL = ((GRID_DIM * BLOCK_DIM + WARP_SIZE - 1) / WARP_SIZE);

  inline constexpr graph_node_t JOB_CHUNK_SIZE = 8;
  //static_assert(2 * JOB_CHUNK_SIZE <= GRAPH_DEGREE); 

  // this is the maximum unroll size

  inline constexpr int DETECT_LEVEL = 1;
  inline constexpr int STOP_LEVEL = 2;
}

#include <chrono>

#define TIME_INIT() cudaEvent_t gpu_start, gpu_end;             \
    float kernel_time, total_kernel = 0.0, total_host = 0.0;    \
    auto cpu_start = std::chrono::high_resolution_clock::now(); \
    auto cpu_end = std::chrono::high_resolution_clock::now();   \
    std::chrono::duration<double> diff = cpu_end - cpu_start;

#define TIME_START()                                            \
    cpu_start = std::chrono::high_resolution_clock::now();      \
    cudaEventCreate(&gpu_start);                                \
    cudaEventCreate(&gpu_end);                                  \
    cudaEventRecord(gpu_start);

#define TIME_END()                                              \
    cpu_end = std::chrono::high_resolution_clock::now();        \
    cudaEventRecord(gpu_end);                                   \
    cudaEventSynchronize(gpu_start);                            \
    cudaEventSynchronize(gpu_end);                              \
    cudaEventElapsedTime(&kernel_time, gpu_start, gpu_end);     \
    total_kernel += kernel_time;                                \
    diff = cpu_end - cpu_start;                                 \
    total_host += diff.count();

#define PRINT_LOCAL_TIME(name)                                  \
    std::cout << "# " << name << ", time (ms): "                \
    << static_cast<unsigned long>(diff.count() * 1000)          \
    << "(host), "                                               \
    << static_cast<unsigned long>(kernel_time)                  \
    << "(kernel)\n";

#define PRINT_TOTAL_TIME(name)                                  \
    std::cout << "# " << name << ", time (ms): "                \
    << static_cast<unsigned long>(total_host * 1000)            \
    << "(host), "                                               \
    << static_cast<unsigned long>(total_kernel)                 \
    << "(kernel)\n";

