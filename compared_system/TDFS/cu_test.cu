#include <string>
#include <iostream>
#include "src/gpu_match.cuh"

#include "src/Ouroboros/include/device/Ouroboros_impl.cuh"
#include "src/Ouroboros/include/device/MemoryInitialization.cuh"
#include "src/Ouroboros/include/InstanceDefinitions.cuh"
#include "src/Ouroboros/include/Utility.h"

using namespace std;
using namespace STMatch;

#define TIMEOUT_QUEUE_CAP 1'000'000
#define NUM_POINTERS 40

int main(int argc, char *argv[]) {
    printf("[DEBUG] Starting program execution\n");

    cudaSetDevice(1);
    printf("[DEBUG] CUDA device set to 0\n");

    STMatch::GraphPreprocessor g(argv[1]);
    STMatch::PatternPreprocessor p(argv[2]);
    g.build_src_vtx(p);
    printf("[DEBUG] Graph and pattern preprocessing completed\n");

    check_gpu_memory();

    std::cout << "conditions: " << std::endl;
    for (int i = 0; i < p.order_.size(); i++) {
        std::cout << i << ": ";
        for (int j = 0; j < p.order_[i].size(); j++) {
            std::cout << GetCondOperatorString(p.order_[i][j].first) << "(" << p.order_[i][j].second << "), ";
        }
        std::cout << std::endl;
    }

    size_t instantitation_size = 4ULL * 1024ULL * 1024ULL * 1024ULL; //4GB
    printf("[DEBUG] Initializing memory manager with size: %lu bytes\n", instantitation_size);
    MemoryManagerType memory_manager;
    memory_manager.initialize(instantitation_size);
    printf("[DEBUG] Memory manager initialization completed\n");

    check_gpu_memory();

    // copy graph and pattern to GPU global memory
    printf("[DEBUG] Starting GPU memory allocation and data transfer\n");
    Graph *gpu_graph = g.to_gpu();
    printf("[DEBUG] Data graph copied to GPU\n");
    Pattern *gpu_pattern = p.to_gpu();
    printf("[DEBUG] Pattern graph copied to GPU\n");
    auto jq = JobQueuePreprocessor(g.g, p);
    printf("[DEBUG] Job queue created on CPU\n");
    JobQueue *gpu_queue = jq.to_gpu();
    printf("[DEBUG] Job queue copied to GPU\n");
    CallStack *gpu_callstack = nullptr;

    // allocate the callstack for all warps in global memory
    // printf("[DEBUG] Allocating callstack memory\n");
    // graph_node_t *slot_storage = nullptr;
    // cudaMalloc(&slot_storage, sizeof(graph_node_t) * NWARPS_TOTAL * MAX_SLOT_NUM * GRAPH_DEGREE);

    std::vector<CallStack> stk(NWARPS_TOTAL);

    graph_node_t **index_map = nullptr;
    cudaCheck(cudaMalloc(&index_map, 8 * NUM_POINTERS * NWARPS_TOTAL * PAT_SIZE));
    printf("--- allocate %lf MB\n", 8 * NUM_POINTERS * NWARPS_TOTAL * PAT_SIZE / 1024. / 1024.);
    printf("[DEBUG] Index map memory allocated\n");

    for (int i = 0; i < NWARPS_TOTAL; i++) {
        auto &s = stk[i];
        memset(s.iter, 0, sizeof(s.iter));
        memset(s.slot_size, 0, sizeof(s.slot_size));

        s.slot_storage.mm = memory_manager.getDeviceMemoryManager();
        for (int j = 0; j < PAT_SIZE; ++j) {
            s.slot_storage.buffers[j].index_map = index_map + NUM_POINTERS * (i * PAT_SIZE + j);
        }
    }
    printf("[DEBUG] Callstack initialization completed\n");

    cudaCheck(cudaMalloc(&gpu_callstack, NWARPS_TOTAL * sizeof(CallStack)));
    cudaMemcpy(gpu_callstack, stk.data(), sizeof(CallStack) * NWARPS_TOTAL, cudaMemcpyHostToDevice);
    printf("[DEBUG] Callstack copied to GPU\n");
    printf("--- allocate %lf MB\n", NWARPS_TOTAL * sizeof(CallStack) / 1024. / 1024.);

    printf("[DEBUG] Allocating result and control variables\n");
    size_t *gpu_res;
    cudaCheck(cudaMalloc(&gpu_res, sizeof(size_t) * NWARPS_TOTAL));
    cudaMemset(gpu_res, 0, sizeof(size_t) * NWARPS_TOTAL);
    size_t *res = new size_t[NWARPS_TOTAL];

    int *idle_warps;
    cudaCheck(cudaMalloc(&idle_warps, sizeof(int) * GRID_DIM));
    cudaMemset(idle_warps, 0, sizeof(int) * GRID_DIM);

    int *idle_warps_count;
    cudaCheck(cudaMalloc(&idle_warps_count, sizeof(int)));
    cudaMemset(idle_warps_count, 0, sizeof(int));

    int *global_mutex;
    cudaCheck(cudaMalloc(&global_mutex, sizeof(int) * GRID_DIM));
    cudaMemset(global_mutex, 0, sizeof(int) * GRID_DIM);

    bool *stk_valid;
    cudaCheck(cudaMalloc(&stk_valid, sizeof(bool) * GRID_DIM));
    cudaMemset(stk_valid, 0, sizeof(bool) * GRID_DIM);

    int *gpu_timeout_queue_space;
    cudaCheck(cudaMalloc(&gpu_timeout_queue_space, sizeof(int) * TIMEOUT_QUEUE_CAP * (STOP_LEVEL + 1)));
    Queue *gpu_timeout_queue;
    cudaCheck(cudaMallocManaged(&gpu_timeout_queue, sizeof(Queue)));
    gpu_timeout_queue->queue_ = gpu_timeout_queue_space;
    gpu_timeout_queue->size_ = TIMEOUT_QUEUE_CAP * (STOP_LEVEL + 1);
    gpu_timeout_queue->resetQueue();
    printf("[DEBUG] Timeout queue initialization completed\n");

    // record #pages used during execution
    int *gpu_page_consumption;
    cudaCheck(cudaMalloc(&gpu_page_consumption, sizeof(int) * NWARPS_TOTAL * PAT_SIZE));
    printf("[DEBUG] Page consumption tracking initialized\n");

    printf("[DEBUG] Starting GPU memory allocation\n");
    allocate_memory <<< GRID_DIM, BLOCK_DIM>>>(memory_manager.getDeviceMemoryManager(), gpu_callstack);
    HANDLE_ERROR(cudaDeviceSynchronize());
    printf("[DEBUG] GPU memory allocation completed\n");

    // timer starts here
    printf("[DEBUG] Starting execution timer\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // std::vector<uint32_t> h_partial_order;
    // p.restriction_generation(h_partial_order);
    // for (auto mask : h_partial_order) {
    //     printf("%x\n", mask);
    // }

    // int *d_partial_order = nullptr;
    // cudaMalloc(&d_partial_order, sizeof(int) * h_partial_order.size());
    // cudaMemcpy(d_partial_order, h_partial_order.size(), sizeof(int) * h_partial_order.size(), cudaMemcpyHostToDevice);

    printf("[DEBUG] Starting parallel matching kernel\n");
    _parallel_match << <GRID_DIM, BLOCK_DIM >> > (memory_manager.getDeviceMemoryManager(), gpu_graph, gpu_pattern, gpu_callstack, gpu_queue, gpu_res, idle_warps,
                    idle_warps_count, global_mutex, gpu_timeout_queue, gpu_page_consumption);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    printf("[DEBUG] Parallel matching kernel completed\n");

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("[DEBUG] Copying results from GPU to host\n");
    cudaMemcpy(res, gpu_res, sizeof(size_t) * NWARPS_TOTAL, cudaMemcpyDeviceToHost);

    // obtain how many pages each level use
    printf("[DEBUG] Calculating page consumption\n");
    int *page_consumption = new int[NWARPS_TOTAL * PAT_SIZE];
    cudaMemcpy(page_consumption, gpu_page_consumption, sizeof(int) * NWARPS_TOTAL * PAT_SIZE, cudaMemcpyDeviceToHost);
    ull page_ttl = 0;
    for (int i = 0; i < NWARPS_TOTAL * PAT_SIZE; ++i) {
        // printf("page_consumption[%d]=%d\n", i, page_consumption[i]);
        // assert(page_consumption[i] > 0);
        // printf("%d,", page_consumption[i]);
        page_ttl += page_consumption[i];
    }
    printf("[DEBUG] Total pages used: %llu\n", page_ttl);
    printf(" = %.3f M bytes\n", (float)page_ttl * LARGEST_PAGE_SIZE / 1024. / 1024.);
    delete[] page_consumption;

    printf("[DEBUG] Calculating total match count\n");
    unsigned long long tot_count = 0;
    for (int i = 0; i < NWARPS_TOTAL; i++) {
        tot_count += res[i];
    }

    if (!LABELED) {
        tot_count = tot_count * p.PatternMultiplicity;
    }

    printf("[DEBUG] Final results:\n");
    printf("Pattern: %s\n", argv[2]);
    printf("Execution time (ms): %f\n", milliseconds);
    printf("Total matches: %llu\n", tot_count);

    printf("[DEBUG] Program execution completed\n");

#ifdef QUEUE_TIME_PROFILING
    int clock_kHz = 0;
    cudaDeviceGetAttribute(&clock_kHz, cudaDevAttrClockRate, 0);

    printf("[DEBUG] Queue time profiling (kHz): %d\n", clock_kHz);
    printf("Enqueue total time: %.2lf milliseconds\n", gpu_timeout_queue->getEnqueueCycles() / (double)clock_kHz);
    printf("Dequeue total time: %.2lf milliseconds\n", gpu_timeout_queue->getDequeueCycles() / (double)clock_kHz);
    printf("Enqueue count: %d\n", gpu_timeout_queue->getEnqueueCount());
    printf("Dequeue count: %d\n", gpu_timeout_queue->getDequeueCount());
#endif

    return 0;
}
