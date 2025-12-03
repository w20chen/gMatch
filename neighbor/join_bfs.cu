#include "join.h"
#include "candidate.h"
#include "params.h"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <vector>


__global__ void count_matches_kernel_sym(
    const int* d_cand_0, int cand_0_size,
    const ull* d_adj_index, const int* d_adj_array,
    const char* d_cand_1, int* d_counts,
    unsigned* partial_order
) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (partial_order[1] & (1 << 0)) {
        for (int i = thread_idx; i < cand_0_size; i += blockDim.x * gridDim.x) {
            int v0 = d_cand_0[i];
            long long start = d_adj_index[v0];
            long long end = d_adj_index[v0 + 1];
            int count = 0;

            // for each neighbor v1 of v0
            for (long long j = start; j < end; j++) {
                int v1 = d_adj_array[j];
                // condition of symmetry breaking
                if (v1 >= v0) {
                    break;
                }
                if (d_cand_1[v1]) {
                    count++;
                }
            }
            d_counts[i] = count;
        }
    }
    else {
        for (int i = thread_idx; i < cand_0_size; i += blockDim.x * gridDim.x) {
            int v0 = d_cand_0[i];
            long long start = d_adj_index[v0];
            long long end = d_adj_index[v0 + 1];
            int count = 0;

            for (long long j = start; j < end; j++) {
                int v1 = d_adj_array[j];
                if (d_cand_1[v1]) {
                    count++;
                }
            }
            d_counts[i] = count;
        }
    }
}

__global__ void fill_matches_kernel_sym(
    const int* d_cand_0, int cand_0_size,
    const ull* d_adj_index, const int* d_adj_array,
    const char* d_cand_1, const int* d_offsets,
    int2* d_output,
    unsigned* partial_order
) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (partial_order[1] & (1 << 0)) {
        for (int i = thread_idx; i < cand_0_size; i += blockDim.x * gridDim.x) {
            int v0 = d_cand_0[i];
            long long start = d_adj_index[v0];
            long long end = d_adj_index[v0 + 1];
            int pos = d_offsets[i];

            for (long long j = start; j < end; j++) {
                int v1 = d_adj_array[j];
                if (v1 >= v0) {
                    break;
                }
                if (d_cand_1[v1]) {
                    d_output[pos] = make_int2(v0, v1);
                    pos++;
                }
            }
        }
    }
    else {
        for (int i = thread_idx; i < cand_0_size; i += blockDim.x * gridDim.x) {
            int v0 = d_cand_0[i];
            long long start = d_adj_index[v0];
            long long end = d_adj_index[v0 + 1];
            int pos = d_offsets[i];

            for (long long j = start; j < end; j++) {
                int v1 = d_adj_array[j];

                if (d_cand_1[v1]) {
                    d_output[pos] = make_int2(v0, v1);
                    pos++;
                }
            }
        }
    }
}

__global__ void count_matches_kernel(
    const int* d_cand_0, int cand_0_size,
    const ull* d_adj_index, const int* d_adj_array,
    const char* d_cand_1, int* d_counts
) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_idx; i < cand_0_size; i += blockDim.x * gridDim.x) {
        int v0 = d_cand_0[i];
        long long start = d_adj_index[v0];
        long long end = d_adj_index[v0 + 1];
        int count = 0;

        for (long long j = start; j < end; j++) {
            int v1 = d_adj_array[j];

            if (d_cand_1[v1]) {
                count++;
            }
        }
        d_counts[i] = count;
    }
}

__global__ void fill_matches_kernel(
    const int* d_cand_0, int cand_0_size,
    const ull* d_adj_index, const int* d_adj_array,
    const char* d_cand_1, const int* d_offsets,
    int2* d_output
) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_idx; i < cand_0_size; i += blockDim.x * gridDim.x) {
        int v0 = d_cand_0[i];
        long long start = d_adj_index[v0];
        long long end = d_adj_index[v0 + 1];
        int pos = d_offsets[i];

        for (long long j = start; j < end; j++) {
            int v1 = d_adj_array[j];

            if (d_cand_1[v1]) {
                d_output[pos] = make_int2(v0, v1);
                pos++;
            }
        }
    }
}

int *match_first_edge_sym(
    const std::vector<int>& cand_0,
    const std::vector<char>& cand_1,
    candidate_graph_GPU &cg,
    int &cnt,
    std::vector<uint32_t> &partial_order
) {
    int *d_cand_0 = nullptr;
    char *d_cand_1 = nullptr;

    int2 *d_output = nullptr;
    int *d_counts = nullptr;

    int cand_0_size = cand_0.size();

    cudaCheck(cudaMalloc(&d_cand_0, cand_0_size * sizeof(int)));
    cudaCheck(cudaMemcpy(d_cand_0, cand_0.data(), cand_0_size * sizeof(int), cudaMemcpyHostToDevice));

    cudaCheck(cudaMalloc(&d_cand_1, cand_1.size() * sizeof(char)));
    cudaCheck(cudaMemcpy(d_cand_1, cand_1.data(), cand_1.size() * sizeof(char), cudaMemcpyHostToDevice));

    cudaCheck(cudaMalloc(&d_counts, cand_0_size * sizeof(int)));

    const int block_size = 256;
    int grid_size = (cand_0_size + block_size - 1) / block_size;

    unsigned *d_partial_order = nullptr;
    cudaCheck(cudaMalloc(&d_partial_order, partial_order.size() * sizeof(unsigned)));
    cudaCheck(cudaMemcpy(d_partial_order, partial_order.data(), partial_order.size() * sizeof(unsigned), cudaMemcpyHostToDevice));

    count_matches_kernel_sym<<<grid_size, block_size>>>(
        d_cand_0, cand_0_size, cg.g_nbr_offset, cg.g_nbr_array, d_cand_1, d_counts, d_partial_order
    );
    cudaCheck(cudaDeviceSynchronize());

    int total_matches = 0;
    cudaCheck(cudaMemcpy(&total_matches, d_counts + cand_0_size - 1, sizeof(int), cudaMemcpyDeviceToHost));

    thrust::exclusive_scan(thrust::device, d_counts, d_counts + cand_0_size, d_counts);

    int tmp = 0;
    cudaCheck(cudaMemcpy(&tmp, d_counts + cand_0_size - 1, sizeof(int), cudaMemcpyDeviceToHost));
    total_matches += tmp;

    cudaCheck(cudaMalloc(&d_output, total_matches * sizeof(int2)));

    fill_matches_kernel_sym<<<grid_size, block_size>>>(
        d_cand_0, cand_0_size, cg.g_nbr_offset, cg.g_nbr_array, d_cand_1, d_counts, d_output, d_partial_order
    );
    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaFree(d_cand_0));
    cudaCheck(cudaFree(d_cand_1));
    cudaCheck(cudaFree(d_counts));
    cudaCheck(cudaFree(d_partial_order));

    cnt = total_matches;

    return (int *)d_output;
}

int *match_first_edge(
    const std::vector<int>& cand_0,
    const std::vector<char>& cand_1,
    candidate_graph_GPU &cg,
    int &cnt
) {
    int *d_cand_0 = nullptr;
    char *d_cand_1 = nullptr;

    int2 *d_output = nullptr;
    int *d_counts = nullptr;

    int cand_0_size = cand_0.size();

    cudaCheck(cudaMalloc(&d_cand_0, cand_0_size * sizeof(int)));
    cudaCheck(cudaMemcpy(d_cand_0, cand_0.data(), cand_0_size * sizeof(int), cudaMemcpyHostToDevice));

    cudaCheck(cudaMalloc(&d_cand_1, cand_1.size() * sizeof(char)));
    cudaCheck(cudaMemcpy(d_cand_1, cand_1.data(), cand_1.size() * sizeof(char), cudaMemcpyHostToDevice));

    cudaCheck(cudaMalloc(&d_counts, cand_0_size * sizeof(int)));

    const int block_size = 256;
    int grid_size = (cand_0_size + block_size - 1) / block_size;

    count_matches_kernel<<<grid_size, block_size>>>(
        d_cand_0, cand_0_size, cg.g_nbr_offset, cg.g_nbr_array, d_cand_1, d_counts
    );
    cudaCheck(cudaDeviceSynchronize());

    int total_matches = 0;
    cudaCheck(cudaMemcpy(&total_matches, d_counts + cand_0_size - 1, sizeof(int), cudaMemcpyDeviceToHost));

    thrust::exclusive_scan(thrust::device, d_counts, d_counts + cand_0_size, d_counts);

    int tmp = 0;
    cudaCheck(cudaMemcpy(&tmp, d_counts + cand_0_size - 1, sizeof(int), cudaMemcpyDeviceToHost));
    total_matches += tmp;

    cudaCheck(cudaMalloc(&d_output, total_matches * sizeof(int2)));

    fill_matches_kernel<<<grid_size, block_size>>>(
        d_cand_0, cand_0_size, cg.g_nbr_offset, cg.g_nbr_array, d_cand_1, d_counts, d_output
    );
    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaFree(d_cand_0));
    cudaCheck(cudaFree(d_cand_1));
    cudaCheck(cudaFree(d_counts));

    cnt = total_matches;

    return (int *)d_output;
}

__host__ int *
set_beginning_partial_matchings(
    Graph &q,
    Graph &g,
    candidate_graph_GPU &cg,
    int &cnt
) {

    TIME_INIT();
    TIME_START();

    std::vector<int> cand_0;
    cand_0.reserve(1e6);

    std::vector<char> cand_1(g.vcount(), 0);

#ifndef UNLABELED
    for (int u = 0; u < 2; u++) {
        int label = q.label(u);
        for (int v = 0; v < g.vcount(); v++) {
            if (g.label(v) != label) {
                continue;
            }
            bool fail = false;
#ifdef NLF_FILTER
            for (auto p : q.nlf_[u]) {
                int label = p.first;
                int count = p.second;
                if (count > g.nlf_[v][label]) {
                    fail = true;
                    break;
                }
            }
#else
            if (g.degree(v) < q.degree(u)) {
                fail = true;
            }
#endif
            if (!fail) {
                if (u == 0) {
                    cand_0.push_back(v);
                }
                else {
                    cand_1[v] = true;
                }
            }
        }
    }
#else
    for (int u = 0; u < 2; u++) {
        for (int v = 0; v < g.vcount(); v++) {
            if (g.degree(v) >= q.degree(u)) {
                if (u == 0) {
                    cand_0.push_back(v);
                }
                else {
                    cand_1[v] = true;
                }
            }
        }
    }
#endif

    TIME_END();
    PRINT_LOCAL_TIME("Generate the candidate sets for the first two vertices");

    printf("|C(0)| = %d\n", (int)cand_0.size());

    TIME_START();

    int *d_output = match_first_edge(cand_0, cand_1, cg, cnt);
    printf("Number of matches for the first edge: %d\n", cnt);

    TIME_END();
    PRINT_LOCAL_TIME("Match the first edge");

    return d_output;
}


__host__ int *
set_beginning_partial_matchings_sym(
    Graph &q,
    Graph &g,
    candidate_graph_GPU &cg,
    int &cnt,
    std::vector<uint32_t> &partial_order
) {

    TIME_INIT();
    TIME_START();

    std::vector<int> cand_0;
    cand_0.reserve(1e6);

    std::vector<char> cand_1(g.vcount(), 0);

#ifndef UNLABELED
    for (int u = 0; u < 2; u++) {
        int label = q.label(u);
        for (int v = 0; v < g.vcount(); v++) {
            if (g.label(v) != label) {
                continue;
            }
            bool fail = false;
#ifdef NLF_FILTER
            for (auto p : q.nlf_[u]) {
                int label = p.first;
                int count = p.second;
                if (count > g.nlf_[v][label]) {
                    fail = true;
                    break;
                }
            }
#else
            if (g.degree(v) < q.degree(u)) {
                fail = true;
            }
#endif
            if (!fail) {
                if (u == 0) {
                    cand_0.push_back(v);
                }
                else {
                    cand_1[v] = true;
                }
            }
        }
    }
#else
    for (int u = 0; u < 2; u++) {
        for (int v = 0; v < g.vcount(); v++) {
            if (g.degree(v) >= q.degree(u)) {
                if (u == 0) {
                    cand_0.push_back(v);
                }
                else {
                    cand_1[v] = true;
                }
            }
        }
    }
#endif

    TIME_END();
    PRINT_LOCAL_TIME("Generate the candidate sets for the first two vertices");

    printf("|C(0)| = %d\n", (int)cand_0.size());

    TIME_START();

    int *d_output = match_first_edge_sym(cand_0, cand_1, cg, cnt, partial_order);
    printf("Number of matches for the first edge: %d\n", cnt);

    TIME_END();
    PRINT_LOCAL_TIME("Match the first edge");

    return d_output;
}


void __global__
BFS_Extend(
    const Graph_GPU Q,
    const Graph_GPU G,
    const candidate_graph_GPU cg,
    MemManager *d_MM,
    int cur_query_vertex,
    int partial_offset,
    int last_flag,
    int *d_error_flag
) {

    /**
     * After the function ends, the values of prev_head and blk_write_cnt are retained for the next call
     * and cached in shared memory to reduce global memory access.
     */

    __shared__ int shared_blk_write_cnt[warpsPerBlock];
    __shared__ int *shared_prev_head[warpsPerBlock];

    int warp_id = (threadIdx.x + blockDim.x * blockIdx.x) / warpSize;
    char lane_id = (threadIdx.x + blockDim.x * blockIdx.x) % warpSize;
    char warp_id_within_blk = warp_id % warpsPerBlock;

    if (lane_id == 0) {
        shared_prev_head[warp_id_within_blk] = d_MM->prev_head[warp_id];
        shared_blk_write_cnt[warp_id_within_blk] = d_MM->blk_write_cnt[warp_id];
    }
    __syncthreads();

    // If last_flag is 1, it indicates this is the last call to BFS_Extend
    // Record all partial matchings not yet recorded in d_MM
    if (last_flag == 1) {
        if (lane_id == 0) {
            if (shared_blk_write_cnt[warp_id_within_blk] > 0) {
                // Use partial_offset to temporarily denote the length of partial matching
                d_MM->add_new_props(shared_prev_head[warp_id_within_blk], partial_offset, shared_blk_write_cnt[warp_id_within_blk]);
            }
            d_MM->prev_head[warp_id] = nullptr;
            d_MM->blk_write_cnt[warp_id] = 0;
        }
        return;
    }

    // Assign partial matching to each warp
    int *this_partial_matching = d_MM->get_partial(warp_id + partial_offset);
    if (this_partial_matching == nullptr) {
        return;
    }

    // Each block can write a maximum number of partial matchings
    const int blk_partial_max_num = memPoolBlockIntNum / (cur_query_vertex + 1);

    int min_len = INT32_MAX;
    int *min_set = nullptr;

    unsigned bn_mask = Q.d_bknbrs_[cur_query_vertex];

    while (bn_mask) {
        char uu = __ffs(bn_mask) - 1;
        bn_mask &= ~(1u << uu);
        int this_len = 0;
        int *this_set = cg.get_nbr(this_partial_matching[uu], this_len);
        if (this_len < min_len) {
            min_len = this_len;
            min_set = this_set;
        }
    }

    // Ensure there is one memory block for each warp
    if (lane_id == 0 && shared_prev_head[warp_id_within_blk] == nullptr) {
        shared_prev_head[warp_id_within_blk] = d_MM->mempool_to_write()->alloc();
        shared_blk_write_cnt[warp_id_within_blk] = 0;
    }
    __syncwarp();

#ifndef UNLABELED
    int label_u = Q.d_label_[cur_query_vertex];
#endif

    // Compute extendable candidate set
    // Each thread in the warp is responsible for one candidate, and handles the next candidate with a step size of 32
    for (int t = lane_id; t < min_len; t += warpSize) {
        // For each vertex v in min_set
        int v = min_set[t];
        bool flag = true;

#ifndef UNLABELED
        if (label_u != G.d_label_[v]) {
            flag = false;
        }
#endif

        // Make sure that v has not been mapped before
        for (char j = 0; flag && j < cur_query_vertex; j++) {
            if (this_partial_matching[j] == v) {
                flag = false;
                break;
            }
        }

        if (flag) {
            // For each backward neighbor uu of u
            unsigned bn_mask = Q.d_bknbrs_[cur_query_vertex];
            while (bn_mask) {
                char uu = __ffs(bn_mask) - 1;
                bn_mask &= ~(1u << uu);
                int this_len = 0;
                int *this_set = cg.get_nbr(this_partial_matching[uu], this_len);
                if (this_set == min_set) {
                    continue;
                }
                if (!binary_search_int(this_set, this_len, v)) {
                    flag = false;
                    break;
                }
            }
        }
        __syncwarp();

        // Use ballot_sync to collect flag values from all threads and generate a mask
        unsigned int flag_mask = __ballot_sync(FULL_MASK, flag);
        // Use popc to count the number of bits set to 1 in the mask, i.e., the total number of threads with flag true
        // unsigned int flag_cnt = __popc(flag_mask);

        // flag_idx indicates the position of each thread in its warp where flag == true
        // flag_idx starts counting from 0
        // For threads with flag == false, flag_idx is meaningless
        char flag_idx = __popc(flag_mask & ((1 << lane_id) - 1));

        // For the current process, it will write partial matching to a block that already has thread_old_cnt partial matchings
        int thread_old_cnt = shared_blk_write_cnt[warp_id_within_blk] + flag_idx;
        // Indicates whether the current process needs to write the partial matching to a new block instead of the old one
        bool thread_need_new_blk = false;

        // rest_cnt_in_blk indicates the number of partial matchings that can still be written to the old block
        int rest_cnt_in_blk = blk_partial_max_num - shared_blk_write_cnt[warp_id_within_blk];

        if (flag && thread_old_cnt + 1 > blk_partial_max_num) {
            // For this thread, it will write partial matching to a new block because the old block is full
            thread_need_new_blk = true;
            // Mark shared_blk_write_cnt[warp_id_within_blk] == -1
            // to indicate that there are threads in the current warp that need to request a new block
            shared_blk_write_cnt[warp_id_within_blk] = -1;
            // In the new block, the number of blocks that already exist before this thread
            thread_old_cnt = flag_idx - rest_cnt_in_blk;

            if (thread_old_cnt >= blk_partial_max_num) {
                // Ensure that the number of partial matchings that can be written in a block blk_partial_max_num is greater than warpSize
                // Otherwise, more than two blocks need to be requested in one round of 32-thread parallel execution
                // printf("The program should not reach here.\n");
                // printf("Debug Info - Thread %d in warp %d:\n", tid, warp_id);
                // printf("Old count: %d\n", thread_old_cnt);
                // printf("Rest count in block: %d\n", rest_cnt_in_blk);
                // printf("Limit: %d\n", blk_partial_max_num);
                // printf("Partial matching length: %d\n", partial_matching_len);
                // printf("Flag index: %d\n", flag_idx);
                // printf("Flag count: %d\n", flag_cnt);
                // printf("Need new block: %d\n", thread_need_new_blk);
                // printf("Block write count: %d\n", shared_blk_write_cnt[warp_id_within_blk]);
                // assert(0);
            }
        }
        __syncwarp();

        // Address of the newly allocated block
        int *d_new_head = 0;
        unsigned int d_new_head_lower = 0;
        unsigned int d_new_head_upper = 0;

        if (lane_id == 0 && shared_blk_write_cnt[warp_id_within_blk] == -1) {
            // If shared_blk_write_cnt[warp_id_within_blk] == -1,
            // it means there are threads in this warp that need to write results to a new memory block
            // Register this block and allocate a new block for this warp
            shared_blk_write_cnt[warp_id_within_blk] = -rest_cnt_in_blk;
            d_MM->add_new_props(shared_prev_head[warp_id_within_blk], cur_query_vertex + 1, blk_partial_max_num);

            d_new_head = d_MM->mempool_to_write()->alloc();
            d_new_head_lower = (unsigned)d_new_head;
            d_new_head_upper = (unsigned)((unsigned long long)d_new_head >> 32);
        }
        __syncwarp();

        d_new_head_lower = __shfl_sync(FULL_MASK, d_new_head_lower, 0);
        d_new_head_upper = __shfl_sync(FULL_MASK, d_new_head_upper, 0);
        d_new_head = (int *)(((unsigned long long)d_new_head_upper << 32) |
                             (unsigned long long)d_new_head_lower);
        // d_new_head is null if no new block is allocated

        if (lane_id == 0) {
            // If shared_blk_write_cnt[warp_id_within_blk] != -1, it means no new block is needed
            shared_blk_write_cnt[warp_id_within_blk] += __popc(flag_mask);
        }
        __syncwarp();

        // Some threads write to the new block, some write to the old block
        int *blk_to_write = shared_prev_head[warp_id_within_blk];
        if (thread_need_new_blk == true) {
            blk_to_write = d_new_head;
            shared_prev_head[warp_id_within_blk] = d_new_head;
        }
        __syncwarp();

        if (blk_to_write == nullptr) {
            *d_error_flag = 1;
            return;
        }

        if (flag) {
            // Write the newly found partial matching
            int idx = thread_old_cnt * (cur_query_vertex + 1) + cur_query_vertex;
            if (idx >= memPoolBlockIntNum || idx < 0) {
                // printf("idx: %d, memPoolBlockIntNum: %lld\n", idx, memPoolBlockIntNum);
                // assert(0);
            }
            blk_to_write[idx] = v;
            for (char i = 0; i < cur_query_vertex; i++) {
                idx = thread_old_cnt * (cur_query_vertex + 1) + i;
                blk_to_write[idx] = this_partial_matching[i];
            }
        }
    }

    if (lane_id == 0) {
        d_MM->prev_head[warp_id] = shared_prev_head[warp_id_within_blk];
        d_MM->blk_write_cnt[warp_id] = shared_blk_write_cnt[warp_id_within_blk];
    }
}


void __global__
BFS_Extend_sym(
    const Graph_GPU Q,
    const Graph_GPU G,
    const candidate_graph_GPU cg,
    MemManager *d_MM,
    int cur_query_vertex,
    int partial_offset,
    int last_flag,
    int *d_error_flag,
    int *d_partial_order
) {

    __shared__ int shared_blk_write_cnt[warpsPerBlock];
    __shared__ int *shared_prev_head[warpsPerBlock];
    __shared__ uint32_t s_partial_order[32];

    int warp_id = (threadIdx.x + blockDim.x * blockIdx.x) / warpSize;
    char lane_id = (threadIdx.x + blockDim.x * blockIdx.x) % warpSize;
    char warp_id_within_blk = warp_id % warpsPerBlock;

    if (lane_id == 0) {
        shared_prev_head[warp_id_within_blk] = d_MM->prev_head[warp_id];
        shared_blk_write_cnt[warp_id_within_blk] = d_MM->blk_write_cnt[warp_id];
    }

    if (threadIdx.x < Q.vcount()) {
        s_partial_order[threadIdx.x] = d_partial_order[threadIdx.x];
    }
    __syncthreads();

    if (last_flag == 1) {
        if (lane_id == 0) {
            if (shared_blk_write_cnt[warp_id_within_blk] > 0) {
                d_MM->add_new_props(shared_prev_head[warp_id_within_blk], partial_offset, shared_blk_write_cnt[warp_id_within_blk]);
            }
            d_MM->prev_head[warp_id] = nullptr;
            d_MM->blk_write_cnt[warp_id] = 0;
        }
        return;
    }

    int *this_partial_matching = d_MM->get_partial(warp_id + partial_offset);
    if (this_partial_matching == nullptr) {
        return;
    }

    const int blk_partial_max_num = memPoolBlockIntNum / (cur_query_vertex + 1);

    int min_len = INT32_MAX;
    int *min_set = nullptr;

    unsigned bn_mask = Q.d_bknbrs_[cur_query_vertex];

    while (bn_mask) {
        char uu = __ffs(bn_mask) - 1;
        bn_mask &= ~(1u << uu);
        int this_len = 0;
        int *this_set = cg.get_nbr(this_partial_matching[uu], this_len);
        if (this_len < min_len) {
            min_len = this_len;
            min_set = this_set;
        }
    }

    if (lane_id == 0 && shared_prev_head[warp_id_within_blk] == nullptr) {
        shared_prev_head[warp_id_within_blk] = d_MM->mempool_to_write()->alloc();
        shared_blk_write_cnt[warp_id_within_blk] = 0;
    }
    __syncwarp();

#ifndef UNLABELED
    int label_u = Q.d_label_[cur_query_vertex];
#endif

    for (int t = lane_id; t < min_len; t += warpSize) {
        int v = min_set[t];
        bool flag = true;

#ifndef UNLABELED
        if (label_u != G.d_label_[v]) {
            flag = false;
        }
#endif

        for (char j = 0; flag && j < cur_query_vertex; j++) {
            if (this_partial_matching[j] == v) {
                flag = false;
                break;
            }
            else if (s_partial_order[cur_query_vertex] & (1 << j)) {
                if (v >= this_partial_matching[j]) {
                    flag = false;
                    break;
                }
            }
        }

        if (flag) {
            unsigned bn_mask = Q.d_bknbrs_[cur_query_vertex];
            while (bn_mask) {
                char uu = __ffs(bn_mask) - 1;
                bn_mask &= ~(1u << uu);
                int this_len = 0;
                int *this_set = cg.get_nbr(this_partial_matching[uu], this_len);
                if (this_set == min_set) {
                    continue;
                }
                if (!binary_search_int(this_set, this_len, v)) {
                    flag = false;
                    break;
                }
            }
        }
        __syncwarp();

        unsigned int flag_mask = __ballot_sync(FULL_MASK, flag);
        char flag_idx = __popc(flag_mask & ((1 << lane_id) - 1));
        int thread_old_cnt = shared_blk_write_cnt[warp_id_within_blk] + flag_idx;
        bool thread_need_new_blk = false;
        int rest_cnt_in_blk = blk_partial_max_num - shared_blk_write_cnt[warp_id_within_blk];

        if (flag && thread_old_cnt + 1 > blk_partial_max_num) {
            thread_need_new_blk = true;
            shared_blk_write_cnt[warp_id_within_blk] = -1;
            thread_old_cnt = flag_idx - rest_cnt_in_blk;
        }
        __syncwarp();

        int *d_new_head = 0;
        unsigned int d_new_head_lower = 0;
        unsigned int d_new_head_upper = 0;

        if (lane_id == 0 && shared_blk_write_cnt[warp_id_within_blk] == -1) {
            shared_blk_write_cnt[warp_id_within_blk] = -rest_cnt_in_blk;
            d_MM->add_new_props(shared_prev_head[warp_id_within_blk], cur_query_vertex + 1, blk_partial_max_num);
            d_new_head = d_MM->mempool_to_write()->alloc();
            d_new_head_lower = (unsigned)d_new_head;
            d_new_head_upper = (unsigned)((unsigned long long)d_new_head >> 32);
        }
        __syncwarp();

        d_new_head_lower = __shfl_sync(FULL_MASK, d_new_head_lower, 0);
        d_new_head_upper = __shfl_sync(FULL_MASK, d_new_head_upper, 0);
        d_new_head = (int *)(((unsigned long long)d_new_head_upper << 32) |
                             (unsigned long long)d_new_head_lower);

        if (lane_id == 0) {
            shared_blk_write_cnt[warp_id_within_blk] += __popc(flag_mask);
        }
        __syncwarp();

        int *blk_to_write = shared_prev_head[warp_id_within_blk];
        if (thread_need_new_blk == true) {
            blk_to_write = d_new_head;
            shared_prev_head[warp_id_within_blk] = d_new_head;
        }
        __syncwarp();

        if (blk_to_write == nullptr) {
            *d_error_flag = 1;
            return;
        }

        if (flag) {
            int idx = thread_old_cnt * (cur_query_vertex + 1) + cur_query_vertex;
            blk_to_write[idx] = v;
            for (char i = 0; i < cur_query_vertex; i++) {
                idx = thread_old_cnt * (cur_query_vertex + 1) + i;
                blk_to_write[idx] = this_partial_matching[i];
            }
        }
    }

    if (lane_id == 0) {
        d_MM->prev_head[warp_id] = shared_prev_head[warp_id_within_blk];
        d_MM->blk_write_cnt[warp_id] = shared_blk_write_cnt[warp_id_within_blk];
    }
}