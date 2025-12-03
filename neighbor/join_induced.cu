#include "join.h"
#include "candidate.h"
#include "params.h"
#include "dfs_stk.h"


static __device__ void
init_dfs_stacks_induced(
    candidate_graph_GPU cg,
    stk_elem_fixed *this_stk_elem_fixed,
    stk_elem *this_stk_elem,
    MemManager *d_MM,
    int this_offset,
    int *warp_buffer
) {
    // This function assumes that UNROLL_MIN == 1 and that BEGIN_LEN == 2
    int *this_partial_matching = d_MM->get_partial(this_offset);
    int v0 = this_partial_matching[0];
    int v1 = this_partial_matching[1];
    int nbr0_len, nbr1_len;
    int *nbr0 = cg.get_nbr(v0, nbr0_len);
    int *nbr1 = cg.get_nbr(v1, nbr1_len);

    int intersection_length = warp_set_intersection(nbr0, nbr0_len, nbr1, nbr1_len, warp_buffer);

    this_stk_elem_fixed[0].mapped_v[0] = v1;
    this_stk_elem_fixed[0].mapped_v_nbr_set[0] = warp_buffer;
    this_stk_elem_fixed[0].mapped_v_nbr_len[0] = intersection_length;

    this_stk_elem[1].cand_set[0] = warp_buffer;
    this_stk_elem[1].cand_len[0] = intersection_length;
    this_stk_elem[1].cand_len[1] = -1;
    this_stk_elem[1].start_idx_within_set = 0;
    this_stk_elem[1].start_set_idx = 0;
}


__global__ void static
dfs_kernel_induced(
    Graph_GPU Q,
    Graph_GPU G,
    candidate_graph_GPU cg,
    ull *sum,
    MemManager *d_MM,
    int partial_matching_cnt,
    int *begin_offset,
    int *buffer
) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = tid / warpSize;
    int warp_id_within_blk = warp_id % warpsPerBlock;
    int lane_id = tid % warpSize;

    const int BEGIN_LEN = 1;

    // Each warp has a buffer with length MAX_DEGREE
    int *warp_buffer = buffer + warp_id * MAX_DEGREE;

    extern __shared__ char shared_mem[];

    stk_elem_fixed *s_stk_elem_fixed = (stk_elem_fixed *)shared_mem;
    stk_elem *s_stk_elem = (stk_elem *)(s_stk_elem_fixed + warpsPerBlock * BEGIN_LEN);

    stk_elem_fixed *this_stk_elem_fixed = s_stk_elem_fixed + warp_id_within_blk * BEGIN_LEN;
    stk_elem *this_stk_elem = s_stk_elem + warp_id_within_blk * (Q.vcount() - 1 - BEGIN_LEN) - BEGIN_LEN;

    __syncthreads();

    int this_offset = -1;

    while (true) {
        if (lane_id == 0) {
            this_offset = atomicAdd(begin_offset, UNROLL_MIN);
        }
        this_offset = __shfl_sync(FULL_MASK, this_offset, 0);

        if (this_offset >= partial_matching_cnt) {
            break;
        }

        init_dfs_stacks_induced(cg, this_stk_elem_fixed, this_stk_elem, d_MM, this_offset, warp_buffer);

        int this_stk_len = BEGIN_LEN + 1;

        while (this_stk_len > BEGIN_LEN) {
            int this_u = this_stk_len - 1;
            stk_elem *e_ptr = &this_stk_elem[this_u];

            int lane_parent_idx = -1;
            int lane_idx_within_set = -1;
            int lane_v = -1;

            int prefix_sum = -e_ptr->start_idx_within_set;
            for (int i = e_ptr->start_set_idx; e_ptr->cand_len[i] != -1; i++) {
                prefix_sum += e_ptr->cand_len[i];
                if (prefix_sum > lane_id) {
                    lane_parent_idx = i;
                    lane_idx_within_set = lane_id - prefix_sum + e_ptr->cand_len[i];
                    lane_v = __ldg(&e_ptr->cand_set[lane_parent_idx][lane_idx_within_set]);
                    break;
                }
            }

            bool flag = true;
            if (lane_v == -1) {
                flag = false;
            }

            if (lane_id == warpSize - 1) {
                if (lane_v == -1) {
                    e_ptr->start_set_idx = -1;
                }
                else if (lane_idx_within_set == e_ptr->cand_len[lane_parent_idx] - 1) {
                    e_ptr->start_set_idx = lane_parent_idx + 1;
                    e_ptr->start_idx_within_set = 0;
                    if (e_ptr->cand_len[e_ptr->start_set_idx] == -1) {
                        e_ptr->start_set_idx = -1;
                    }
                }
                else {
                    e_ptr->start_set_idx = lane_parent_idx;
                    e_ptr->start_idx_within_set = lane_idx_within_set + 1;
                }
            }

            if (flag) {
                int cur_parent = lane_parent_idx;
                bool reach_bound = false;

                for (int uu = this_stk_len - 2; uu >= 0; uu--) {
                    int vv;
                    int old_cur_parent = cur_parent;
                    if (uu < BEGIN_LEN) {
                        vv = this_stk_elem_fixed[uu].mapped_v[cur_parent];
                    }
                    else {
                        vv = this_stk_elem[uu].mapped_v[cur_parent];
                        cur_parent = this_stk_elem[uu].parent_idx[cur_parent];
                    }

                    if (lane_v >= vv) {
                        flag = false;
                        reach_bound = true;
                        break;
                    }

                    int this_len = 0;
                    int *this_set = nullptr;

                    if (uu < BEGIN_LEN) {
                        this_len = this_stk_elem_fixed[uu].mapped_v_nbr_len[old_cur_parent];
                        this_set = this_stk_elem_fixed[uu].mapped_v_nbr_set[old_cur_parent];
                    }
                    else {
                        this_set = cg.get_nbr(vv, this_len);
                    }

                    if (this_set == e_ptr->cand_set[lane_parent_idx]) {
                        continue;
                    }

                    if (false == binary_search_int(this_set, this_len, lane_v)) {
                        flag = false;
                        break;
                    }
                }

                if (lane_id == warpSize - 1 && reach_bound && lane_v != -1) {
                    e_ptr->start_set_idx = lane_parent_idx + 1;
                    e_ptr->start_idx_within_set = 0;
                    if (e_ptr->cand_len[e_ptr->start_set_idx] == -1) {
                        e_ptr->start_set_idx = -1;
                    }
                }
            }

            unsigned int flag_mask = __ballot_sync(FULL_MASK, flag);

            if (flag_mask == 0) {
                if (lane_id == 0) {
                    do {
                        if (this_stk_elem[this_stk_len - 1].start_set_idx != -1) {
                            break;
                        }
                        else {
                            this_stk_len--;
                        }
                    }
                    while (this_stk_len > BEGIN_LEN);
                }
                this_stk_len = __shfl_sync(FULL_MASK, this_stk_len, 0);
            }
            else {
                if (this_stk_len == Q.vcount_ - 1) {
                    if (lane_id == 0) {
                        sum[warp_id] += (ull)__popc(flag_mask);
                        do {
                            if (this_stk_elem[this_stk_len - 1].start_set_idx != -1) {
                                break;
                            }
                            else {
                                this_stk_len--;
                            }
                        }
                        while (this_stk_len > BEGIN_LEN);
                    }
                    this_stk_len = __shfl_sync(FULL_MASK, this_stk_len, 0);
                }
                else {
                    if (lane_id == 0) {
                        this_stk_elem[this_stk_len].start_set_idx = 0;
                        this_stk_elem[this_stk_len].start_idx_within_set = 0;
                    }
                    this_stk_len++;

                    stk_elem *new_e_ptr = &this_stk_elem[this_stk_len - 1];

                    if (lane_id == 0) {
                        new_e_ptr->cand_len[__popc(flag_mask)] = -1;
                    }
                    __syncwarp();

                    int chosen_index = -1;
                    if (flag_mask & (1 << lane_id)) {
                        unsigned mask_low = flag_mask & ((1 << (lane_id + 1)) - 1);
                        chosen_index = __popc(mask_low) - 1;
                    }

                    if (chosen_index >= 0) {
                        e_ptr->parent_idx[chosen_index] = lane_parent_idx;
                        e_ptr->mapped_v[chosen_index] = lane_v;

                        int min_len = 0;
                        int *min_set = cg.get_nbr(lane_v, min_len);

                        if (min_len > e_ptr->cand_len[lane_parent_idx]) {
                            min_len = e_ptr->cand_len[lane_parent_idx];
                            min_set = e_ptr->cand_set[lane_parent_idx];
                        }

                        new_e_ptr->cand_set[chosen_index] = min_set;
                        new_e_ptr->cand_len[chosen_index] = min_len;
                    }
                    __syncwarp();
                }
            }
        }
    }
}


__global__ void static __launch_bounds__(512, 3)
dfs_kernel_induced_orientation(
    Graph_GPU Q,
    Graph_GPU G,
    candidate_graph_GPU cg,
    ull *sum,
    MemManager *d_MM,
    int partial_matching_cnt,
    int *begin_offset,
    int *buffer
) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = tid / warpSize;
    int warp_id_within_blk = warp_id % warpsPerBlock;
    int lane_id = tid % warpSize;

    const int BEGIN_LEN = 1;

    // Each warp has a buffer with length MAX_DEGREE
    int *warp_buffer = buffer + warp_id * MAX_DEGREE;

    extern __shared__ char shared_mem[];

    stk_elem_fixed *s_stk_elem_fixed = (stk_elem_fixed *)shared_mem;
    stk_elem *s_stk_elem = (stk_elem *)(s_stk_elem_fixed + warpsPerBlock * BEGIN_LEN);

    stk_elem_fixed *this_stk_elem_fixed = s_stk_elem_fixed + warp_id_within_blk * BEGIN_LEN;
    stk_elem *this_stk_elem = s_stk_elem + warp_id_within_blk * (Q.vcount() - 1 - BEGIN_LEN) - BEGIN_LEN;

    __syncthreads();

    int this_offset = -1;

    while (true) {
        if (lane_id == 0) {
            this_offset = atomicAdd(begin_offset, UNROLL_MIN);
        }
        this_offset = __shfl_sync(FULL_MASK, this_offset, 0);

        if (this_offset >= partial_matching_cnt) {
            break;
        }

        init_dfs_stacks_induced(cg, this_stk_elem_fixed, this_stk_elem, d_MM, this_offset, warp_buffer);

        int this_stk_len = BEGIN_LEN + 1;

        while (this_stk_len > BEGIN_LEN) {
            stk_elem *e_ptr = &this_stk_elem[this_stk_len - 1];

            int lane_parent_idx = -1;
            int lane_idx_within_set = -1;
            int lane_v = -1;

            int prefix_sum = -e_ptr->start_idx_within_set;
            for (int i = e_ptr->start_set_idx; e_ptr->cand_len[i] != -1; i++) {
                prefix_sum += e_ptr->cand_len[i];
                if (prefix_sum > lane_id) {
                    lane_parent_idx = i;
                    lane_idx_within_set = lane_id - prefix_sum + e_ptr->cand_len[i];
                    lane_v = __ldg(&e_ptr->cand_set[lane_parent_idx][lane_idx_within_set]);
                    break;
                }
            }

            if (lane_id == warpSize - 1) {
                if (lane_v == -1) {
                    e_ptr->start_set_idx = -1;
                }
                else if (lane_idx_within_set == e_ptr->cand_len[lane_parent_idx] - 1) {
                    e_ptr->start_set_idx = lane_parent_idx + 1;
                    e_ptr->start_idx_within_set = 0;
                    if (e_ptr->cand_len[e_ptr->start_set_idx] == -1) {
                        e_ptr->start_set_idx = -1;
                    }
                }
                else {
                    e_ptr->start_set_idx = lane_parent_idx;
                    e_ptr->start_idx_within_set = lane_idx_within_set + 1;
                }
            }

            bool flag = (lane_v != -1);

            if (flag) {
                int cur_parent = lane_parent_idx;

                for (int uu = this_stk_len - 2; uu >= 0; uu--) {
                    int this_len = 0;
                    int *this_set = nullptr;

                    if (uu < BEGIN_LEN) {
                        this_len = this_stk_elem_fixed[uu].mapped_v_nbr_len[cur_parent];
                        this_set = this_stk_elem_fixed[uu].mapped_v_nbr_set[cur_parent];
                    }
                    else {
                        this_set = cg.get_nbr(this_stk_elem[uu].mapped_v[cur_parent], this_len);
                        cur_parent = this_stk_elem[uu].parent_idx[cur_parent];
                    }

                    if (this_set == e_ptr->cand_set[lane_parent_idx]) {
                        continue;
                    }

                    if (false == binary_search_int(this_set, this_len, lane_v)) {
                        flag = false;
                        break;
                    }
                }
            }

            unsigned int flag_mask = __ballot_sync(FULL_MASK, flag);

            if (flag_mask == 0) {
                if (lane_id == 0) {
                    do {
                        if (this_stk_elem[this_stk_len - 1].start_set_idx != -1) {
                            break;
                        }
                        else {
                            this_stk_len--;
                        }
                    }
                    while (this_stk_len > BEGIN_LEN);
                }
                this_stk_len = __shfl_sync(FULL_MASK, this_stk_len, 0);
            }
            else {
                if (this_stk_len == Q.vcount_ - 1) {
                    if (lane_id == 0) {
                        sum[warp_id] += (ull)__popc(flag_mask);
                        do {
                            if (this_stk_elem[this_stk_len - 1].start_set_idx != -1) {
                                break;
                            }
                            else {
                                this_stk_len--;
                            }
                        }
                        while (this_stk_len > BEGIN_LEN);
                    }
                    this_stk_len = __shfl_sync(FULL_MASK, this_stk_len, 0);
                }
                else {
                    if (lane_id == 0) {
                        this_stk_elem[this_stk_len].start_set_idx = 0;
                        this_stk_elem[this_stk_len].start_idx_within_set = 0;
                    }
                    this_stk_len++;

                    stk_elem *new_e_ptr = &this_stk_elem[this_stk_len - 1];

                    if (lane_id == 0) {
                        new_e_ptr->cand_len[__popc(flag_mask)] = -1;
                    }
                    __syncwarp();

                    int chosen_index = -1;
                    if (flag_mask & (1 << lane_id)) {
                        unsigned mask_low = flag_mask & ((1 << (lane_id + 1)) - 1);
                        chosen_index = __popc(mask_low) - 1;
                    }

                    if (chosen_index >= 0) {
                        e_ptr->parent_idx[chosen_index] = lane_parent_idx;
                        e_ptr->mapped_v[chosen_index] = lane_v;

                        int min_len = 0;
                        int *min_set = cg.get_nbr(lane_v, min_len);

                        if (min_len > e_ptr->cand_len[lane_parent_idx]) {
                            min_len = e_ptr->cand_len[lane_parent_idx];
                            min_set = e_ptr->cand_set[lane_parent_idx];
                        }

                        new_e_ptr->cand_set[chosen_index] = min_set;
                        new_e_ptr->cand_len[chosen_index] = min_len;
                    }
                    __syncwarp();
                }
            }
        }
    }
}


ull
join_induced(
    Graph &q,
    Graph &g,
    Graph_GPU &Q,
    Graph_GPU &G,
    candidate_graph &_cg,
    candidate_graph_GPU &cg,
    std::vector<uint32_t> &partial_order
) {

    auto data_preparation_start = std::chrono::high_resolution_clock::now();
    int partial_matching_cnt = 0;
    int *d_partial_matchings = set_beginning_partial_matchings_sym(q, g, cg, partial_matching_cnt, partial_order);
    if (partial_matching_cnt == 0) {
        return 0;
    }

    MemManager h_MM(true);
    h_MM.init(d_partial_matchings, partial_matching_cnt, 0, partial_matching_cnt);
    MemManager *d_MM = nullptr;
    cudaCheck(cudaMalloc(&d_MM, sizeof(MemManager)));
    get_partial_init(&h_MM);
    h_MM.init_prev_head();
    cudaCheck(cudaMemcpy(d_MM, &h_MM, sizeof(MemManager), cudaMemcpyHostToDevice));

    auto data_preparation_end = std::chrono::high_resolution_clock::now();
    auto data_preparation_duration = std::chrono::duration_cast<std::chrono::microseconds>(data_preparation_end - data_preparation_start);
    std::cout << "Data preparation: " << data_preparation_duration.count() / 1000 << " ms\n";

    ull *d_sum = nullptr;
    cudaCheck(cudaMalloc(&d_sum, sizeof(ull) * warpNum));
    cudaCheck(cudaMemset(d_sum, 0, sizeof(ull) * warpNum));

    int *begin_offset = nullptr;
    cudaCheck(cudaMalloc(&begin_offset, sizeof(int)));
    cudaCheck(cudaMemset(begin_offset, 0, sizeof(int)));

    int dynamic_shared_size = warpsPerBlock * (Q.vcount() - 2) * sizeof(stk_elem) + warpsPerBlock * sizeof(stk_elem_fixed);
    printf("Shared memory usage: %.2f KB per thread block\n", dynamic_shared_size / 1024.);
    printf("DFS kernel theoretical occupancy %.2f%%\n", calculateOccupancy((const void *)dfs_kernel_induced, threadsPerBlock, dynamic_shared_size));

    int thread_block_num = 0;
    int current_device;
    cudaGetDevice(&current_device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, current_device);
    int num_SMs = prop.multiProcessorCount;

    int *buffer = nullptr;
    printf("Allocating %.2lf MB for warp buffers.\n",  sizeof(int) * warpNum * MAX_DEGREE / 1024. / 1024.);
    cudaCheck(cudaMalloc(&buffer, sizeof(int) * warpNum * MAX_DEGREE));

    TIME_INIT();
    TIME_START();

    if (!q.is_clique()) {
        printf("join_induced only supports clique.\n");
        exit(1);
    }
    else {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&thread_block_num, dfs_kernel_induced, threadsPerBlock, dynamic_shared_size);
        thread_block_num = thread_block_num * num_SMs;
        printf("#thread blocks per SM: %d, #SMs: %d, #threads per block: %d\n", thread_block_num / num_SMs, num_SMs, threadsPerBlock);
        dfs_kernel_induced <<< thread_block_num, threadsPerBlock, dynamic_shared_size >>> (
            Q, G, cg, d_sum,
            d_MM, partial_matching_cnt,
            begin_offset, buffer
        );
    }
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    TIME_END();
    PRINT_TOTAL_TIME("Processing");

    ull *h_sum = (ull *)malloc(sizeof(ull) * warpNum);
    cudaCheck(cudaMemcpy(h_sum, d_sum, sizeof(ull) * warpNum, cudaMemcpyDeviceToHost));

    ull ret = 0;
    for (int i = 0; i < warpNum; i++) {
        ret += h_sum[i];
    }

    h_MM.deallocate();
    free(h_sum);
    cudaCheck(cudaFree(d_sum));
    cudaCheck(cudaFree(begin_offset));

    return ret;
}


ull
join_induced_orientation(
    Graph &q,
    Graph &g,
    Graph_GPU &Q,
    Graph_GPU &G,
    candidate_graph &_cg,
    candidate_graph_GPU &cg
) {

    auto data_preparation_start = std::chrono::high_resolution_clock::now();
    int partial_matching_cnt = 0;
    int *d_partial_matchings = set_beginning_partial_matchings(q, g, cg, partial_matching_cnt);
    if (partial_matching_cnt == 0) {
        return 0;
    }

    MemManager h_MM(true);
    h_MM.init(d_partial_matchings, partial_matching_cnt, 0, partial_matching_cnt);
    MemManager *d_MM = nullptr;
    cudaCheck(cudaMalloc(&d_MM, sizeof(MemManager)));
    get_partial_init(&h_MM);
    h_MM.init_prev_head();
    cudaCheck(cudaMemcpy(d_MM, &h_MM, sizeof(MemManager), cudaMemcpyHostToDevice));

    auto data_preparation_end = std::chrono::high_resolution_clock::now();
    auto data_preparation_duration = std::chrono::duration_cast<std::chrono::microseconds>(data_preparation_end - data_preparation_start);
    std::cout << "Data preparation: " << data_preparation_duration.count() / 1000 << " ms\n";

    ull *d_sum = nullptr;
    cudaCheck(cudaMalloc(&d_sum, sizeof(ull) * warpNum));
    cudaCheck(cudaMemset(d_sum, 0, sizeof(ull) * warpNum));

    int *begin_offset = nullptr;
    cudaCheck(cudaMalloc(&begin_offset, sizeof(int)));
    cudaCheck(cudaMemset(begin_offset, 0, sizeof(int)));

    int dynamic_shared_size = warpsPerBlock * (Q.vcount() - 2) * sizeof(stk_elem) + warpsPerBlock * sizeof(stk_elem_fixed);
    printf("Shared memory usage: %.2f KB per thread block\n", dynamic_shared_size / 1024.);
    printf("DFS kernel theoretical occupancy %.2f%%\n", calculateOccupancy((const void *)dfs_kernel_induced_orientation, threadsPerBlock, dynamic_shared_size));

    int thread_block_num = 0;
    int current_device;
    cudaGetDevice(&current_device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, current_device);
    int num_SMs = prop.multiProcessorCount;

    int *buffer = nullptr;
    printf("Allocating %.2lf MB for warp buffers.\n",  sizeof(int) * warpNum * MAX_DEGREE / 1024. / 1024.);
    cudaCheck(cudaMalloc(&buffer, sizeof(int) * warpNum * MAX_DEGREE));

    TIME_INIT();
    TIME_START();

    if (!q.is_clique()) {
        printf("join_induced_orientation only supports clique.\n");
        exit(1);
    }
    else {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&thread_block_num, dfs_kernel_induced_orientation, threadsPerBlock, dynamic_shared_size);
        thread_block_num = thread_block_num * num_SMs;
        printf("#thread blocks per SM: %d, #SMs: %d, #threads per block: %d\n", thread_block_num / num_SMs, num_SMs, threadsPerBlock);
        dfs_kernel_induced_orientation <<< thread_block_num, threadsPerBlock, dynamic_shared_size >>> (
            Q, G, cg, d_sum,
            d_MM, partial_matching_cnt,
            begin_offset, buffer
        );
    }
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    TIME_END();
    PRINT_TOTAL_TIME("Processing");

    ull *h_sum = (ull *)malloc(sizeof(ull) * warpNum);
    cudaCheck(cudaMemcpy(h_sum, d_sum, sizeof(ull) * warpNum, cudaMemcpyDeviceToHost));

    ull ret = 0;
    for (int i = 0; i < warpNum; i++) {
        ret += h_sum[i];
    }

    h_MM.deallocate();
    free(h_sum);
    cudaCheck(cudaFree(d_sum));
    cudaCheck(cudaFree(begin_offset));

    return ret;
}