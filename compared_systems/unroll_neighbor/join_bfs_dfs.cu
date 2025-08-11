#include "join.h"
#include "params.h"
#include "dfs_stk.h"

#ifdef IDLE_CNT
__device__ static ull sample_cnt;
__device__ static ull idle_cnt;
#endif

__device__ void
init_dfs_stacks(
    Graph_GPU Q,
    Graph_GPU G,

    int *this_stk_len,
    stk_elem_fixed *this_stk_elem_fixed,
    stk_elem *this_stk_elem,

    int *d_partial_matchings,
    int *matching_order,
    int this_offset,
    int partial_matching_cnt,
    int lane_id,
    int begin_len
) {

    int partial_matching_id = this_offset + lane_id;
    stk_elem *e_ptr = this_stk_elem + begin_len;

    if (lane_id < UNROLL_MIN && partial_matching_id < partial_matching_cnt) {
        int *this_partial_matching = d_partial_matchings + partial_matching_id * 2;

        for (int depth = 0; depth < begin_len; depth++) {
            this_stk_elem_fixed[depth].mapped[lane_id] = this_partial_matching[depth];
        }

        // Calculate the minimum length of the candidate set for the current partial matching
        int next_u = matching_order[begin_len];
        int first_uu_index = Q.d_bknbrs_[Q.d_bknbrs_offset_[next_u]];
        int first_uu = matching_order[first_uu_index];
        int first_vv = this_partial_matching[first_uu_index];

        int min_len = 0;
        int *min_set = G.d_get_nbrs(first_vv, min_len);

        for (int k = Q.d_bknbrs_offset_[next_u] + 1; k < Q.d_bknbrs_offset_[next_u + 1]; k++) {
            int uu_index = Q.d_bknbrs_[k];
            int uu = matching_order[uu_index];
            int vv = this_partial_matching[uu_index];
            int this_len = 0;
            int *this_set = G.d_get_nbrs(vv, this_len);
            if (this_len < min_len) {
                min_len = this_len;
                min_set = this_set;
            }
        }

        e_ptr->cand_set[lane_id] = min_set;
        e_ptr->cand_len[lane_id] = min_len;
    }

    if (lane_id == 0) {
        if (this_offset + UNROLL_MIN > partial_matching_cnt) {
            e_ptr->unroll_size = partial_matching_cnt - this_offset;
        }
        else {
            e_ptr->unroll_size = UNROLL_MIN;
        }

        e_ptr->start_idx_within_set = 0;
        e_ptr->start_set_idx = 0;
        *this_stk_len = begin_len + 1;
    }
    __syncwarp();
}


__global__ void static
dfs_kernel_sym(
    Graph_GPU Q,
    Graph_GPU G,
    ull *sum,                       // Sum of the number of valid partial matchings
    int *d_matching_order,          // Matching order on device, length = Q.vcount()
    int *d_partial_matchings,
    int partial_matching_cnt,       // Total number of the beginning partial results
    int *begin_offset,
    int begin_len,
    int *d_partial_order
) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;    // Global thread id
    int warp_id = tid / warpSize;                       // Global warp id
    int warp_id_within_blk = warp_id % warpsPerBlock;   // Warp id within the block
    int lane_id = tid % warpSize;                       // Lane id within the warp

    __shared__ int s_matching_order[32];
    __shared__ int s_len[warpsPerBlock];
    __shared__ int s_partial_order[32];
    extern __shared__ char shared_mem[];

    stk_elem_fixed *s_stk_elem_fixed = (stk_elem_fixed *)shared_mem;
    stk_elem *s_stk_elem = (stk_elem *)(s_stk_elem_fixed + warpsPerBlock * begin_len);

    // Each warp has one stack
    int *this_stk_len = s_len + warp_id_within_blk;
    stk_elem_fixed *this_stk_elem_fixed = s_stk_elem_fixed + warp_id_within_blk * begin_len;
    stk_elem *this_stk_elem = s_stk_elem + warp_id_within_blk * (Q.vcount() - begin_len) - begin_len;

    if (threadIdx.x < Q.vcount()) {
        s_matching_order[threadIdx.x] = d_matching_order[threadIdx.x];
    }

    if (threadIdx.x < Q.vcount()) {
        s_partial_order[threadIdx.x] = d_partial_order[threadIdx.x];
    }

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

        // Initialize the stack with "UNROLL_MIN" beginning partial matchings
        init_dfs_stacks(Q, G,
                        this_stk_len, this_stk_elem_fixed, this_stk_elem,
                        d_partial_matchings, s_matching_order,
                        this_offset, partial_matching_cnt,
                        lane_id, begin_len);

restart:
        char current_state = 1;

        while (*this_stk_len > begin_len) {

            stk_elem *e_ptr = &this_stk_elem[*this_stk_len - 1];

            if (current_state == 1) {
                // Process the candidate set of the current stack element
                int lane_parent_idx = -1;
                int lane_idx_within_set = -1;
                int lane_v = -1;
                int this_u = s_matching_order[*this_stk_len - 1];

                int prefix_sum = -e_ptr->start_idx_within_set;
                for (int i = e_ptr->start_set_idx; i < e_ptr->unroll_size; i++) {
                    prefix_sum += e_ptr->cand_len[i];
                    if (prefix_sum > lane_id) {
                        lane_parent_idx = i;
                        lane_idx_within_set = lane_id - prefix_sum + e_ptr->cand_len[i];
                        lane_v = e_ptr->cand_set[lane_parent_idx][lane_idx_within_set];
                        break;
                    }
                }

                bool flag = true;
                if (lane_v == -1) {
                    flag = false;
                }
#ifndef UNLABELED
                if (flag && Q.label(this_u) != G.label(lane_v)) {
                    flag = false;
                }
#endif

#ifdef IDLE_CNT
                unsigned idle_mask = __ballot_sync(FULL_MASK, lane_v == -1);
                if (lane_id == 0) {
                    atomicAdd(&idle_cnt, __popc(idle_mask));
                    atomicAdd(&sample_cnt, 1);
                }
#endif

                if (lane_id == warpSize - 1) {
                    if (lane_v == -1) {
                        e_ptr->next_start_set_idx = -1;
                    }
                    else if (lane_idx_within_set == e_ptr->cand_len[lane_parent_idx] - 1) {
                        e_ptr->next_start_set_idx = lane_parent_idx + 1;
                        e_ptr->next_start_idx_within_set = 0;
                        if (e_ptr->next_start_set_idx == e_ptr->unroll_size) {
                            e_ptr->next_start_set_idx = -1;
                        }
                    }
                    else {
                        e_ptr->next_start_set_idx = lane_parent_idx;
                        e_ptr->next_start_idx_within_set = lane_idx_within_set + 1;
                    }
                }

                if (flag) {
                    int cur_parent = lane_parent_idx;
                    int j = *this_stk_len - 2;

                    // For each backward neighbor uu of this_u (enumerate bn_index in descending order)
                    for (int k = Q.d_bknbrs_offset_[this_u + 1] - 1; k >= Q.d_bknbrs_offset_[this_u]; k--) {
                        int bn_index = Q.d_bknbrs_[k];
                        int uu = s_matching_order[bn_index];
                        int vv = -1;

                        while (j > bn_index) {
                            if (j < begin_len) {
                                vv = this_stk_elem_fixed[j].mapped[cur_parent];
                            }
                            else {
                                vv = this_stk_elem[j].cand_set[this_stk_elem[j].parent_idx[cur_parent]]
                                     [this_stk_elem[j].mapped_idx[cur_parent]];
                                cur_parent = this_stk_elem[j].parent_idx[cur_parent];
                            }
                            j--;

                            if (vv == lane_v) {
                                // lane_v has been mapped before
                                flag = false;
                                break;
                            }
                            else if (s_partial_order[j + 1] & (1 << this_u)) {
                                if (lane_v < vv) {
                                    flag = false;
                                    break;
                                }
                            }
                        }

                        if (flag == false) {
                            break;
                        }

                        // Now j == bn_index
                        if (j < begin_len) {
                            vv = this_stk_elem_fixed[j].mapped[cur_parent];
                        }
                        else {
                            vv = this_stk_elem[j].cand_set[this_stk_elem[j].parent_idx[cur_parent]]
                                 [this_stk_elem[j].mapped_idx[cur_parent]];
                            cur_parent = this_stk_elem[j].parent_idx[cur_parent];
                        }
                        j--;

                        if (vv == lane_v) {
                            flag = false;
                            break;
                        }
                        else if (s_partial_order[uu] & (1 << this_u)) {
                            if (lane_v < vv) {
                                flag = false;
                                break;
                            }
                        }

                        int this_len = 0;
                        int *this_set = G.d_get_nbrs(vv, this_len);

                        if (this_set == e_ptr->cand_set[lane_parent_idx]) {
                            continue;
                        }

                        if (false == binary_search(this_set, this_len, lane_v)) {
                            flag = false;
                            break;
                        }
                    }

                    if (flag) {
                        while (j >= 0) {
                            int vv = 0;
                            if (j < begin_len) {
                                vv = this_stk_elem_fixed[j].mapped[cur_parent];
                            }
                            else {
                                vv = this_stk_elem[j].cand_set[this_stk_elem[j].parent_idx[cur_parent]]
                                     [this_stk_elem[j].mapped_idx[cur_parent]];
                                cur_parent = this_stk_elem[j].parent_idx[cur_parent];
                            }
                            if (vv == lane_v) {
                                flag = false;
                                break;
                            }
                            else if (s_partial_order[j] & (1 << this_u)) {
                                if (lane_v < vv) {
                                    flag = false;
                                    break;
                                }
                            }
                            j--;
                        }
                    }
                }

                unsigned int flag_mask = __ballot_sync(FULL_MASK, flag);
                e_ptr->warp_flag = flag_mask;
                current_state = 2;
            }

            if (e_ptr->warp_flag == 0) {
                if (lane_id == 0) {
                    current_state = 1;
                    e_ptr->start_set_idx = e_ptr->next_start_set_idx;
                    e_ptr->start_idx_within_set = e_ptr->next_start_idx_within_set;

                    if (e_ptr->start_set_idx == -1) {
                        current_state = 2;
                        (*this_stk_len)--;
                    }
                }
                current_state = __shfl_sync(FULL_MASK, current_state, 0);
            }
            else {
                // When e_ptr->warp_flag != 0
                if (*this_stk_len == Q.vcount()) {
                    if (lane_id == 0) {
                        int cnt = __popc(e_ptr->warp_flag);
                        sum[warp_id] += (ull)cnt;
                        e_ptr->warp_flag = 0;
                    }
                    __syncwarp();
                }
                else {
                    current_state = 1;
                    if (lane_id == 0) {
                        this_stk_elem[*this_stk_len].start_set_idx = 0;
                        this_stk_elem[*this_stk_len].start_idx_within_set = 0;
                        (*this_stk_len)++;
                    }
                    __syncwarp();

                    // "e_ptr" is the pointer of previous stack top
                    // "new_e_ptr" is the pointer of the level we are about to search
                    stk_elem *new_e_ptr = &this_stk_elem[*this_stk_len - 1];

                    // Select candidates at the current level
                    // Organize their minimum candidate sets
                    // Proceed to the next level

                    int chosen_index = -1;
                    if (e_ptr->warp_flag & (1 << lane_id)) {
                        unsigned mask_low = e_ptr->warp_flag & ((1 << (lane_id + 1)) - 1);
                        chosen_index = __popc(mask_low) - 1;
                    }

                    if (chosen_index >= 0 && chosen_index < UNROLL_MAX) {
                        int prefix_sum = -e_ptr->start_idx_within_set;
                        for (int i = e_ptr->start_set_idx; i < e_ptr->unroll_size; i++) {
                            prefix_sum += e_ptr->cand_len[i];
                            if (prefix_sum > lane_id) {
                                e_ptr->parent_idx[chosen_index] = i;
                                e_ptr->mapped_idx[chosen_index] = lane_id - prefix_sum + e_ptr->cand_len[i];
                                break;
                            }
                        }
                    }

                    if (chosen_index >= 0 && chosen_index < UNROLL_MAX) {
                        int j = *this_stk_len - 2;
                        int cur_parent = chosen_index;
                        int next_u = s_matching_order[*this_stk_len - 1];
                        int *min_set = nullptr;
                        int min_len = INF;

                        for (int i = Q.d_bknbrs_offset_[next_u + 1] - 1; i >= Q.d_bknbrs_offset_[next_u]; i--) {
                            int bn_index = Q.d_bknbrs_[i];
                            int uu = s_matching_order[bn_index];
                            int vv = -1;

                            while (j > bn_index) {
                                if (j >= begin_len) {
                                    cur_parent = this_stk_elem[j].parent_idx[cur_parent];
                                    j--;
                                }
                                else {
                                    j = bn_index;
                                    break;
                                }
                            }

                            // Now j == bn_index
                            if (j < begin_len) {
                                vv = this_stk_elem_fixed[j].mapped[cur_parent];
                            }
                            else {
                                vv = this_stk_elem[j].cand_set[this_stk_elem[j].parent_idx[cur_parent]]
                                     [this_stk_elem[j].mapped_idx[cur_parent]];
                            }

                            int this_len = 0;
                            int *this_set = G.d_get_nbrs(vv, this_len);

                            if (min_len > this_len) {
                                min_len = this_len;
                                min_set = this_set;
                            }
                        }

                        new_e_ptr->cand_set[chosen_index] = min_set;
                        new_e_ptr->cand_len[chosen_index] = min_len;
                    }

                    __syncwarp();

                    if (lane_id == 0) {
                        int c1 = __popc(e_ptr->warp_flag);
                        if (c1 > UNROLL_MAX) {
                            c1 = UNROLL_MAX;
                        }

                        new_e_ptr->unroll_size = c1;

                        unsigned tmp = e_ptr->warp_flag;
                        for (int t = 0; t < c1; t++) {
                            tmp &= tmp - 1;
                        }
                        e_ptr->warp_flag = tmp;
                    }
                    __syncwarp();
                }
            }
        }
    }
}

ull
join_bfs_dfs_sym(
    Graph &q,
    Graph &g,
    Graph_GPU Q,
    Graph_GPU G,
    std::vector<int> &matching_order,
    std::vector<uint32_t> &partial_order
) {
#ifdef IDLE_CNT
    cudaCheck(cudaMemcpyToSymbol(sample_cnt, &Zero_ull, sizeof(ull)));
    cudaCheck(cudaMemcpyToSymbol(idle_cnt, &Zero_ull, sizeof(ull)));
#endif

    TIME_INIT();
    TIME_START();

    int partial_matching_cnt = 0;
    int *d_partial_matchings = match_first_edge_sym(q, g, Q, G, matching_order, partial_matching_cnt, partial_order);
    if (partial_matching_cnt == 0) {
        return 0;
    }

    TIME_END();
    PRINT_LOCAL_TIME("Match first edge");
    printf("Match first edge: %d\n", partial_matching_cnt);

    int *d_matching_order = nullptr;
    cudaCheck(cudaMalloc(&d_matching_order, sizeof(int) * q.vcount()));
    cudaCheck(cudaMemcpy(d_matching_order, matching_order.data(), sizeof(int) * q.vcount(), cudaMemcpyHostToDevice));

    check_gpu_memory();

    
    const int l = 2;
    printf("Conducting DFS from level %d to %d\n", l + 1, Q.vcount());
    
    ull *d_sum = nullptr;
    cudaCheck(cudaMalloc(&d_sum, sizeof(ull) * warpNum));
    cudaCheck(cudaMemset(d_sum, 0, sizeof(ull) * warpNum));
    
    int *begin_offset = nullptr;
    cudaCheck(cudaMalloc(&begin_offset, sizeof(int)));
    cudaCheck(cudaMemset(begin_offset, 0, sizeof(int)));
    
    int shared_size = warpsPerBlock * l * sizeof(stk_elem_fixed)
        + warpsPerBlock * (Q.vcount() - l) * sizeof(stk_elem);
    
    int *d_partial_order = nullptr;
    cudaCheck(cudaMalloc(&d_partial_order, sizeof(uint32_t) * q.vcount()));
    cudaCheck(cudaMemcpy(d_partial_order, partial_order.data(), sizeof(uint32_t) * q.vcount(), cudaMemcpyHostToDevice));

    TIME_START();

    dfs_kernel_sym <<< maxBlocks, threadsPerBlock, shared_size >>> (
        Q, G, d_sum,
        d_matching_order,
        d_partial_matchings, partial_matching_cnt,
        begin_offset, l,
        d_partial_order
    );
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    TIME_END();
    PRINT_LOCAL_TIME("Search");

    ull *h_sum = (ull *)malloc(sizeof(ull) * warpNum);
    cudaCheck(cudaMemcpy(h_sum, d_sum, sizeof(ull) * warpNum, cudaMemcpyDeviceToHost));

    ull ret = 0;
    for (int i = 0; i < warpNum; i++) {
        ret += h_sum[i];
    }

    free(h_sum);
    cudaCheck(cudaFree(d_sum));
    cudaCheck(cudaFree(d_matching_order));

#ifdef IDLE_CNT
    ull h_sample_cnt = 0, h_idle_cnt = 0;
    cudaCheck(cudaMemcpyFromSymbol(&h_sample_cnt, sample_cnt, sizeof(ull)));
    cudaCheck(cudaMemcpyFromSymbol(&h_idle_cnt, idle_cnt, sizeof(ull)));
    double idle_rate = (double)h_idle_cnt / (h_sample_cnt * 32.0);
    printf("Sample Count: %lld. Idle Count: %lld. Idle Rate: %.2lf%%\n", h_sample_cnt, h_idle_cnt, idle_rate * 100);
#endif

    return ret;
}