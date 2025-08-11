#include "join.h"
#include "candidate.h"
#include "params.h"
#include "dfs_stk.h"

#ifdef IDLE_CNT
__device__ static ull sample_cnt;
__device__ static ull idle_cnt;
#endif

#ifdef BALANCE_CNT
__device__ static ull compute_cnt[warpNum];
__device__ static ull warp_time[warpNum];
__device__ static ull assign_stop_time;
#endif

__device__ void
init_dfs_stacks(
    const Graph_GPU Q,
    const candidate_graph_GPU cg,

    int *this_stk_len,
    stk_elem_fixed *this_stk_elem_fixed,
    stk_elem *this_stk_elem,

    MemManager *d_MM,
    int *matching_order,
    int this_offset,
    int partial_matching_cnt,
    int lane_id,
    const int begin_len
) {

    int partial_matching_id = this_offset + lane_id;
    stk_elem *e_ptr = this_stk_elem + begin_len;

    if (lane_id < UNROLL_MIN && partial_matching_id < partial_matching_cnt) {
        int *this_partial_matching = d_MM->get_partial(partial_matching_id);

        for (int depth = 0; depth < begin_len; depth++) {
            this_stk_elem_fixed[depth].mapped[lane_id] = this_partial_matching[depth];
        }

        // Calculate the minimum length of the candidate set for the current partial matching
        int next_u = matching_order[begin_len];
        int first_uu_index = Q.d_bknbrs_[Q.d_bknbrs_offset_[next_u]];
        int first_uu = matching_order[first_uu_index];
        int first_vv = this_partial_matching[first_uu_index];

        int min_len = 0;
        int *min_set = cg.d_get_candidates(first_uu, next_u, first_vv, min_len);

        for (int k = Q.d_bknbrs_offset_[next_u] + 1; k < Q.d_bknbrs_offset_[next_u + 1]; k++) {
            int uu_index = Q.d_bknbrs_[k];
            int uu = matching_order[uu_index];
            int vv = this_partial_matching[uu_index];
            int this_len = 0;
            int *this_set = cg.d_get_candidates(uu, next_u, vv, this_len);
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
dfs_kernel(
    const Graph_GPU Q,
    const Graph_GPU G,
    const candidate_graph_GPU cg,
    ull *sum,                       // Sum of the number of valid partial matchings
    int *d_matching_order,          // Matching order on device, length = Q.vcount()
    MemManager *d_MM,               // Where previous partial results are obtained
    int partial_matching_cnt,       // Total number of the beginning partial results
    int *begin_offset,
    const int begin_len
) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;    // Global thread id
    int warp_id = tid / warpSize;                       // Global warp id
    int warp_id_within_blk = warp_id % warpsPerBlock;   // Warp id within the block
    int lane_id = tid % warpSize;                       // Lane id within the warp

    __shared__ int s_len[warpsPerBlock];

#ifdef WORK_STEALING
    __shared__ int steal_states[warpsPerBlock];         // Signal denoting whether this warp is active(-1), unavailable(-2), refused(-3), or requested(>=0)
    __shared__ int active_warps;                        // Number of active warps

    if (threadIdx.x == 0) {
        active_warps = warpsPerBlock;
    }

    if (lane_id == 0) {
        steal_states[warp_id_within_blk] = -1;
    }
#endif

    extern __shared__ char shared_mem[];

    stk_elem_fixed *s_stk_elem_fixed = (stk_elem_fixed *)shared_mem;
    stk_elem *s_stk_elem = (stk_elem *)(s_stk_elem_fixed + warpsPerBlock * begin_len);

#ifdef BALANCE_CNT
    if (lane_id == 0) {
        compute_cnt[warp_id] = 0;
        warp_time[warp_id] = clock64();
    }
#endif

    // Each warp has one stack
    int *this_stk_len = s_len + warp_id_within_blk;
    stk_elem_fixed *this_stk_elem_fixed = s_stk_elem_fixed + warp_id_within_blk * begin_len;
    stk_elem *this_stk_elem = s_stk_elem + warp_id_within_blk * (Q.vcount() - begin_len) - begin_len;

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
        init_dfs_stacks(Q, cg,
                        this_stk_len, this_stk_elem_fixed, this_stk_elem,
                        d_MM, d_matching_order,
                        this_offset, partial_matching_cnt,
                        lane_id, begin_len);

restart:
        char current_state = 1;

        while (*this_stk_len > begin_len) {

#ifdef WORK_STEALING
            if (lane_id == 0) {
                int requested_id = atomicAdd(steal_states + warp_id_within_blk, 0);
                if (requested_id >= 0) {
                    char *that_stk_status = s_status + requested_id * Q.vcount();
                    stk_elem_fixed *that_stk_elem_fixed = s_stk_elem_fixed + requested_id * begin_len;
                    stk_elem *that_stk_elem = s_stk_elem + requested_id * (Q.vcount() - begin_len) - begin_len;

                    for (int j = 0; j < begin_len; j++) {
                        that_stk_elem_fixed[j] = this_stk_elem_fixed[j];
                    }

                    bool success = false;
                    for (int i = 0; begin_len + i < *this_stk_len && i < 2; i++) {
                        if (this_stk_status[begin_len + i] == 2 && this_stk_elem[begin_len + i].next_start_set_idx != -1) {
                            for (int j = 0; j < i; j++) {
                                that_stk_status[begin_len + j] = 2;
                            }
                            that_stk_status[begin_len + i] = 1;

                            s_len[requested_id] = begin_len + i + 1;

                            for (int j = 0; j <= i; j++) {
                                that_stk_elem[begin_len + j] = this_stk_elem[begin_len + j];
                            }

                            that_stk_elem[begin_len + i].start_set_idx = that_stk_elem[begin_len + i].next_start_set_idx;
                            that_stk_elem[begin_len + i].start_idx_within_set = that_stk_elem[begin_len + i].next_start_idx_within_set;

                            this_stk_elem[begin_len + i].next_start_set_idx = -1;

                            for (int j = 0; j < i; j++) {
                                this_stk_elem[begin_len + j].warp_flag = 0;
                            }

                            __threadfence_block();

                            atomicExch(steal_states + requested_id, -1);
                            atomicExch(steal_states + warp_id_within_blk, -1);
                            success = true;
                            break;
                        }
                    }

                    if (!success) {
                        atomicExch(steal_states + requested_id, -3);
                        __threadfence_block();
                        // Mark this warp as unavailable to ignore any future request
                        atomicExch(steal_states + warp_id_within_blk, -2);
                        atomicSub(&active_warps, 1);
                    }
                }
            }
            __syncwarp();
#endif

            stk_elem *e_ptr = &this_stk_elem[*this_stk_len - 1];

            if (current_state == 1) {
                // Process the candidate set of the current stack element
                int lane_parent_idx = -1;
                int lane_idx_within_set = -1;
                int lane_v = -1;
                int this_u = d_matching_order[*this_stk_len - 1];

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

#ifdef IDLE_CNT
                unsigned idle_mask = __ballot_sync(FULL_MASK, lane_v == -1);
                if (lane_id == 0) {
                    atomicAdd(&idle_cnt, __popc(idle_mask));
                    atomicAdd(&sample_cnt, 1);
                }
#endif

#ifdef BALANCE_CNT
                if (lane_id == 0) {
                    compute_cnt[warp_id]++;
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
                        int uu = d_matching_order[bn_index];
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

                        int this_len = 0;
                        int *this_set = cg.d_get_candidates(uu, this_u, vv, this_len);

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
                        int next_u = d_matching_order[*this_stk_len - 1];
                        int *min_set = nullptr;
                        int min_len = INF;

                        for (int i = Q.d_bknbrs_offset_[next_u + 1] - 1; i >= Q.d_bknbrs_offset_[next_u]; i--) {
                            int bn_index = Q.d_bknbrs_[i];
                            int uu = d_matching_order[bn_index];
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
                            int *this_set = cg.d_get_candidates(uu, next_u, vv, this_len);

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

#ifdef BALANCE_CNT
    if (lane_id == 0) {
        atomicCAS(&assign_stop_time, 0, (clock64() - warp_time[warp_id]) / 10000);
    }
#endif

#ifdef WORK_STEALING
    bool request_success = false;

    if (lane_id == 0) {
        if (atomicAdd(steal_states + warp_id_within_blk, 0) != -2) {
            atomicExch(steal_states + warp_id_within_blk, -2);
            atomicSub(&active_warps, 1);
        }

        while (atomicAdd(&active_warps, 0) > 0) {
            int target = -1;
            for (int i = 1; i < warpsPerBlock; i++) {
                int t = (warp_id_within_blk + i) % warpsPerBlock;
                if (-1 == atomicCAS(steal_states + t, -1, warp_id_within_blk)) {
                    target = t;
                    break;
                }
            }

            if (target != -1) {
                while (true) {
                    int tmp = atomicAdd(steal_states + warp_id_within_blk, 0);
                    if (tmp == -1 || tmp >= 0) {
                        atomicAdd(&active_warps, 1);
                        request_success = true;
                        break;
                    }
                    else if (tmp == -3) {
                        atomicExch(steal_states + warp_id_within_blk, -2);
                        break;
                    }
                    else if (-2 == atomicAdd(steal_states + target, 0)) {
                        if (-2 == atomicAdd(steal_states + warp_id_within_blk, 0)) {
                            break;
                        }
                    }
                    else if (atomicAdd(&active_warps, 0) <= 0) {
                        break;
                    }
                    __nanosleep(100);
                }
                if (request_success) {
                    break;
                }
            }
        }
    }
    __syncwarp();

    request_success = __shfl_sync(FULL_MASK, request_success, 0);
    if (request_success) {
        goto restart;
    }
#endif

#ifdef BALANCE_CNT
    if (lane_id == 0) {
        warp_time[warp_id] = clock64() - warp_time[warp_id];
        warp_time[warp_id] /= 10000;
    }
#endif
}

ull
join_bfs_dfs(
    const Graph &q,
    const Graph &g,
    const Graph_GPU &Q,
    const Graph_GPU &G,
    const candidate_graph &_cg,
    const candidate_graph_GPU &cg,
    const std::vector<int> &matching_order
) {
#ifdef IDLE_CNT
    cudaCheck(cudaMemcpyToSymbol(sample_cnt, &Zero_ull, sizeof(ull)));
    cudaCheck(cudaMemcpyToSymbol(idle_cnt, &Zero_ull, sizeof(ull)));
#endif

    int partial_matching_cnt = 0;
    int *d_partial_matchings = set_beginning_partial_matchings(q, g, _cg, matching_order, partial_matching_cnt);
    if (partial_matching_cnt == 0) {
        return 0;
    }

    int *d_matching_order = nullptr;
    cudaCheck(cudaMalloc(&d_matching_order, sizeof(int) * q.vcount()));
    cudaCheck(cudaMemcpy(d_matching_order, matching_order.data(), sizeof(int) * q.vcount(), cudaMemcpyHostToDevice));

    MemManager h_MM;
    h_MM.init(d_partial_matchings, partial_matching_cnt, 0, partial_matching_cnt);
    MemManager *d_MM = nullptr;
    cudaCheck(cudaMalloc(&d_MM, sizeof(MemManager)));
    get_partial_init(&h_MM);
    h_MM.init_prev_head();
    cudaCheck(cudaMemcpy(d_MM, &h_MM, sizeof(MemManager), cudaMemcpyHostToDevice));

    TIME_INIT();
    TIME_START();

    int l = 2;
    int expect = 1e6;
    for (; partial_matching_cnt < expect && l < q.vcount(); l++) {
        for (int offset = 0; offset < partial_matching_cnt; offset += warpNum) {
            BFS_Extend <<< maxBlocks, threadsPerBlock >>> (
                Q, G, cg, d_MM, matching_order[l], offset, 0, d_matching_order
            );
            cudaCheck(cudaGetLastError());
            cudaCheck(cudaDeviceSynchronize());
        }

        BFS_Extend <<< maxBlocks, threadsPerBlock >>> (
            Q, G, cg, d_MM, matching_order[l], l + 1, 1, d_matching_order
        );
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        cudaCheck(cudaMemcpy(&h_MM, d_MM, sizeof(MemManager), cudaMemcpyDeviceToHost));
        h_MM.swap_mem_pool();
        get_partial_init(&h_MM);
        h_MM.init_prev_head();
        partial_matching_cnt = h_MM.get_partial_cnt();
        cudaCheck(cudaMemcpy(d_MM, &h_MM, sizeof(MemManager), cudaMemcpyHostToDevice));
        printf("BFS extended to level %d. Query vertex %d has been matched. Partial matching count: %d. Number of warps: %d\n",
               l + 1, matching_order[l], partial_matching_cnt, warpNum);

        if (partial_matching_cnt == 0) {
            break;
        }
    }

    TIME_END();
    PRINT_LOCAL_TIME("BFS Finished");

    // Free some global memory for DFS
    h_MM.mempool_to_write()->deallocate();

    if (l == q.vcount() || partial_matching_cnt == 0) {
        cudaCheck(cudaFree(d_matching_order));
        return partial_matching_cnt;
    }

    check_gpu_memory();

    // BFS finished. Prepare for DFS
    printf("Conducting DFS from level %d to %d\n", l + 1, Q.vcount());

#ifdef BALANCE_CNT
    cudaCheck(cudaMemcpyToSymbol(assign_stop_time, &Zero_ull, sizeof(ull)));
#endif

    ull *d_sum = nullptr;
    cudaCheck(cudaMalloc(&d_sum, sizeof(ull) * warpNum));
    cudaCheck(cudaMemset(d_sum, 0, sizeof(ull) * warpNum));

    int *begin_offset = nullptr;
    cudaCheck(cudaMalloc(&begin_offset, sizeof(int)));
    cudaCheck(cudaMemset(begin_offset, 0, sizeof(int)));

    int shared_size = warpsPerBlock * l * sizeof(stk_elem_fixed)
                      + warpsPerBlock * (Q.vcount() - l) * sizeof(stk_elem);

    dfs_kernel <<< maxBlocks, threadsPerBlock, shared_size >>> (
        Q, G, cg, d_sum,
        d_matching_order,
        d_MM, partial_matching_cnt,
        begin_offset, l
    );
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    ull *h_sum = (ull *)malloc(sizeof(ull) * warpNum);
    cudaCheck(cudaMemcpy(h_sum, d_sum, sizeof(ull) * warpNum, cudaMemcpyDeviceToHost));

    ull ret = 0;
    for (int i = 0; i < warpNum; i++) {
        ret += h_sum[i];
    }

    h_MM.deallocate();
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

#ifdef BALANCE_CNT
    ull *h_compute_cnt = (ull *)malloc(sizeof(ull) * warpNum);
    cudaCheck(cudaMemcpyFromSymbol(h_compute_cnt, compute_cnt, sizeof(ull) * warpNum));
    printStatistics("computation count", h_compute_cnt, warpNum);
    free(h_compute_cnt);

    ull *h_warp_time = (ull *)malloc(sizeof(ull) * warpNum);
    cudaCheck(cudaMemcpyFromSymbol(h_warp_time, warp_time, sizeof(ull) * warpNum));
    printStatistics("warp time (1000 cycles)", h_warp_time, warpNum);
    free(h_warp_time);

    ull t = 0;
    cudaCheck(cudaMemcpyFromSymbol(&t, assign_stop_time, sizeof(ull)));
    printf("Assignment stop time (1000 cycles): %llu\n", t);
#endif

    return ret;
}