#include "join_dfs.h"
#include "join_bfs.h"
#include "params.h"
#include "dfs_stk.h"


__global__ void
bfs_dfs_kernel(
    const Graph_GPU Q,
    const Graph_GPU G,
    const candidate_graph_GPU cg,
    int begin_length,
    int *d_begin_offset,            // partial matching id
    int *sum,                       // sum of valid matchings
    int *d_matching_order,          // matching order
    int *d_rank,                    // rank of each vertex
    MemManager *d_MM,               // previous partial results
    dfs_stk *d_stack_storage,
    int partial_matching_cnt
) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    int bid = blockIdx.x;

    dfs_stk *this_stk = d_stack_storage + warp_id;
    int this_u = d_matching_order[begin_length];  // Current vertex to match

    while (true) {
        int this_partial_id;
        if (lane_id == 0) {
            this_partial_id = atomicAdd(d_begin_offset, 1);
        }
        this_partial_id = __shfl_sync(FULL_MASK, this_partial_id, 0);

        if (this_partial_id >= partial_matching_cnt) {
            break;
        }

        // Initialize stack with one partial matching
        if (lane_id == 0) {
            int *partial_matching = d_MM->get_partial(this_partial_id);
            // copy first `begin_length' vertices
            for (int i = 0; i < begin_length; i++) {
                this_stk->stk[i].selected_vertex = partial_matching[i];
            }
            this_stk->stk[begin_length].cand_len[0] = 0;
            this_stk->stk[begin_length].current_idx = 0;
            this_stk->stk[begin_length].uiter = 0;
        }
        __syncwarp();

        int stk_len = begin_length + 1;

        while (stk_len > begin_length) {
            int u = d_matching_order[stk_len - 1];
            // Compute candidate set if not calculated
            if (this->stk[stk_len - 1].uiter == 0 && this_stk->stk[stk_len - 1].cand_len[0] == 0) {
                // Extend subgraphs for all unrolled iterations
                int min_len = INT_MAX;
                int *min_set = nullptr;
                unsigned min_set_upper = 0, min_set_lower = 0;

                if (lane_id == 0) {
                    // Find smallest candidate set
                    for (int i = 0; i <= stk_len - 2; i++) {
                        int uu = d_matching_order[i];
                        if (Q.d_bknbrs_mask_[u] & (1 << uu)) {
                            int len = 0;
                            int *set = cg.d_get_candidates(uu, u, this_stk->stk[i].selected_vertex, len);
                            if (len < min_len) {
                                min_len = len;
                                min_set = set;
                                min_set_upper = (unsigned)((uint64_t)set >> 32);
                                min_set_lower = (unsigned)(uint64_t)set;
                            }
                        }
                    }
                }
                __syncwarp();

                min_len = __shfl_sync(FULL_MASK, min_len, 0);
                min_set_upper = __shfl_sync(FULL_MASK, min_set_upper, 0);
                min_set_lower = __shfl_sync(FULL_MASK, min_set_lower, 0);
                min_set = (int *)((uint64_t)min_set_upper << 32 | min_set_lower);

                int valid_count = 0;
                int rounds = ceil_div(min_len, warpSize);

                for (int rd = 0; rd < rounds; rd++) {
                    int idx = rd * warpSize + lane_id;
                    int candidate = (idx < min_len) ? min_set[idx] : -1;
                    bool valid = (candidate != -1);

                    // Check against all backward neighbors
                    for (int i = 0; valid && i <= stk_len - 2; i++) {
                        int uu = d_matching_order[i];
                        if (this_stk->stk[i].selected_vertex == candidate) {
                            valid = false;
                            break;
                        }
                        if (Q.d_bknbrs_mask_[u] & (1 << uu)) {
                            int len = 0;
                            int *set = cg.d_get_candidates(uu, u, this_stk->stk[i].selected_vertex, len);
                            if (set != min_set) {
                                valid = binary_search(set, len, candidate);
                            }
                        }
                    }

                    // Compact valid candidates
                    unsigned flag_mask = __ballot_sync(FULL_MASK, valid);
                    unsigned mask_low = flag_mask & ((1 << (lane_id + 1)) - 1);
                    int pos = __popc(mask_low) - 1;
                    if (valid) {
                        this_stk->stk[stk_len - 1].cand_set[valid_count + pos] = candidate;
                    }
                    valid_count += __popc(flag_mask);
                }

                if (valid_count == 0) {
                    stk_len--;
                    continue;
                }

                if (stk_len == Q.vcount()) {
                    if (lane_id == 0) {
                        atomicAdd(sum + bid, valid_count);
                    }
                    stk_len--;
                }
                else {
                    this_stk->stk[stk_len - 1].cand_len = valid_count;
                    this_stk->stk[stk_len - 1].current_idx = 0;
                    this_stk->stk[stk_len - 1].selected_vertex = this_stk->stk[stk_len - 1].cand_set[0];    
                    this_stk->stk[stk_len].cand_len = 0;
                    stk_len++;
                }
            }
            else {  // this_stk->stk[stk_len - 1].cand_len != 0
                if (this_stk->stk[stk_len - 1].current_idx == this_stk->stk[stk_len - 1].cand_len - 1) {
                    stk_len--;
                }
                else {
                    if (lane_id == 0) {
                        this_stk->stk[stk_len - 1].current_idx++;
                        this_stk->stk[stk_len - 1].selected_vertex = this_stk->stk[stk_len - 1].cand_set[this_stk->stk[stk_len - 1].current_idx];
                        this_stk->stk[stk_len].cand_len = 0;
                    }
                    __syncwarp();
                    stk_len++;
                }
            }
        }
    }
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

    int partial_matching_cnt = 0;
    int *d_partial_matchings = set_beginning_partial_matchings(q, g, _cg,
                               matching_order, partial_matching_cnt);
    assert(partial_matching_cnt > 0);

    int *d_rank = nullptr;
    std::vector<int> h_rank(matching_order.size());
    for (int i = 0; i < matching_order.size(); i++) {
        h_rank[matching_order[i]] = i;
    }
    cudaCheck(cudaMalloc(&d_rank, sizeof(int) * matching_order.size()));
    cudaCheck(cudaMemcpy(d_rank, h_rank.data(), sizeof(int) * matching_order.size(),
                         cudaMemcpyHostToDevice));

    MemManager h_MM;
    h_MM.init(d_partial_matchings, partial_matching_cnt, 0, partial_matching_cnt);
    MemManager *d_MM = nullptr;
    cudaCheck(cudaMalloc(&d_MM, sizeof(MemManager)));
    get_partial_init(&h_MM);
    h_MM.init_prev_head();
    cudaCheck(cudaMemcpy(d_MM, &h_MM, sizeof(MemManager), cudaMemcpyHostToDevice));

    // extend until there are more partial matchings than warps
    int l = 2;
    for (; partial_matching_cnt < 1e6 && l < q.vcount(); l++) {
        for (int offset = 0; offset < partial_matching_cnt; offset += warpNum) {
            BFS_Extend <<< maxBlocks, threadsPerBlock>>>(
                Q, G, cg, d_MM, matching_order[l], d_rank, offset, 0
            );
            cudaCheck(cudaGetLastError());
            cudaCheck(cudaDeviceSynchronize());
        }

        BFS_Extend <<< maxBlocks, threadsPerBlock>>>(
            Q, G, cg, d_MM, matching_order[l], d_rank, l + 1, 1
        );
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        cudaCheck(cudaMemcpy(&h_MM, d_MM, sizeof(MemManager), cudaMemcpyDeviceToHost));
        h_MM.swap_mem_pool();
        get_partial_init(&h_MM);
        h_MM.init_prev_head();
        partial_matching_cnt = h_MM.get_partial_cnt();
        cudaCheck(cudaMemcpy(d_MM, &h_MM, sizeof(MemManager), cudaMemcpyHostToDevice));
        printf("BFS extended to level %d. Query vertex %d is matched. Partial matching cnt: %d. warpNum: %d\n",
               l + 1, matching_order[l], partial_matching_cnt, warpNum);
    }

    if (l == q.vcount()) {
        cudaCheck(cudaFree(d_rank));
        return partial_matching_cnt;
    }

    // BFS finished. Prepare DFS
    printf("Conducting DFS...\n");

    // copy matching order to device
    int *d_matching_order;
    cudaCheck(cudaMalloc(&d_matching_order, sizeof(int) * q.vcount()));
    cudaCheck(cudaMemcpy(d_matching_order, matching_order.data(),
                         sizeof(int) * q.vcount(), cudaMemcpyHostToDevice));

    int *d_sum;
    cudaCheck(cudaMalloc(&d_sum, sizeof(int) * maxBlocks));
    cudaCheck(cudaMemset(d_sum, 0, sizeof(int) * maxBlocks));

    dfs_stk *d_stack_storage = nullptr;
    printf("Attempt to allocate %.2lf MB\n", sizeof(dfs_stk) * warpNum / 1024. / 1024.);
    cudaCheck(cudaMalloc(&d_stack_storage, sizeof(dfs_stk) * warpNum));
    cudaCheck(cudaMemset(d_stack_storage, 0, sizeof(dfs_stk) * warpNum));

    int *d_begin_offset = nullptr;
    cudaCheck(cudaMalloc(&d_begin_offset, sizeof(int)));
    cudaCheck(cudaMemset(d_begin_offset, 0, sizeof(int)));

    bfs_dfs_kernel <<< maxBlocks, threadsPerBlock>>>(
        Q, G, cg, l,
        d_begin_offset, d_sum,
        d_matching_order, d_rank, d_MM,
        d_stack_storage, partial_matching_cnt
    );
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    int *h_sum = (int *)malloc(sizeof(int) * maxBlocks);
    cudaCheck(cudaMemcpy(h_sum, d_sum, sizeof(int) * maxBlocks,
                         cudaMemcpyDeviceToHost));

    ull ret = 0;
    for (int i = 0; i < maxBlocks; i++) {
        ret += h_sum[i];
    }

    free(h_sum);
    cudaCheck(cudaFree(d_sum));
    cudaCheck(cudaFree(d_matching_order));
    cudaCheck(cudaFree(d_rank));

    return ret;
}