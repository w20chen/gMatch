#include "join.h"
#include "candidate.h"
#include "params.h"


__host__ int *
set_beginning_partial_matchings(
    const Graph &Q,
    const Graph &G,
    const candidate_graph &cg,
    const std::vector<int> &matching_order,
    int &cnt
) {
    int u0 = matching_order[0];
    int u1 = matching_order[1];

    assert(Q.is_adjacent(u0, u1));
    std::vector<std::pair<int, int>> h_beginning_partial_matching;
    for (int v0 : cg.cand[u0]) {
        for (int v1 : cg.cand[u1]) {
            if (G.is_adjacent(v0, v1)) {
                h_beginning_partial_matching.emplace_back(v0, v1);
            }
        }
    }

    int beginning_partial_matching_cnt = h_beginning_partial_matching.size();

    int *d_dst = nullptr;
    cudaCheck(cudaMalloc(&d_dst, sizeof(std::pair<int, int>) * beginning_partial_matching_cnt));
    cudaCheck(cudaMemcpy(d_dst, h_beginning_partial_matching.data(),
                         sizeof(std::pair<int, int>) * beginning_partial_matching_cnt,
                         cudaMemcpyHostToDevice));
    printf("Beginning partial results have been moved to the GPU.\n");

    cnt = beginning_partial_matching_cnt;

    return d_dst;
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
    int *d_matching_order
) {

    /**
     * After the function ends, the values of prev_head and blk_write_cnt are retained for the next call
     * and cached in shared memory to reduce global memory access.
     */

    __shared__ int shared_blk_write_cnt[warpsPerBlock];
    __shared__ int *shared_prev_head[warpsPerBlock];

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    int warp_id_within_blk = warp_id % warpsPerBlock;

    shared_prev_head[warp_id_within_blk] = (int *)d_MM->prev_head[warp_id];
    shared_blk_write_cnt[warp_id_within_blk] = d_MM->blk_write_cnt[warp_id];

    // If last_flag is 1, it indicates this is the last call to BFS_Extend
    // Record all partial matchings not yet recorded in d_MM
    if (last_flag == 1) {
        if (lane_id == 0 && shared_blk_write_cnt[warp_id_within_blk] > 0) {
            partial_props p;
            p.start_addr = shared_prev_head[warp_id_within_blk];
            // Use partial_offset to temporarily denote the length of partial matching
            p.partial_len = partial_offset;
            p.partial_cnt = shared_blk_write_cnt[warp_id_within_blk];
            d_MM->add_new_props(p);
        }
        __syncwarp();

        d_MM->prev_head[warp_id] = nullptr;
        d_MM->blk_write_cnt[warp_id] = 0;
        return;
    }

    // Assign partial matching to each warp
    int u = cur_query_vertex;
    int partial_matching_len = 0;
    int *this_partial_matching = d_MM->get_partial(warp_id + partial_offset, &partial_matching_len);

    if (this_partial_matching == nullptr) {
        return;
    }

    // Each block can write a maximum number of partial matchings
    const int blk_partial_max_num = memPoolBlockIntNum / (partial_matching_len + 1);

    int first_uu_index = Q.d_bknbrs_[Q.d_bknbrs_offset_[u]];
    int first_uu = d_matching_order[first_uu_index];
    int first_vv = this_partial_matching[first_uu_index];

    int min_len = 0;
    int *min_set = cg.d_get_candidates(first_uu, u, first_vv, min_len);

    for (int ii = Q.d_bknbrs_offset_[u] + 1; ii < Q.d_bknbrs_offset_[u + 1]; ii++) {
        int uu_index = Q.d_bknbrs_[ii];
        int uu = d_matching_order[uu_index];
        int vv = this_partial_matching[uu_index];
        int this_len = 0;
        int *this_set = cg.d_get_candidates(uu, u, vv, this_len);
        if (this_len < min_len) {
            min_len = this_len;
            min_set = this_set;
        }
    }

    // Ensure there is one memory block for each warp
    if (shared_prev_head[warp_id_within_blk] == nullptr) {
        int *d_head = shared_prev_head[warp_id_within_blk];
        unsigned d_head_lower = 0;
        unsigned d_head_upper = 0;
        if (lane_id == 0) {
            d_head = d_MM->mempool_to_write()->alloc();
            d_head_lower = (unsigned)d_head;
            d_head_upper = (unsigned)((unsigned long long)d_head >> 32);
        }
        d_head_lower = __shfl_sync(FULL_MASK, d_head_lower, 0);
        d_head_upper = __shfl_sync(FULL_MASK, d_head_upper, 0);
        d_head = (int *)(((unsigned long long)d_head_upper << 32) |
                         (unsigned long long)d_head_lower);

        shared_prev_head[warp_id_within_blk] = d_head;
        shared_blk_write_cnt[warp_id_within_blk] = 0;
    }

    __syncwarp();

    // Compute extendable candidate set
    // Each thread in the warp is responsible for one candidate, and handles the next candidate with a step size of 32
    for (int t = lane_id; t < min_len; t += warpSize) {
        // For each vertex v in min_set
        int v = min_set[t];
        bool flag = true;
        // If v has not been mapped before
        for (int j = 0; j < partial_matching_len; j++) {
            if (this_partial_matching[j] == v) {
                flag = false;
                break;
            }
        }
        __syncwarp();

        if (flag) {
            // For each backward neighbor uu of u
            for (int ii = Q.d_bknbrs_offset_[u]; ii < Q.d_bknbrs_offset_[u + 1]; ii++) {
                int uu_index = Q.d_bknbrs_[ii];
                int uu = d_matching_order[uu_index];
                int vv = this_partial_matching[uu_index];
                int this_len = 0;
                int *this_set = cg.d_get_candidates(uu, u, vv, this_len);
                if (this_set == min_set) {
                    continue;
                }
                if (!binary_search(this_set, this_len, v)) {
                    flag = false;
                    break;
                }
            }
        }
        __syncwarp();

        // Use ballot_sync to collect flag values from all threads and generate a mask
        unsigned int flag_mask = __ballot_sync(FULL_MASK, flag);
        // Use popc to count the number of bits set to 1 in the mask, i.e., the total number of threads with flag true
        unsigned int flag_cnt = __popc(flag_mask);
        __syncwarp();

        // flag_idx indicates the position of each thread in its warp where flag == true
        // flag_idx starts counting from 0
        // For threads with flag == false, flag_idx is meaningless
        int flag_idx = __popc(flag_mask & ((1 << lane_id) - 1));
        __syncwarp();

        // For the current process, it will write partial matching to a block that already has thread_old_cnt partial matchings
        int thread_old_cnt = shared_blk_write_cnt[warp_id_within_blk] + flag_idx;
        // Indicates whether the current process needs to write the partial matching to a new block instead of the old one
        bool thread_need_new_blk = false;

        // rest_cnt_in_blk indicates the number of partial matchings that can still be written to the old block
        int rest_cnt_in_blk = blk_partial_max_num - shared_blk_write_cnt[warp_id_within_blk];
        __syncwarp();

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
                printf("The program should not reach here.\n");
                printf("Debug Info - Thread %d in warp %d:\n", tid, warp_id);
                printf("Old count: %d\n", thread_old_cnt);
                printf("Rest count in block: %d\n", rest_cnt_in_blk);
                printf("Limit: %d\n", blk_partial_max_num);
                printf("Partial matching length: %d\n", partial_matching_len);
                printf("Flag index: %d\n", flag_idx);
                printf("Flag count: %d\n", flag_cnt);
                printf("Need new block: %d\n", thread_need_new_blk);
                printf("Block write count: %d\n", shared_blk_write_cnt[warp_id_within_blk]);
                assert(0);
            }
        }
        __syncwarp();

        if (lane_id == 0 && shared_blk_write_cnt[warp_id_within_blk] != -1) {
            // If lane_id is 0 and shared_blk_write_cnt[warp_id_within_blk] != -1, it means no new block is needed
            shared_blk_write_cnt[warp_id_within_blk] += flag_cnt;
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
            partial_props p;
            p.start_addr = shared_prev_head[warp_id_within_blk];
            p.partial_len = partial_matching_len + 1;
            p.partial_cnt = blk_partial_max_num;
            d_MM->add_new_props(p);

            d_new_head = d_MM->mempool_to_write()->alloc(1);
            d_new_head_lower = (unsigned)d_new_head;
            d_new_head_upper = (unsigned)((unsigned long long)d_new_head >> 32);
        }
        __syncwarp();

        d_new_head_lower = __shfl_sync(FULL_MASK, d_new_head_lower, 0);
        d_new_head_upper = __shfl_sync(FULL_MASK, d_new_head_upper, 0);
        d_new_head = (int *)(((unsigned long long)d_new_head_upper << 32) |
                             (unsigned long long)d_new_head_lower);
        // d_new_head is null if no new block is allocated

        // Some threads write to the new block, some write to the old block
        int *blk_to_write = shared_prev_head[warp_id_within_blk];
        if (thread_need_new_blk == true) {
            blk_to_write = d_new_head;
        }
        __syncwarp();

        if (flag) {
            // Write the newly found partial matching
            int idx = thread_old_cnt * (partial_matching_len + 1) + partial_matching_len;
            if (idx >= memPoolBlockIntNum || idx < 0) {
                printf("idx: %d, memPoolBlockIntNum: %lld\n", idx, memPoolBlockIntNum);
                assert(0);
            }
            blk_to_write[idx] = v;
            for (int i = 0; i < partial_matching_len; i++) {
                idx = thread_old_cnt * (partial_matching_len + 1) + i;
                blk_to_write[idx] = this_partial_matching[i];
            }
        }
        __syncwarp();

        if (shared_blk_write_cnt[warp_id_within_blk] == -1) {
            shared_blk_write_cnt[warp_id_within_blk] = flag_cnt - rest_cnt_in_blk;
        }
        __syncwarp();

        if (thread_need_new_blk && d_new_head != nullptr) {
            shared_prev_head[warp_id_within_blk] = d_new_head;
        }
        __syncwarp();
    }
    __syncwarp();

    d_MM->prev_head[warp_id] = shared_prev_head[warp_id_within_blk];
    d_MM->blk_write_cnt[warp_id] = shared_blk_write_cnt[warp_id_within_blk];
}


int
join_bfs_inner(
    const Graph &q,
    const Graph &g,
    const Graph_GPU &Q,
    const Graph_GPU &G,
    const candidate_graph_GPU &cg,
    const std::vector<int> &matching_order,
    int begin_offset_,
    int begin_size_,
    int partial_matching_cnt_,
    int *d_partial_matchings_
) {

    int partial_matching_cnt = partial_matching_cnt_;
    int *d_partial_matchings = d_partial_matchings_;

    MemManager h_MM;
    // Move a table of initial partial matchings (candidate edges) to memory pool.
    h_MM.init(d_partial_matchings, partial_matching_cnt, begin_offset_, begin_size_);

    MemManager *d_MM = nullptr;
    cudaCheck(cudaMalloc(&d_MM, sizeof(MemManager)));

    int *d_matching_order = nullptr;
    cudaCheck(cudaMalloc(&d_matching_order, sizeof(int) * matching_order.size()));
    cudaCheck(cudaMemcpy(d_matching_order, matching_order.data(), sizeof(int) * matching_order.size(), cudaMemcpyHostToDevice));

    for (int partial_matching_len = 2; partial_matching_len < matching_order.size(); partial_matching_len++) {

        int partial_matching_cnt = h_MM.get_partial_cnt();
        if (partial_matching_cnt == 0) {
            break;
        }

        // Initialize partial matching prefix sum (cnt_prefix_sum)
        // The result will be used for finding partial matching for each warp (when calling get_partial in BFS_Extend)
        get_partial_init(&h_MM);

        // Initialize prev_head[warp_id] == nullptr and blk_write_cnt[warp_id] == 0
        h_MM.init_prev_head();

        cudaCheck(cudaMemcpy(d_MM, &h_MM, sizeof(MemManager), cudaMemcpyHostToDevice));

        for (int partial_offset = 0; partial_offset < partial_matching_cnt;
                partial_offset += maxBlocks * warpsPerBlock) {
            BFS_Extend <<< maxBlocks, threadsPerBlock>>>(
                Q, G, cg, d_MM, matching_order[partial_matching_len], partial_offset, 0, d_matching_order
            );
            cudaCheck(cudaGetLastError());
            cudaCheck(cudaDeviceSynchronize());
        }

        // Finally call BFS_Extend once more with last_flag set to 1
        // Record all partial matchings not yet recorded in d_MM
        // Temporarily set partial_offset to partial_matching_len + 1
        BFS_Extend <<< maxBlocks, threadsPerBlock>>>(
            Q, G, cg, d_MM, matching_order[partial_matching_len],
            partial_matching_len + 1, 1, d_matching_order
        );

        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        cudaCheck(cudaMemcpy(&h_MM, d_MM, sizeof(MemManager), cudaMemcpyDeviceToHost));
        h_MM.swap_mem_pool();
        cudaCheck(cudaDeviceSynchronize());
    }

    int ret = h_MM.get_partial_cnt();

    h_MM.deallocate();
    cudaCheck(cudaFree(d_MM));
    cudaCheck(cudaFree(d_matching_order));

    return ret;
}


ull
join_bfs(
    const Graph &q,
    const Graph &g,
    const Graph_GPU &Q,
    const Graph_GPU &G,
    const candidate_graph &_cg,
    const candidate_graph_GPU &cg,
    const std::vector<int> &matching_order
) {

    // Generate a table of candidate edges.
    int partial_matching_cnt = 0;
    int *d_partial_matchings = set_beginning_partial_matchings(q, g, _cg,
                               matching_order, partial_matching_cnt);
    if (partial_matching_cnt <= 0) {
        return 0;
    }

    ull sum = 0;

    // begin_size is the number of partial matchings to process in each iteration
    int begin_size = BFS_BEGIN_SIZE;

    // begin_offset is the offset of partial matchings to process in each iteration
    for (int begin_offset = 0; begin_offset < partial_matching_cnt; begin_offset += begin_size) {
        printf("BFS begin_offset: (%d/%d), sum: %llu\n", begin_offset, partial_matching_cnt, sum);
        sum += join_bfs_inner(q, g, Q, G, cg, matching_order,
                              begin_offset, begin_size, partial_matching_cnt, d_partial_matchings);
    }

    cudaCheck(cudaFree(d_partial_matchings));
    return sum;
}
