#include "join_bfs.h"
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
    printf("Beginning partial matching count: %d\n",
           beginning_partial_matching_cnt);
    assert(beginning_partial_matching_cnt > 0);

    int ll = std::min(10, beginning_partial_matching_cnt);
    printf("First %d partial matchings: ", ll);
    for (int i = 0; i < ll; i++) {
        printf("(%d, %d) ", h_beginning_partial_matching[i].first,
               h_beginning_partial_matching[i].second);
    }
    printf("\n");

    int *d_dst;
    cudaCheck(cudaMalloc(&d_dst,
                         sizeof(std::pair<int, int>) * beginning_partial_matching_cnt));
    cudaCheck(cudaMemcpy(d_dst, h_beginning_partial_matching.data(),
                         sizeof(std::pair<int, int>) * beginning_partial_matching_cnt,
                         cudaMemcpyHostToDevice));
    printf("Beginning partial results moved to GPU.\n");

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
    int *d_rank,
    int partial_offset,
    int last_flag
) {

    /**
     * After the function ends, the values of prev_head and blk_write_cnt are retained for the next call
     * and cached in shared memory to reduce global memory access.
     */

    assert(sizeof(int *) == sizeof(int) * 2);
    assert(warpsPerBlock == blockDim.x / warpSize);

    __shared__ int shared_blk_write_cnt[warpsPerBlock];
    __shared__ int *shared_prev_head[warpsPerBlock];

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    int warp_id_within_blk = warp_id % warpsPerBlock;

    shared_prev_head[warp_id_within_blk] = (int *)d_MM->prev_head[warp_id];
    shared_blk_write_cnt[warp_id_within_blk] = d_MM->blk_write_cnt[warp_id];

    assert(shared_blk_write_cnt[warp_id_within_blk] >= 0);

    // If last_flag is 1, it indicates this is the last call to BFS_Extend
    // Record all partial matchings not yet recorded in d_MM
    if (last_flag == 1) {
        if (lane_id == 0 && shared_blk_write_cnt[warp_id_within_blk] > 0) {
            assert(partial_offset >= 2);
            assert(shared_prev_head[warp_id_within_blk] != nullptr);
            partial_props p;
            p.start_addr = shared_prev_head[warp_id_within_blk];
            // Use partial_offset to denote the length of partial matching temporarily
            p.partial_len = partial_offset;
            p.partial_cnt = shared_blk_write_cnt[warp_id_within_blk];
            d_MM->add_new_props(p);
        }
        __syncwarp();

        d_MM->prev_head[warp_id] = nullptr;
        d_MM->blk_write_cnt[warp_id] = 0;
        return;
    }
    assert(last_flag == 0);

    // Assign partial matching to each warp
    int u = cur_query_vertex;
    int partial_matching_len = 0;
    int *this_partial_matching = d_MM->get_partial(warp_id + partial_offset,
                                 &partial_matching_len);

    if (this_partial_matching == nullptr) {
        return;
    }

    assert(partial_matching_len >= 2);

    // Each block can write a maximum number of partial matchings
    const int blk_partial_max_num = memPoolBlockIntNum / (partial_matching_len + 1);
    assert(blk_partial_max_num > warpSize);

    assert(Q.d_bknbrs_offset_ != nullptr);
    assert(Q.d_bknbrs_ != nullptr);
    assert(Q.d_bknbrs_offset_[u] <= Q.d_bknbrs_offset_[u + 1]);
    int first_uu = Q.d_bknbrs_[Q.d_bknbrs_offset_[u]];
    int first_vv = this_partial_matching[d_rank[first_uu]];
    assert(first_vv >= 0);

    int min_len = 0;
    int *min_set = cg.d_get_candidates(first_uu, u, first_vv, min_len);
    assert(min_set != nullptr);

    for (int ii = Q.d_bknbrs_offset_[u] + 1; ii < Q.d_bknbrs_offset_[u + 1]; ii++) {
        int uu = Q.d_bknbrs_[ii];
        int vv = this_partial_matching[d_rank[uu]];
        int this_len = 0;
        int *this_set = cg.d_get_candidates(uu, u, vv, this_len);
        if (this_len < min_len) {
            min_len = this_len;
            min_set = this_set;
        }
    }

    // Make sure there is one memory block for each warp
    if (shared_prev_head[warp_id_within_blk] == nullptr) {
        int *d_head = shared_prev_head[warp_id_within_blk];
        unsigned d_head_lower = 0;
        unsigned d_head_upper = 0;
        if (lane_id == 0) {
            d_head = d_MM->mempool_to_write()->alloc();
            assert(d_head != nullptr);
            d_head_lower = (unsigned)d_head;
            d_head_upper = (unsigned)((unsigned long long)d_head >> 32);
        }
        d_head_lower = __shfl_sync(FULL_MASK, d_head_lower, 0);
        d_head_upper = __shfl_sync(FULL_MASK, d_head_upper, 0);
        d_head = (int *)(((unsigned long long)d_head_upper << 32) |
                         (unsigned long long)d_head_lower);
        assert(d_head != nullptr);

        shared_prev_head[warp_id_within_blk] = d_head;
        shared_blk_write_cnt[warp_id_within_blk] = 0;
    }

    __syncwarp();

    // Compute extendable candidate set
    // Each thread in the warp is responsible for one candidate, and handles the next candidate with a step size of 32
    for (int t = lane_id; t < min_len; t += warpSize) {
        // for each vertex v in min_set
        int v = min_set[t];
        bool flag = true;
        // if v has not been mapped before
        for (int j = 0; j < partial_matching_len; j++) {
            if (this_partial_matching[j] == v) {
                flag = false;
                break;
            }
        }
        __syncwarp();

        if (flag) {
            // for each backward neighbor uu of u
            for (int ii = Q.d_bknbrs_offset_[u]; ii < Q.d_bknbrs_offset_[u + 1]; ii++) {
                int uu = Q.d_bknbrs_[ii];
                int vv = this_partial_matching[d_rank[uu]];
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
        // unsigned int flag_mask = __ballot_sync(__activemask(), flag);
        // Use popc to count the number of bits set to 1 in the mask, i.e., the total number of threads with flag true
        unsigned int flag_cnt = __popc(flag_mask);
        assert(flag_cnt <= 32);
        __syncwarp();

        // flag_idx indicates the position of each thread in its warp where flag == true
        // flag_idx starts counting from 0
        // For threads with flag == false, flag_idx is meaningless
        int flag_idx = __popc(flag_mask & ((1 << lane_id) - 1));
        assert(flag_idx >= 0 && flag_idx < 32);
        __syncwarp();

        assert(shared_blk_write_cnt[warp_id_within_blk] >= 0);

        // For the current process, it will write partial matching to a block that already has thread_old_cnt partial matchings
        int thread_old_cnt = shared_blk_write_cnt[warp_id_within_blk] + flag_idx;
        // Indicates whether the current process needs to write the partial matching to a new block instead of the old one
        bool thread_need_new_blk = false;

        // rest_cnt_in_blk indicates the number of partial matchings that can still be written to the old block
        int rest_cnt_in_blk = blk_partial_max_num -
                              shared_blk_write_cnt[warp_id_within_blk];
        assert(shared_blk_write_cnt[warp_id_within_blk] >= 0);
        assert(rest_cnt_in_blk >= 0);
        __syncwarp();

        if (flag && thread_old_cnt + 1 > blk_partial_max_num) {
            // For this thread, it will write partial matching to a new block because the old block is full
            thread_need_new_blk = true;
            // Mark shared_blk_write_cnt[warp_id_within_blk] == -1 
            // to indicate that there are threads in the current warp that need to request a new block
            shared_blk_write_cnt[warp_id_within_blk] = -1;
            // In the new block, the number of blocks that already exist before this thread
            thread_old_cnt = flag_idx - rest_cnt_in_blk;
            assert(thread_old_cnt >= 0);
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
            assert(thread_old_cnt < blk_partial_max_num);
        }
        __syncwarp();

        if (lane_id == 0 && shared_blk_write_cnt[warp_id_within_blk] != -1) {
            // If lane_id is 0 and shared_blk_write_cnt[warp_id_within_blk] != -1, it means no new block is needed
            assert(thread_need_new_blk == false);
            shared_blk_write_cnt[warp_id_within_blk] += flag_cnt;
            assert(shared_blk_write_cnt[warp_id_within_blk] >= 0);
            assert(shared_blk_write_cnt[warp_id_within_blk] <= blk_partial_max_num);
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
        assert(blk_to_write);
        __syncwarp();

        if (flag) {
            // Write the newly found partial matching
            int idx = thread_old_cnt * (partial_matching_len + 1) + partial_matching_len;
            if (idx >= memPoolBlockIntNum || idx < 0) {
                printf("idx: %d, memPoolBlockIntNum: %lld\n", idx, memPoolBlockIntNum);
                assert(0);
            }
            assert(idx < memPoolBlockIntNum && idx >= 0);
            assert(thread_old_cnt >= 0 && thread_old_cnt < blk_partial_max_num);
            blk_to_write[idx] = v;
            for (int i = 0; i < partial_matching_len; i++) {
                idx = thread_old_cnt * (partial_matching_len + 1) + i;
                blk_to_write[idx] = this_partial_matching[i];
            }
        }
        __syncwarp();

        if (shared_blk_write_cnt[warp_id_within_blk] == -1) {
            shared_blk_write_cnt[warp_id_within_blk] = flag_cnt - rest_cnt_in_blk;
            assert(shared_blk_write_cnt[warp_id_within_blk] >= 0);
            assert(shared_prev_head[warp_id_within_blk] != nullptr);
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
    assert(d_MM->blk_write_cnt[warp_id] >= 0);
    assert(d_MM->prev_head[warp_id] != nullptr);
}
