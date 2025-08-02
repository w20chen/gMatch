#include "join.h"
#include "candidate.h"
#include "params.h"
#include "dfs_stk.h"

#include <cstdio>
#include <cuda_runtime.h>


#define BEGIN_LEN 2


#ifdef BALANCE_CNT
__device__ static ull warp_time[warpNum];
#endif


// Function to implement upper_bound
static __device__ __host__ int
upper_bound(ull arr[], int N, ull X)
{
    int mid;

    // Initialise starting index and
    // ending index
    int low = 0;
    int high = N;

    // Till low is less than high
    while (low < high) {
        // Find the middle index
        mid = low + (high - low) / 2;

        // If X is greater than or equal
        // to arr[mid] then find
        // in right subarray
        if (X >= arr[mid]) {
            low = mid + 1;
        }

        // If X is less than arr[mid]
        // then find in left subarray
        else {
            high = mid;
        }
    }
  
    // if X is greater than arr[n-1]
    if (low < N && arr[low] <= X) {
        low++;
    }

    // Return the upper_bound index
    return low;
}


static __device__ void
init_dfs_stacks(
    Graph_GPU Q,
    candidate_graph_GPU cg,
    stk_elem_fixed *this_stk_elem_fixed,
    stk_elem *this_stk_elem,
    int lane_id,
    int v0,
    int v1
) {

    stk_elem *e_ptr = this_stk_elem + 2;

    if (lane_id == 0) {
        int this_partial_matching[2];
        this_partial_matching[0] = v0;
        this_partial_matching[1] = v1;

        for (int depth = 0; depth < 2; depth++) {
            this_stk_elem_fixed[depth].mapped_v[0] = this_partial_matching[depth];
        }

        int next_u = 2;
        int min_len = INT32_MAX;
        int *min_set = nullptr;

        unsigned bn_mask = Q.d_bknbrs_[next_u];

        while (bn_mask) {
            int uu = __ffs(bn_mask) - 1;
            bn_mask &= ~(1u << uu);
            int vv = this_partial_matching[uu];
            int this_len = 0;
            int *this_set = cg.get_nbr(vv, this_len);
            if (this_len < min_len) {
                min_len = this_len;
                min_set = this_set;
            }
        }

        e_ptr->cand_set[0] = min_set;
        e_ptr->cand_len[0] = min_len;

        e_ptr->unroll_size = 1;
        e_ptr->start_idx_within_set = 0;
        e_ptr->start_set_idx = 0;
    }
    __syncwarp();
}


__global__ void static
dfs_kernel_no_filtering(
    Graph_GPU Q,
    Graph_GPU G,
    candidate_graph_GPU cg,
    ull *sum,
    int *d_partial_order,
    ull *d_begin_offset
) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = tid / warpSize;
    int warp_id_within_blk = warp_id % warpsPerBlock;
    int lane_id = tid % warpSize;

    __shared__ uint32_t s_partial_order[32];
    extern __shared__ char shared_mem[];

    stk_elem_fixed *s_stk_elem_fixed = (stk_elem_fixed *)shared_mem;
    stk_elem *s_stk_elem = (stk_elem *)(s_stk_elem_fixed + warpsPerBlock * BEGIN_LEN);

    if (threadIdx.x < Q.vcount()) {
        s_partial_order[threadIdx.x] = d_partial_order[threadIdx.x];
    }

    stk_elem_fixed *this_stk_elem_fixed = s_stk_elem_fixed + warp_id_within_blk * BEGIN_LEN;
    stk_elem *this_stk_elem = s_stk_elem + warp_id_within_blk * (Q.vcount() - BEGIN_LEN) - BEGIN_LEN;

    __syncthreads();

    const ull max_offset = (ull)G.ecount() * 2;

    int v0 = 0;
    int v1 = 0;

    while (true) {
        if (lane_id == 0) {
            ull edge_index = atomicAdd(d_begin_offset, 1);
            if (edge_index >= max_offset) {
                v0 = -1;
            }
            else {
                v0 = upper_bound(cg.nbr_offset + v0, (G.vcount() + 1 - v0), edge_index) - 1 + v0;
                v1 = cg.nbr_array[edge_index];
            }
        }
        __syncwarp();

        v0 = __shfl_sync(FULL_MASK, v0, 0);
        v1 = __shfl_sync(FULL_MASK, v1, 0);

        if (v0 == -1) {
            break;
        }

        if (s_partial_order[1] & (1 << 0)) {
            if (v1 <= v0) {
                continue;
            }
        }

        init_dfs_stacks(Q, cg, this_stk_elem_fixed, this_stk_elem, lane_id, v0, v1);

        int this_stk_len = BEGIN_LEN + 1;

        while (this_stk_len > BEGIN_LEN) {
            int this_u = this_stk_len - 1;
            stk_elem *e_ptr = &this_stk_elem[this_u];

            // Process the candidate set of the current stack element
            int lane_parent_idx = -1;
            int lane_idx_within_set = -1;
            int lane_v = -1;

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

            if (lane_id == warpSize - 1) {
                if (lane_v == -1) {
                    e_ptr->start_set_idx = -1;
                }
                else if (lane_idx_within_set == e_ptr->cand_len[lane_parent_idx] - 1) {
                    e_ptr->start_set_idx = lane_parent_idx + 1;
                    e_ptr->start_idx_within_set = 0;
                    if (e_ptr->start_set_idx == e_ptr->unroll_size) {
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
                int j = this_stk_len - 2;
                unsigned bn_mask = Q.d_bknbrs_[this_u];
                // For each backward neighbor uu of this_u (enumerate uu in descending order)

                while (bn_mask) {
                    int uu = 31 - __clz(bn_mask);
                    bn_mask &= ~(1u << uu);
                    int vv = -1;

                    while (j > uu) {
                        if (j < BEGIN_LEN) {
                            vv = this_stk_elem_fixed[j].mapped_v[cur_parent];
                        }
                        else {
                            vv = this_stk_elem[j].mapped_v[cur_parent];
                            cur_parent = this_stk_elem[j].parent_idx[cur_parent];
                        }
                        j--;

                        if (vv == lane_v) {
                            flag = false;
                            break;
                        }
                        else if (s_partial_order[this_u] & (1 << (j + 1))) {
                            if (lane_v < vv) {
                                flag = false;
                                break;
                            }
                        }
                    }

                    if (flag == false) {
                        break;
                    }

                    if (j < BEGIN_LEN) {
                        vv = this_stk_elem_fixed[j].mapped_v[cur_parent];
                    }
                    else {
                        vv = this_stk_elem[j].mapped_v[cur_parent];
                        cur_parent = this_stk_elem[j].parent_idx[cur_parent];
                    }
                    j--;

                    if (vv == lane_v) {
                        flag = false;
                        break;
                    }
                    else if (s_partial_order[this_u] & (1 << uu)) {
                        if (lane_v < vv) {
                            flag = false;
                            break;
                        }
                    }

                    int this_len = 0;
                    int *this_set = cg.get_nbr(vv, this_len);

                    if (this_set == e_ptr->cand_set[lane_parent_idx]) {
                        continue;
                    }

                    if (false == binary_search<int>(this_set, this_len, lane_v)) {
                        flag = false;
                        break;
                    }
                }

                if (flag) {
                    while (j >= 0) {
                        int vv = -1;
                        if (j < BEGIN_LEN) {
                            vv = this_stk_elem_fixed[j].mapped_v[cur_parent];
                        }
                        else {
                            vv = this_stk_elem[j].mapped_v[cur_parent];
                            cur_parent = this_stk_elem[j].parent_idx[cur_parent];
                        }
                        if (vv == lane_v) {
                            flag = false;
                            break;
                        }
                        else if (s_partial_order[this_u] & (1 << j)) {
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
            else {  // flag_mask != 0
                if (this_stk_len == Q.vcount()) {
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
                else {  // *this_stk_len != Q.vcount()
                    if (lane_id == 0) {
                        this_stk_elem[this_stk_len].start_set_idx = 0;
                        this_stk_elem[this_stk_len].start_idx_within_set = 0;
                        this_stk_len++;
                    }
                    this_stk_len = __shfl_sync(FULL_MASK, this_stk_len, 0);

                    // "e_ptr" is the pointer of previous stack top
                    // "new_e_ptr" is the pointer of the level we are about to search
                    int next_u = this_stk_len - 1;
                    stk_elem *new_e_ptr = &this_stk_elem[next_u];

                    if (lane_id == 0) {
                        new_e_ptr->unroll_size = __popc(flag_mask);
                    }
                    __syncwarp();

                    // Select candidates at the current level
                    // Organize their minimum candidate sets
                    // Proceed to the next level

                    int chosen_index = -1;
                    if (flag_mask & (1 << lane_id)) {
                        unsigned mask_low = flag_mask & ((1 << (lane_id + 1)) - 1);
                        chosen_index = __popc(mask_low) - 1;
                    }

                    if (chosen_index >= 0) {
                        e_ptr->parent_idx[chosen_index] = lane_parent_idx;
                        e_ptr->mapped_v[chosen_index] = lane_v;

                        int j = this_stk_len - 2;
                        int cur_parent = chosen_index;
                        int min_len = INT32_MAX;
                        int *min_set = nullptr;
                        unsigned bn_mask = Q.d_bknbrs_[next_u];

                        while (bn_mask) {
                            int uu = 31 - __clz(bn_mask);
                            bn_mask &= ~(1u << uu);
                            int vv = -1;

                            while (j > uu) {
                                if (j >= BEGIN_LEN) {
                                    cur_parent = this_stk_elem[j].parent_idx[cur_parent];
                                    j--;
                                }
                                else {
                                    j = uu;
                                    break;
                                }
                            }

                            // Now j == uu
                            if (j < BEGIN_LEN) {
                                vv = this_stk_elem_fixed[j].mapped_v[cur_parent];
                            }
                            else {
                                vv = this_stk_elem[j].mapped_v[cur_parent];
                            }

                            int this_len = 0;
                            int *this_set = cg.get_nbr(vv, this_len);

                            if (min_len > this_len) {
                                min_len = this_len;
                                min_set = this_set;
                            }
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

__global__ void static
dfs_kernel_no_filtering_clique(
    Graph_GPU Q,
    Graph_GPU G,
    candidate_graph_GPU cg,
    ull *sum,
    ull *d_begin_offset
) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = tid / warpSize;
    int warp_id_within_blk = warp_id % warpsPerBlock;
    int lane_id = tid % warpSize;

    extern __shared__ char shared_mem[];

    stk_elem_fixed *s_stk_elem_fixed = (stk_elem_fixed *)shared_mem;
    stk_elem *s_stk_elem = (stk_elem *)(s_stk_elem_fixed + warpsPerBlock * BEGIN_LEN);

    stk_elem_fixed *this_stk_elem_fixed = s_stk_elem_fixed + warp_id_within_blk * BEGIN_LEN;
    stk_elem *this_stk_elem = s_stk_elem + warp_id_within_blk * (Q.vcount() - BEGIN_LEN) - BEGIN_LEN;

    __syncthreads();

    const ull max_offset = (ull)G.ecount() * 2;

    int v0 = 0;
    int v1 = 0;

#ifdef BALANCE_CNT
    if (lane_id == 0) {
        warp_time[warp_id] = clock64();
    }
#endif

    while (true) {
        if (lane_id == 0) {
            ull edge_index = atomicAdd(d_begin_offset, 1);
            if (edge_index >= max_offset) {
                v0 = -1;
            }
            else {
                v0 = upper_bound(cg.nbr_offset + v0, (G.vcount() + 1 - v0), edge_index) - 1 + v0;
                v1 = cg.nbr_array[edge_index];
            }
        }
        __syncwarp();

        v0 = __shfl_sync(FULL_MASK, v0, 0);
        v1 = __shfl_sync(FULL_MASK, v1, 0);

        if (v0 == -1) {
            break;
        }

        if (v1 >= v0) {
            continue;
        }

        init_dfs_stacks(Q, cg, this_stk_elem_fixed, this_stk_elem, lane_id, v0, v1);

        int this_stk_len = BEGIN_LEN + 1;

        while (this_stk_len > BEGIN_LEN) {
            int this_u = this_stk_len - 1;
            stk_elem *e_ptr = &this_stk_elem[this_u];

            // Process the candidate set of the current stack element
            int lane_parent_idx = -1;
            int lane_idx_within_set = -1;
            int lane_v = -1;

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

            if (lane_id == warpSize - 1) {
                if (lane_v == -1) {
                    e_ptr->start_set_idx = -1;
                }
                else if (lane_idx_within_set == e_ptr->cand_len[lane_parent_idx] - 1) {
                    e_ptr->start_set_idx = lane_parent_idx + 1;
                    e_ptr->start_idx_within_set = 0;
                    if (e_ptr->start_set_idx == e_ptr->unroll_size) {
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

                for (int uu = this_stk_len - 2; uu >= 0; uu--) {
                    int vv;

                    if (uu < BEGIN_LEN) {
                        vv = this_stk_elem_fixed[uu].mapped_v[cur_parent];
                    }
                    else {
                        vv = this_stk_elem[uu].mapped_v[cur_parent];
                        cur_parent = this_stk_elem[uu].parent_idx[cur_parent];
                    }

                    if (lane_v >= vv) {
                        flag = false;
                        break;
                    }

                    int this_len = 0;
                    int *this_set = cg.get_nbr(vv, this_len);

                    if (this_set == e_ptr->cand_set[lane_parent_idx]) {
                        continue;
                    }

                    if (false == binary_search<int>(this_set, this_len, lane_v)) {
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
            else {  // flag_mask != 0
                if (this_stk_len == Q.vcount()) {
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
                else {  // *this_stk_len != Q.vcount()
                    if (lane_id == 0) {
                        this_stk_elem[this_stk_len].start_set_idx = 0;
                        this_stk_elem[this_stk_len].start_idx_within_set = 0;
                        this_stk_len++;
                    }
                    this_stk_len = __shfl_sync(FULL_MASK, this_stk_len, 0);

                    // "e_ptr" is the pointer of previous stack top
                    // "new_e_ptr" is the pointer of the level we are about to search
                    int next_u = this_stk_len - 1;
                    stk_elem *new_e_ptr = &this_stk_elem[next_u];

                    if (lane_id == 0) {
                        new_e_ptr->unroll_size = __popc(flag_mask);
                    }
                    __syncwarp();

                    // Select candidates at the current level
                    // Organize their minimum candidate sets
                    // Proceed to the next level

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
#ifdef BALANCE_CNT
    if (lane_id == 0) {
        warp_time[warp_id] = clock64() - warp_time[warp_id];
        warp_time[warp_id] /= 1e6;
    }
#endif
}


ull
join_no_filtering(
    Graph &q,
    Graph &g,
    Graph_GPU &Q,
    Graph_GPU &G,
    candidate_graph &_cg,
    candidate_graph_GPU &cg,
    std::vector<uint32_t> &partial_order
) {

    printf("### Using no-filtering mode\n");

    ull *d_begin_offset = nullptr;
    cudaCheck(cudaMalloc(&d_begin_offset, sizeof(ull)));
    cudaCheck(cudaMemset(d_begin_offset, 0, sizeof(ull)));

    int *d_partial_order = nullptr;
    cudaCheck(cudaMalloc(&d_partial_order, sizeof(uint32_t) * q.vcount()));
    cudaCheck(cudaMemcpy(d_partial_order, partial_order.data(), sizeof(uint32_t) * q.vcount(), cudaMemcpyHostToDevice));

    check_gpu_memory();

    ull *d_sum = nullptr;
    cudaCheck(cudaMalloc(&d_sum, sizeof(ull) * warpNum));
    cudaCheck(cudaMemset(d_sum, 0, sizeof(ull) * warpNum));

    int dynamic_shared_size = warpsPerBlock * (Q.vcount() - 2) * sizeof(stk_elem) + warpsPerBlock * 2 * sizeof(stk_elem_fixed);

    printf("Shared memory usage: %.2f KB per thread block\n", dynamic_shared_size / 1024.0);
    printf("DFS kernel theoretical occupancy %.2f%%\n", calculateOccupancy((const void *)dfs_kernel_no_filtering, threadsPerBlock, dynamic_shared_size));

    TIME_INIT();
    TIME_START();

    if (!q.is_clique()) {
        dfs_kernel_no_filtering <<< maxBlocks, threadsPerBlock, dynamic_shared_size >>> (
            Q, G, cg, d_sum,
            d_partial_order,
            d_begin_offset
        );
    }
    else {
        printf("### clique\n");
        dfs_kernel_no_filtering_clique <<< maxBlocks, threadsPerBlock, dynamic_shared_size >>> (
            Q, G, cg, d_sum,
            d_begin_offset
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

    free(h_sum);
    cudaCheck(cudaFree(d_sum));

#ifdef BALANCE_CNT
    ull *h_warp_time = (ull *)malloc(sizeof(ull) * warpNum);
    cudaCheck(cudaMemcpyFromSymbol(h_warp_time, warp_time, sizeof(ull) * warpNum));
    printStatistics("Warp time (1e6 cycles)", h_warp_time, warpNum);

    ull *h_block_time = (ull *)malloc(sizeof(ull) * maxBlocks);
    for (int i = 0; i < warpNum; i += warpsPerBlock) {
        ull sum = 0;
        for (int j = i; j < i + warpsPerBlock; j++) {
            sum += h_warp_time[j];
        }
        h_block_time[i / warpsPerBlock] = sum / warpsPerBlock;
    }
    printStatistics("Block average time (1e6 cycles)", h_block_time, maxBlocks);
    free(h_warp_time);
    free(h_block_time);
#endif

    return ret;
}