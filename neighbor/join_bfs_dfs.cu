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
#endif

static __device__ void
init_dfs_stacks(
    Graph_GPU Q,
    candidate_graph_GPU cg,

    stk_elem_fixed *this_stk_elem_fixed,
    stk_elem *this_stk_elem,

    MemManager *d_MM,
    int this_offset,
    int partial_matching_cnt,
    int lane_id,
    int begin_len
) {

    int partial_matching_id = this_offset + lane_id;
    stk_elem *e_ptr = this_stk_elem + begin_len;

    if (lane_id < UNROLL_MIN && partial_matching_id < partial_matching_cnt) {
        int *this_partial_matching = d_MM->get_partial(partial_matching_id);

        for (int depth = 0; depth < begin_len; depth++) {
            this_stk_elem_fixed[depth].mapped_v[lane_id] = this_partial_matching[depth];
            this_stk_elem_fixed[depth].mapped_v_nbr_set[lane_id] = cg.get_nbr(this_stk_elem_fixed[depth].mapped_v[lane_id], this_stk_elem_fixed[depth].mapped_v_nbr_len[lane_id]);
        }

        // Calculate the minimum length of the candidate set for the current partial matching
        int next_u = begin_len;
        int min_len = INT32_MAX;
        int *min_set = nullptr;

        unsigned bn_mask = Q.d_bknbrs_[next_u];

        while (bn_mask) {
            int uu = __ffs(bn_mask) - 1;
            bn_mask &= ~(1u << uu);
            int this_len = this_stk_elem_fixed[uu].mapped_v_nbr_len[lane_id];
            if (this_len < min_len) {
                min_len = this_len;
                min_set = this_stk_elem_fixed[uu].mapped_v_nbr_set[lane_id];
            }
        }

        this_stk_elem[begin_len].cand_set[lane_id] = min_set;
        this_stk_elem[begin_len].cand_len[lane_id] = min_len;
    }

    if (lane_id == 0) {
        if (this_offset + UNROLL_MIN > partial_matching_cnt) {
            e_ptr->cand_len[partial_matching_cnt - this_offset] = -1;
        }
        else {
            e_ptr->cand_len[UNROLL_MIN] = -1;
        }

        e_ptr->start_idx_within_set = 0;
        e_ptr->start_set_idx = 0;
    }
    __syncwarp();
}


__global__ void static
dfs_kernel(
    Graph_GPU Q,
    Graph_GPU G,
    candidate_graph_GPU cg,
    ull *sum,                           // Sum of the number of valid partial matchings
    MemManager *d_MM,                   // Where previous partial results are obtained
    int partial_matching_cnt,           // Total number of the beginning partial results
    int *begin_offset,                  // Atomic variable denoting the next unsolved partial matching
    int begin_len                       // Length of the beginning partial results
) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;    // Global thread id
    int warp_id = tid / warpSize;                       // Global warp id
    int warp_id_within_blk = warp_id % warpsPerBlock;   // Warp id within the block
    int lane_id = tid % warpSize;                       // Lane id within the warp

#ifndef UNLABELED
    __shared__ int s_qlabel[32];
#endif
    extern __shared__ char shared_mem[];

    stk_elem_fixed *s_stk_elem_fixed = (stk_elem_fixed *)shared_mem;
    stk_elem *s_stk_elem = (stk_elem *)(s_stk_elem_fixed + warpsPerBlock * begin_len);

#ifndef UNLABELED
    if (threadIdx.x < Q.vcount()) {
        s_qlabel[threadIdx.x] = Q.d_label_[threadIdx.x];
    }
#endif

#ifdef BALANCE_CNT
    if (lane_id == 0) {
        compute_cnt[warp_id] = 0;
        warp_time[warp_id] = clock64();
    }
#endif

    // Each warp has one stack
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
        init_dfs_stacks(Q, cg, this_stk_elem_fixed,
                        this_stk_elem, d_MM,
                        this_offset, partial_matching_cnt,
                        lane_id, begin_len);

        int this_stk_len = begin_len + 1;

        while (this_stk_len > begin_len) {
            int this_u = this_stk_len - 1;
            stk_elem *e_ptr = &this_stk_elem[this_u];

            // Process the candidate set of the current stack element
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

#ifndef UNLABELED
            if (flag && s_qlabel[this_u] != G.d_label_[lane_v]) {
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

#ifdef BALANCE_CNT
            if (lane_id == 0) {
                compute_cnt[warp_id]++;
            }
#endif

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
                int j = this_stk_len - 2;
                unsigned bn_mask = Q.d_bknbrs_[this_u];
                // For each backward neighbor uu of this_u (enumerate uu in descending order)

                while (bn_mask) {
                    int uu = 31 - __clz(bn_mask);
                    bn_mask &= ~(1u << uu);
                    int vv = -1;

                    while (j > uu) {
                        if (j < begin_len) {
                            vv = this_stk_elem_fixed[j].mapped_v[cur_parent];
                        }
                        else {
                            vv = this_stk_elem[j].mapped_v[cur_parent];
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

                    // Now j == uu
                    if (j < begin_len) {
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

                    int this_len = 0;
                    int *this_set = cg.get_nbr(vv, this_len);

                    if (this_set == e_ptr->cand_set[lane_parent_idx]) {
                        continue;
                    }

                    if (false == binary_search_int(this_set, this_len, lane_v)) {
                        flag = false;
                        break;
                    }
                }

                if (flag) {
                    while (j >= 0) {
                        int vv = -1;
                        if (j < begin_len) {
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
                    while (this_stk_len > begin_len);
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
                        while (this_stk_len > begin_len);
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
                        new_e_ptr->cand_len[__popc(flag_mask)] = -1;
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
                                if (j >= begin_len) {
                                    cur_parent = this_stk_elem[j].parent_idx[cur_parent];
                                    j--;
                                }
                                else {
                                    j = uu;
                                    break;
                                }
                            }

                            // Now j == uu
                            if (j < begin_len) {
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

#ifdef BALANCE_CNT
    if (lane_id == 0) {
        warp_time[warp_id] = clock64() - warp_time[warp_id];
        warp_time[warp_id] /= 1e6;
    }
#endif
}

__global__ void static
dfs_kernel_sym(
    Graph_GPU Q,
    Graph_GPU G,
    candidate_graph_GPU cg,
    ull *sum,
    MemManager *d_MM,
    int partial_matching_cnt,
    int *begin_offset,
    int begin_len,
    int *d_partial_order
) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = tid / warpSize;
    int warp_id_within_blk = warp_id % warpsPerBlock;
    int lane_id = tid % warpSize;

    __shared__ uint32_t s_partial_order[32];
    __shared__ uint32_t s_bknbrs[32];
#ifndef UNLABELED
    __shared__ int s_qlabel[32];
#endif
    extern __shared__ char shared_mem[];

    stk_elem_fixed *s_stk_elem_fixed = (stk_elem_fixed *)shared_mem;
    stk_elem *s_stk_elem = (stk_elem *)(s_stk_elem_fixed + warpsPerBlock * begin_len);

    if (threadIdx.x < Q.vcount()) {
        s_partial_order[threadIdx.x] = d_partial_order[threadIdx.x];
        s_bknbrs[threadIdx.x] = Q.d_bknbrs_[threadIdx.x];
    }

#ifndef UNLABELED
    if (threadIdx.x < Q.vcount()) {
        s_qlabel[threadIdx.x] = Q.d_label_[threadIdx.x];
    }
#endif

#ifdef BALANCE_CNT
    if (lane_id == 0) {
        compute_cnt[warp_id] = 0;
        warp_time[warp_id] = clock64();
    }
#endif

    // Each warp has one stack
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
        init_dfs_stacks(Q, cg, this_stk_elem_fixed,
                        this_stk_elem, d_MM,
                        this_offset, partial_matching_cnt,
                        lane_id, begin_len);

        int this_stk_len = begin_len + 1;

        while (this_stk_len > begin_len) {
            int this_u = this_stk_len - 1;
            int partial_order_u = s_partial_order[this_u];
            stk_elem *e_ptr = &this_stk_elem[this_u];

            // Process the candidate set of the current stack element
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

#ifndef UNLABELED
            if (flag && s_qlabel[this_u] != G.d_label_[lane_v]) {
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

#ifdef BALANCE_CNT
            if (lane_id == 0) {
                compute_cnt[warp_id]++;
            }
#endif

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
                int j = this_stk_len - 2;
                unsigned bn_mask = s_bknbrs[this_u];
                // For each backward neighbor uu of this_u (enumerate uu in descending order)

                bool reach_bound = false;

                while (bn_mask) {
                    int uu = 31 - __clz(bn_mask);
                    bn_mask &= ~(1u << uu);
                    int vv = -1;

                    while (j > uu) {
                        if (j < begin_len) {
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
                        else if (partial_order_u & (1 << (j + 1))) {
                            if (lane_v > vv) {
                                flag = false;
                                reach_bound = true;
                                break;
                            }
                        }
                    }

                    if (flag == false) {
                        break;
                    }

                    int old_cur_parent = cur_parent;

                    if (j < begin_len) {
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
                    else if (partial_order_u & (1 << uu)) {
                        if (lane_v > vv) {
                            flag = false;
                            reach_bound = true;
                            break;
                        }
                    }

                    int this_len = 0;
                    int *this_set = nullptr;

                    if (j + 1 < begin_len) {
                        this_set = this_stk_elem_fixed[j + 1].mapped_v_nbr_set[old_cur_parent];
                        this_len = this_stk_elem_fixed[j + 1].mapped_v_nbr_len[old_cur_parent];
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

                if (flag) {
                    while (j >= 0) {
                        int vv = -1;
                        if (j < begin_len) {
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
                        else if (partial_order_u & (1 << j)) {
                            if (lane_v > vv) {
                                flag = false;
                                reach_bound = true;
                                break;
                            }
                        }
                        j--;
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
                    while (this_stk_len > begin_len);
                }
                this_stk_len = __shfl_sync(FULL_MASK, this_stk_len, 0);
            }
            else {  // flag_mask != 0
                if (this_stk_len == Q.vcount_) {
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
                        while (this_stk_len > begin_len);
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
                        new_e_ptr->cand_len[__popc(flag_mask)] = -1;
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
                        unsigned bn_mask = s_bknbrs[next_u];

                        while (bn_mask) {
                            int uu = 31 - __clz(bn_mask);
                            bn_mask &= ~(1u << uu);
                            int vv = -1;

                            while (j > uu && j >= begin_len) {
                                cur_parent = this_stk_elem[j].parent_idx[cur_parent];
                                j--;
                            }

                            int this_len = 0;
                            int *this_set = nullptr;

                            if (uu < begin_len) {
                                this_len = this_stk_elem_fixed[uu].mapped_v_nbr_len[cur_parent];
                                this_set = this_stk_elem_fixed[uu].mapped_v_nbr_set[cur_parent];
                            }
                            else {
                                this_set = cg.get_nbr(this_stk_elem[uu].mapped_v[cur_parent], this_len);
                            }

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

#ifdef BALANCE_CNT
    if (lane_id == 0) {
        warp_time[warp_id] = clock64() - warp_time[warp_id];
        warp_time[warp_id] /= 1e6;
    }
#endif
}


__global__ void static
dfs_kernel_clique(
    Graph_GPU Q,
    Graph_GPU G,
    candidate_graph_GPU cg,
    ull *sum,
    MemManager *d_MM,
    int partial_matching_cnt,
    int *begin_offset,
    int begin_len
) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = tid / warpSize;
    int warp_id_within_blk = warp_id % warpsPerBlock;
    int lane_id = tid % warpSize;

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
        init_dfs_stacks(Q, cg, this_stk_elem_fixed,
                        this_stk_elem, d_MM,
                        this_offset, partial_matching_cnt,
                        lane_id, begin_len);

        if (lane_id == 0) {
            // int tmp = this_stk_elem_fixed[0].mapped_v[0];
            // this_stk_elem_fixed[0].mapped_v[0] = this_stk_elem_fixed[1].mapped_v[0];
            // this_stk_elem_fixed[1].mapped_v[0] = tmp;

            // this_stk_elem_fixed[0].mapped_v_nbr_set[0] = cg.get_nbr(this_stk_elem_fixed[0].mapped_v[0], this_stk_elem_fixed[0].mapped_v_nbr_len[0]);
            // this_stk_elem_fixed[1].mapped_v_nbr_set[0] = cg.get_nbr(this_stk_elem_fixed[1].mapped_v[0], this_stk_elem_fixed[1].mapped_v_nbr_len[0]);
        }

        int this_stk_len = begin_len + 1;

        while (this_stk_len > begin_len) {
            int this_u = this_stk_len - 1;
            stk_elem *e_ptr = &this_stk_elem[this_u];

            // Process the candidate set of the current stack element
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
                    if (uu < begin_len) {
                        vv = this_stk_elem_fixed[uu].mapped_v[cur_parent];
                    }
                    else {
                        vv = this_stk_elem[uu].mapped_v[cur_parent];
                        cur_parent = this_stk_elem[uu].parent_idx[cur_parent];
                    }

                    // if (lane_v >= vv) {
                    //     flag = false;
                    //     break;
                    // }
                    if (lane_v >= vv) {
                        flag = false;
                        reach_bound = true;
                        break;
                    }

                    int this_len = 0;
                    int *this_set = nullptr;

                    if (uu < begin_len) {
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
                    while (this_stk_len > begin_len);
                }
                this_stk_len = __shfl_sync(FULL_MASK, this_stk_len, 0);
            }
            else {  // flag_mask != 0
                if (this_stk_len == Q.vcount_) {
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
                        while (this_stk_len > begin_len);
                    }
                    this_stk_len = __shfl_sync(FULL_MASK, this_stk_len, 0);
                }
                else {  // *this_stk_len != Q.vcount()
                    if (lane_id == 0) {
                        this_stk_elem[this_stk_len].start_set_idx = 0;
                        this_stk_elem[this_stk_len].start_idx_within_set = 0;
                    }
                    this_stk_len++;

                    // "e_ptr" is the pointer of previous stack top
                    // "new_e_ptr" is the pointer of the level we are about to search
                    stk_elem *new_e_ptr = &this_stk_elem[this_stk_len - 1];

                    if (lane_id == 0) {
                        new_e_ptr->cand_len[__popc(flag_mask)] = -1;
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
join_bfs_dfs(
    Graph &q,
    Graph &g,
    Graph_GPU &Q,
    Graph_GPU &G,
    candidate_graph &_cg,
    candidate_graph_GPU &cg,
    bool no_memory_pool
) {
#ifdef IDLE_CNT
    cudaCheck(cudaMemcpyToSymbol(sample_cnt, &Zero_ull, sizeof(ull)));
    cudaCheck(cudaMemcpyToSymbol(idle_cnt, &Zero_ull, sizeof(ull)));
#endif

    auto data_preparation_start = std::chrono::high_resolution_clock::now();

    int partial_matching_cnt = 0;
    int *d_partial_matchings = set_beginning_partial_matchings(q, g, cg, partial_matching_cnt);
    if (partial_matching_cnt == 0) {
        return 0;
    }

    MemManager h_MM(no_memory_pool);
    h_MM.init(d_partial_matchings, partial_matching_cnt, 0, partial_matching_cnt);
    MemManager *d_MM = nullptr;
    cudaCheck(cudaMalloc(&d_MM, sizeof(MemManager)));
    printf("--- Allocating %d bytes (%d MemManager) for d_MM @ %p\n", sizeof(MemManager), 1, d_MM);
    get_partial_init(&h_MM);
    h_MM.init_prev_head();
    cudaCheck(cudaMemcpy(d_MM, &h_MM, sizeof(MemManager), cudaMemcpyHostToDevice));

    printf("BFS kernel theoretical occupancy %.2f%%\n", calculateOccupancy((const void *)BFS_Extend, threadsPerBlock));

    int *d_error_flag = nullptr;
    cudaCheck(cudaMalloc(&d_error_flag, sizeof(int)));
    printf("--- Allocating %d bytes (%d int) for d_error_flag @ %p\n", sizeof(int), 1, d_error_flag);
    cudaCheck(cudaMemset(d_error_flag, 0, sizeof(int)));

    auto data_preparation_end = std::chrono::high_resolution_clock::now();
    auto data_preparation_duration = std::chrono::duration_cast<std::chrono::microseconds>(data_preparation_end - data_preparation_start);
    std::cout << "Data preparation: " << data_preparation_duration.count() / 1000 << " ms\n";

    TIME_INIT();
    TIME_START();

    int l = 2;
    int expect = EXPECT_BEGIN_NUM;
    // 1e6 should be safe. Decrease this value if any error occurs.

    for (; no_memory_pool == false && partial_matching_cnt < expect && l < q.vcount(); l++) {
        for (int offset = 0; offset < partial_matching_cnt; offset += warpNum) {
            BFS_Extend <<< maxBlocks, threadsPerBlock >>> (
                Q, G, cg, d_MM, l, offset, 0, d_error_flag
            );
            cudaCheck(cudaGetLastError());
            cudaCheck(cudaDeviceSynchronize());

            int h_error_flag = 0;
            cudaCheck(cudaMemcpy(&h_error_flag, d_error_flag, sizeof(int), cudaMemcpyDeviceToHost));
            if (h_error_flag) {
                printf("BFS failed to extend to level %d. Switching to DFS\n", l + 1);
                goto bfs_end;
            }
        }

        BFS_Extend <<< maxBlocks, threadsPerBlock >>> (
            Q, G, cg, d_MM, l, l + 1, 1, d_error_flag
        );
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        cudaCheck(cudaMemcpy(&h_MM, d_MM, sizeof(MemManager), cudaMemcpyDeviceToHost));
        h_MM.swap_mem_pool();
        get_partial_init(&h_MM);
        h_MM.init_prev_head();
        partial_matching_cnt = h_MM.get_partial_cnt();
        cudaCheck(cudaMemcpy(d_MM, &h_MM, sizeof(MemManager), cudaMemcpyHostToDevice));
        printf("BFS extended to level %d. Partial matching count: %d. Number of warps: %d\n",
               l + 1, partial_matching_cnt, warpNum);

        if (partial_matching_cnt == 0) {
            break;
        }
    }

bfs_end:

    TIME_END();
    PRINT_LOCAL_TIME("BFS Finished");

    // Free some global memory for DFS
    h_MM.mempool_to_write()->deallocate();

    if (l == q.vcount() || partial_matching_cnt == 0) {
        PRINT_TOTAL_TIME("Processing");
        return partial_matching_cnt;
    }

    check_gpu_memory();

    // BFS is finished. Preparing for DFS
    printf("Conducting DFS from level %d to %d\n", l + 1, Q.vcount());

    ull *d_sum = nullptr;
    cudaCheck(cudaMalloc(&d_sum, sizeof(ull) * warpNum));
    printf("--- Allocating %d bytes (%d ull) for d_sum @ %p\n", sizeof(ull) * warpNum, warpNum, d_sum);
    cudaCheck(cudaMemset(d_sum, 0, sizeof(ull) * warpNum));

    int *begin_offset = nullptr;
    cudaCheck(cudaMalloc(&begin_offset, sizeof(int)));
    printf("--- Allocating %d bytes (%d int) for begin_offset @ %p\n", sizeof(int), 1, begin_offset);
    cudaCheck(cudaMemset(begin_offset, 0, sizeof(int)));

    int dynamic_shared_size = warpsPerBlock * (Q.vcount() - l) * sizeof(stk_elem)
                              + warpsPerBlock * l * sizeof(stk_elem_fixed);

    printf("Shared memory usage: %.2f KB per thread block\n", dynamic_shared_size / 1024.);
    printf("DFS kernel theoretical occupancy %.2f%%\n", calculateOccupancy((const void *)dfs_kernel, threadsPerBlock, dynamic_shared_size));

    int thread_block_num = 0;
    int current_device;
    cudaGetDevice(&current_device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, current_device);
    int num_SMs = prop.multiProcessorCount;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&thread_block_num, dfs_kernel, threadsPerBlock, dynamic_shared_size);
    thread_block_num = thread_block_num * num_SMs;
    printf("#thread blocks per SM: %d, #SMs: %d, #threads per block: %d\n", thread_block_num / num_SMs, num_SMs, threadsPerBlock);

    TIME_START();
    dfs_kernel <<< thread_block_num, threadsPerBlock, dynamic_shared_size >>> (
        Q, G, cg, d_sum,
        d_MM, partial_matching_cnt,
        begin_offset, l
    );
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
    printStatistics("Computation count", h_compute_cnt, warpNum);
    free(h_compute_cnt);

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


ull
join_bfs_dfs_sym(
    Graph &q,
    Graph &g,
    Graph_GPU &Q,
    Graph_GPU &G,
    candidate_graph &_cg,
    candidate_graph_GPU &cg,
    std::vector<uint32_t> &partial_order,
    bool no_memory_pool
) {
#ifdef IDLE_CNT
    cudaCheck(cudaMemcpyToSymbol(sample_cnt, &Zero_ull, sizeof(ull)));
    cudaCheck(cudaMemcpyToSymbol(idle_cnt, &Zero_ull, sizeof(ull)));
#endif

    auto data_preparation_start = std::chrono::high_resolution_clock::now();

    int partial_matching_cnt = 0;
    int *d_partial_matchings = set_beginning_partial_matchings_sym(q, g, cg, partial_matching_cnt, partial_order);
    if (partial_matching_cnt == 0) {
        return 0;
    }

    MemManager h_MM(no_memory_pool);
    h_MM.init(d_partial_matchings, partial_matching_cnt, 0, partial_matching_cnt);
    MemManager *d_MM = nullptr;
    cudaCheck(cudaMalloc(&d_MM, sizeof(MemManager)));
    printf("--- Allocating %d bytes (%d MemManager) for d_MM @ %p\n", sizeof(MemManager), 1, d_MM);
    get_partial_init(&h_MM);
    h_MM.init_prev_head();
    cudaCheck(cudaMemcpy(d_MM, &h_MM, sizeof(MemManager), cudaMemcpyHostToDevice));

    printf("BFS kernel theoretical occupancy %.2f%%\n", calculateOccupancy((const void *)BFS_Extend_sym, threadsPerBlock));

    int *d_error_flag = nullptr;
    cudaCheck(cudaMalloc(&d_error_flag, sizeof(int)));
    printf("--- Allocating %d bytes (%d int) for d_error_flag @ %p\n", sizeof(int), 1, d_error_flag);
    cudaCheck(cudaMemset(d_error_flag, 0, sizeof(int)));

    auto data_preparation_end = std::chrono::high_resolution_clock::now();
    auto data_preparation_duration = std::chrono::duration_cast<std::chrono::microseconds>(data_preparation_end - data_preparation_start);
    std::cout << "Data preparation: " << data_preparation_duration.count() / 1000 << " ms\n";

    TIME_INIT();
    TIME_START();

    int l = 2;
    int expect = EXPECT_BEGIN_NUM;
    int *d_partial_order = nullptr;
    cudaCheck(cudaMalloc(&d_partial_order, sizeof(uint32_t) * q.vcount()));
    printf("--- Allocating %d bytes (%d uint32_t) for d_partial_order @ %p\n", sizeof(uint32_t) * q.vcount(), q.vcount(), d_partial_order);
    cudaCheck(cudaMemcpy(d_partial_order, partial_order.data(), sizeof(uint32_t) * q.vcount(), cudaMemcpyHostToDevice));

    for (; no_memory_pool == false && partial_matching_cnt < expect && l < q.vcount(); l++) {
        for (int offset = 0; offset < partial_matching_cnt; offset += warpNum) {
            BFS_Extend_sym <<< maxBlocks, threadsPerBlock >>> (
                Q, G, cg, d_MM, l, offset, 0, d_error_flag, d_partial_order
            );
            cudaCheck(cudaGetLastError());
            cudaCheck(cudaDeviceSynchronize());

            int h_error_flag = 0;
            cudaCheck(cudaMemcpy(&h_error_flag, d_error_flag, sizeof(int), cudaMemcpyDeviceToHost));
            if (h_error_flag) {
                printf("BFS failed to extend to level %d. Switching to DFS\n", l + 1);
                goto bfs_end;
            }
        }

        BFS_Extend_sym <<< maxBlocks, threadsPerBlock >>> (
            Q, G, cg, d_MM, l, l + 1, 1, d_error_flag, d_partial_order
        );
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        cudaCheck(cudaMemcpy(&h_MM, d_MM, sizeof(MemManager), cudaMemcpyDeviceToHost));
        h_MM.swap_mem_pool();
        get_partial_init(&h_MM);
        h_MM.init_prev_head();
        partial_matching_cnt = h_MM.get_partial_cnt();
        cudaCheck(cudaMemcpy(d_MM, &h_MM, sizeof(MemManager), cudaMemcpyHostToDevice));
        printf("BFS extended to level %d. Partial matching count: %d. Number of warps: %d\n",
               l + 1, partial_matching_cnt, warpNum);

        if (partial_matching_cnt == 0) {
            break;
        }
    }

bfs_end:

    TIME_END();
    PRINT_LOCAL_TIME("BFS Finished");

    // Free some global memory for DFS
    h_MM.mempool_to_write()->deallocate();

    if (l == q.vcount() || partial_matching_cnt == 0) {
        PRINT_TOTAL_TIME("Processing");
        return partial_matching_cnt;
    }

    check_gpu_memory();

    // BFS is finished. Preparing for DFS
    printf("Conducting DFS from level %d to %d\n", l + 1, Q.vcount());

    ull *d_sum = nullptr;
    cudaCheck(cudaMalloc(&d_sum, sizeof(ull) * warpNum));
    printf("--- Allocating %d bytes (%d ull) for d_sum @ %p\n", sizeof(ull) * warpNum, warpNum, d_sum);
    cudaCheck(cudaMemset(d_sum, 0, sizeof(ull) * warpNum));

    int *begin_offset = nullptr;
    cudaCheck(cudaMalloc(&begin_offset, sizeof(int)));
    printf("--- Allocating %d bytes (%d int) for begin_offset @ %p\n", sizeof(int), 1, begin_offset);
    cudaCheck(cudaMemset(begin_offset, 0, sizeof(int)));

    int dynamic_shared_size = warpsPerBlock * (Q.vcount() - l) * sizeof(stk_elem)
                              + warpsPerBlock * l * sizeof(stk_elem_fixed);

    printf("Shared memory usage: %.2f KB per thread block\n", dynamic_shared_size / 1024.);
    printf("DFS kernel theoretical occupancy %.2f%%\n", calculateOccupancy((const void *)dfs_kernel, threadsPerBlock, dynamic_shared_size));

    int thread_block_num = 0;
    int current_device;
    cudaGetDevice(&current_device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, current_device);
    int num_SMs = prop.multiProcessorCount;

    bool unlabeled = true;
#ifndef UNLABELED
    unlabeled = false;
#endif

    TIME_START();
    if (!q.is_clique() || !unlabeled) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&thread_block_num, dfs_kernel, threadsPerBlock, dynamic_shared_size);
        thread_block_num = thread_block_num * num_SMs;
        printf("#thread blocks per SM: %d, #SMs: %d, #threads per block: %d\n", thread_block_num / num_SMs, num_SMs, threadsPerBlock);
        dfs_kernel_sym <<< thread_block_num, threadsPerBlock, dynamic_shared_size >>> (
            Q, G, cg, d_sum,
            d_MM, partial_matching_cnt,
            begin_offset, l,
            d_partial_order
        );
    }
    else {
        printf("### clique\n");
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&thread_block_num, dfs_kernel_clique, threadsPerBlock, dynamic_shared_size);
        thread_block_num = thread_block_num * num_SMs;
        printf("#thread blocks per SM: %d, #SMs: %d, #threads per block: %d\n", thread_block_num / num_SMs, num_SMs, threadsPerBlock);
        dfs_kernel_clique <<< thread_block_num, threadsPerBlock, dynamic_shared_size >>> (
            Q, G, cg, d_sum,
            d_MM, partial_matching_cnt,
            begin_offset, l
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
    printStatistics("Computation count", h_compute_cnt, warpNum);
    free(h_compute_cnt);

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