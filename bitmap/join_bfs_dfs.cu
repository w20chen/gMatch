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

static __host__ bool
is_trivial(int level, int partial_num) {
    return level > 9 && partial_num < 1000000;
}

static __device__ void
init_dfs_stacks(
    const Graph_GPU Q,
    const candidate_graph_GPU cg,

    int *this_stk_len,
    stk_elem_fixed *this_stk_elem_fixed,
    stk_elem *this_stk_elem,
    stk_elem_cand *first_cand,

    MemManager *d_MM,
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
            this_stk_elem_fixed[depth].mapped_idx[lane_id] = this_partial_matching[depth];
            this_stk_elem_fixed[depth].mapped_v[lane_id] = cg.d_get_mapped_v(depth, this_partial_matching[depth]);
        }

        // Calculate the minimum length of the candidate set for the current partial matching
        int next_u = begin_len;
        unsigned bn_mask = Q.d_bknbrs_[next_u];
        int first_uu = __ffs(bn_mask) - 1;
        bn_mask &= ~(1 << first_uu);
        int first_vv = this_partial_matching[first_uu];

        CandLen_t min_len = 0;
        int min_set = cg.d_get_candidates_offset(first_uu, next_u, first_vv, min_len);
        char min_set_u = first_uu;

        while (bn_mask) {
            int uu = __ffs(bn_mask) - 1;
            bn_mask &= ~(1u << uu);
            int vv = this_partial_matching[uu];
            CandLen_t this_len = 0;
            int this_set = cg.d_get_candidates_offset(uu, next_u, vv, this_len);
            if (this_len < min_len) {
                min_len = this_len;
                min_set = this_set;
                min_set_u = uu;
            }
        }

        first_cand->cand_set[lane_id] = min_set;
        first_cand->cand_len[lane_id] = min_len;
#ifdef BITMAP_SET_INTERSECTION
        e_ptr->cand_set_u[lane_id] = min_set_u;
#endif
    }

    if (lane_id == 0) {
        if (this_offset + UNROLL_MIN > partial_matching_cnt) {
            e_ptr->cand.cand_len[partial_matching_cnt - this_offset] = -1;
        }
        else {
            e_ptr->cand.cand_len[UNROLL_MIN] = -1;
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
    const candidate_graph_GPU cg,
    ull *sum,                           // Sum of the number of valid partial matchings
    MemManager *d_MM,                   // Where previous partial results are obtained
    int partial_matching_cnt,           // Total number of the beginning partial results
    int *begin_offset,                  // Atomic variable denoting the next unsolved partial matching
    const int begin_len,                // Length of the beginning partial results
    stk_elem_cand *d_stk_elem_cand,     // Candidate set pointers and lengths allocated on global memory
    const unsigned label_mask,
    const unsigned backward_mask
) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;    // Global thread id
    int warp_id = tid / warpSize;                       // Global warp id
    int warp_id_within_blk = warp_id % warpsPerBlock;   // Warp id within the block
    int lane_id = tid % warpSize;                       // Lane id within the warp

    __shared__ int s_len[warpsPerBlock];
    extern __shared__ char shared_mem[];

    stk_elem_fixed *s_stk_elem_fixed = (stk_elem_fixed *)shared_mem;
    stk_elem *s_stk_elem = (stk_elem *)(s_stk_elem_fixed + warpsPerBlock * begin_len);

#ifdef BALANCE_CNT
    bool assigned = false;
    if (lane_id == 0) {
        compute_cnt[warp_id] = 0;
        warp_time[warp_id] = clock64();
    }
#endif

    // Each warp has one stack
    int *this_stk_len = s_len + warp_id_within_blk;
    stk_elem_fixed *this_stk_elem_fixed = s_stk_elem_fixed + warp_id_within_blk * begin_len;
    stk_elem *this_stk_elem = s_stk_elem + warp_id_within_blk * (Q.vcount() - begin_len) - begin_len;
    stk_elem_cand *first_cand = nullptr;

#ifndef STK_ELEM_CAND_ON_SHARED
    stk_elem_cand *this_stk_elem_cand = d_stk_elem_cand + warp_id * (Q.vcount() - begin_len) - begin_len;
    first_cand = &this_stk_elem_cand[begin_len];
#else
    first_cand = &this_stk_elem[begin_len].cand;
#endif

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

#ifdef BALANCE_CNT
        assigned = true;
#endif

        // Initialize the stack with "UNROLL_MIN" beginning partial matchings
        init_dfs_stacks(Q, cg, this_stk_len, this_stk_elem_fixed,
                        this_stk_elem, first_cand, d_MM,
                        this_offset, partial_matching_cnt,
                        lane_id, begin_len);

        while (*this_stk_len > begin_len) {
            int this_u = *this_stk_len - 1;
            stk_elem *e_ptr = &this_stk_elem[this_u];
#ifndef STK_ELEM_CAND_ON_SHARED
            stk_elem_cand *cand_ptr = &this_stk_elem_cand[this_u];
#else
            stk_elem_cand *cand_ptr = &e_ptr->cand;
#endif
            // Process the candidate set of the current stack element
            int lane_parent_idx = -1;
            int lane_idx_within_set = -1;
            int lane_v = -1;
            int real_lane_v = -this_u - 1;

            int prefix_sum = -e_ptr->start_idx_within_set;
            for (int i = e_ptr->start_set_idx; e_ptr->cand.cand_len[i] != -1; i++) {
                prefix_sum += cand_ptr->cand_len[i];
                if (prefix_sum > lane_id) {
                    lane_parent_idx = i;
                    lane_idx_within_set = lane_id - prefix_sum + cand_ptr->cand_len[i];
                    lane_v = (cand_ptr->cand_set[lane_parent_idx] + cg.d_cg_array)[lane_idx_within_set];
                    if (backward_mask & (1 << this_u)) {
                        real_lane_v = cg.d_get_mapped_v(this_u, lane_v);
                    }
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
                else if (lane_idx_within_set == cand_ptr->cand_len[lane_parent_idx] - 1) {
                    e_ptr->start_set_idx = lane_parent_idx + 1;
                    e_ptr->start_idx_within_set = 0;
                    if (e_ptr->cand.cand_len[e_ptr->start_set_idx] == -1) {
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
                int j = *this_stk_len - 2;
                unsigned bn_mask = Q.d_bknbrs_[this_u];
                // For each backward neighbor uu of this_u (enumerate uu in descending order)
                while (bn_mask) {
                    int uu = 31 - __clz(bn_mask);
                    bn_mask &= ~(1u << uu);
                    int vv = -1;
                    int real_vv = -1;

#ifdef BITMAP_SET_INTERSECTION
                    if (e_ptr->cand_set_u[lane_parent_idx] == uu) {
                        continue;
                    }
#endif

                    if (real_lane_v >= 0) {
                        while (j > uu) {
                            if (j < begin_len) {
                                real_vv = this_stk_elem_fixed[j].mapped_v[cur_parent];
                            }
                            else {
                                real_vv = this_stk_elem[j].mapped_v[cur_parent];
                                cur_parent = this_stk_elem[j].parent_idx[cur_parent];
                            }
                            j--;

                            if (real_vv == real_lane_v) {
                                // lane_v has been mapped before
                                flag = false;
                                break;
                            }
                        }

                        if (flag == false) {
                            break;
                        }
                    }
                    else {
                        while (j > uu) {
                            if (j >= begin_len) {
                                cur_parent = this_stk_elem[j].parent_idx[cur_parent];
                            }
                            j--;
                        }
                    }

                    // Now j == uu
                    if (real_lane_v >= 0) {
                        if (j < begin_len) {
                            real_vv = this_stk_elem_fixed[j].mapped_v[cur_parent];
                            vv = this_stk_elem_fixed[j].mapped_idx[cur_parent];
                        }
                        else {
                            real_vv = this_stk_elem[j].mapped_v[cur_parent];
                            vv = this_stk_elem[j].mapped_idx[cur_parent];
                            cur_parent = this_stk_elem[j].parent_idx[cur_parent];
                        }
                        j--;

                        if (real_vv == real_lane_v) {
                            flag = false;
                            break;
                        }
                    }
                    else {
                        if (j < begin_len) {
                            vv = this_stk_elem_fixed[j].mapped_idx[cur_parent];
                        }
                        else {
                            vv = this_stk_elem[j].mapped_idx[cur_parent];
                            cur_parent = this_stk_elem[j].parent_idx[cur_parent];
                        }
                        j--;
                    }

#ifdef BITMAP_SET_INTERSECTION
                    if (false == cg.d_check_existence(uu, this_u, vv, lane_v)) {
                        flag = false;
                        break;
                    }
#else
                    CandLen_t this_len = 0;
                    int this_set = cg.d_get_candidates_offset(uu, this_u, vv, this_len);

                    if (this_set == cand_ptr->cand_set[lane_parent_idx]) {
                        continue;
                    }

                    if (false == binary_search<CandLen_t>(cg.d_cg_array + this_set, this_len, lane_v)) {
                        flag = false;
                        break;
                    }
#endif
                }

                if (flag && real_lane_v >= 0) {
                    while (j >= 0) {
                        int real_vv = -1;
                        if (j < begin_len) {
                            real_vv = this_stk_elem_fixed[j].mapped_v[cur_parent];
                        }
                        else {
                            real_vv = this_stk_elem[j].mapped_v[cur_parent];
                            cur_parent = this_stk_elem[j].parent_idx[cur_parent];
                        }
                        if (real_vv == real_lane_v) {
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
                        if (this_stk_elem[*this_stk_len - 1].start_set_idx != -1) {
                            break;
                        }
                        else {
                            (*this_stk_len)--;
                        }
                    }
                    while (*this_stk_len > begin_len);
                }
                __syncwarp();
            }
            else {  // flag_mask != 0
                if (*this_stk_len == Q.vcount()) {
                    if (lane_id == 0) {
                        sum[warp_id] += (ull)__popc(flag_mask);
                        do {
                            if (this_stk_elem[*this_stk_len - 1].start_set_idx != -1) {
                                break;
                            }
                            else {
                                (*this_stk_len)--;
                            }
                        }
                        while (*this_stk_len > begin_len);
                    }
                    __syncwarp();
                }
                else {  // *this_stk_len != Q.vcount()
                    if (lane_id == 0) {
                        this_stk_elem[*this_stk_len].start_set_idx = 0;
                        this_stk_elem[*this_stk_len].start_idx_within_set = 0;
                        (*this_stk_len)++;
                    }
                    __syncwarp();

                    // "e_ptr" is the pointer of previous stack top
                    // "new_e_ptr" is the pointer of the level we are about to search
                    int next_u = *this_stk_len - 1;
                    stk_elem *new_e_ptr = &this_stk_elem[next_u];
#ifndef STK_ELEM_CAND_ON_SHARED
                    stk_elem_cand *new_cand_ptr = &this_stk_elem_cand[next_u];
#else
                    stk_elem_cand *new_cand_ptr = &new_e_ptr->cand;
#endif

                    if (lane_id == 0) {
                        new_e_ptr->cand.cand_len[__popc(flag_mask)] = -1;
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
                        e_ptr->mapped_idx[chosen_index] = lane_v;
                        if (real_lane_v < 0 && (label_mask & (1 << this_u))) {
                            real_lane_v = cg.d_get_mapped_v(this_u, lane_v);
                        }
                        e_ptr->mapped_v[chosen_index] = real_lane_v;

                        int j = *this_stk_len - 2;
                        int cur_parent = chosen_index;
                        int min_set = 0;
#ifdef SHORT_CANDIDATE_SET
                        CandLen_t min_len = SHRT_MAX;
#else
                        CandLen_t min_len = INT_MAX;
#endif
                        char min_set_u = 0;
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
                                vv = this_stk_elem_fixed[j].mapped_idx[cur_parent];
                            }
                            else {
                                vv = this_stk_elem[j].mapped_idx[cur_parent];
                            }

                            CandLen_t this_len = 0;
                            int this_set = cg.d_get_candidates_offset(uu, next_u, vv, this_len);

                            if (min_len > this_len) {
                                min_len = this_len;
                                min_set = this_set;
                                min_set_u = uu;
                            }
                        }

                        new_cand_ptr->cand_set[chosen_index] = min_set;
                        new_cand_ptr->cand_len[chosen_index] = min_len;
#ifdef BITMAP_SET_INTERSECTION
                        new_e_ptr->cand_set_u[chosen_index] = min_set_u;
#endif
                    }
                    __syncwarp();
                }
            }
        }
    }

#ifdef BALANCE_CNT
    if (lane_id == 0) {
        if (assigned) {
            warp_time[warp_id] = (clock64() - warp_time[warp_id]) / 1e6;
        }
        else {
            warp_time[warp_id] = 0;
        }
    }
#endif
}


__global__ void static
dfs_kernel_sym(
    const Graph_GPU Q,
    const candidate_graph_GPU cg,
    ull *sum,
    MemManager *d_MM,
    int partial_matching_cnt,
    int *begin_offset,
    const int begin_len,
    stk_elem_cand *d_stk_elem_cand,
    const unsigned label_mask,
    const unsigned backward_mask,
    int *d_partial_order
) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = tid / warpSize;
    int warp_id_within_blk = warp_id % warpsPerBlock;
    int lane_id = tid % warpSize;

    __shared__ int s_len[warpsPerBlock];
    __shared__ uint32_t s_partial_order[32];
    extern __shared__ char shared_mem[];

    stk_elem_fixed *s_stk_elem_fixed = (stk_elem_fixed *)shared_mem;
    stk_elem *s_stk_elem = (stk_elem *)(s_stk_elem_fixed + warpsPerBlock * begin_len);

    if (threadIdx.x < Q.vcount()) {
        s_partial_order[threadIdx.x] = d_partial_order[threadIdx.x];
    }

#ifdef BALANCE_CNT
    bool assigned = false;
    if (lane_id == 0) {
        compute_cnt[warp_id] = 0;
        warp_time[warp_id] = clock64();
    }
#endif

    int *this_stk_len = s_len + warp_id_within_blk;
    stk_elem_fixed *this_stk_elem_fixed = s_stk_elem_fixed + warp_id_within_blk * begin_len;
    stk_elem *this_stk_elem = s_stk_elem + warp_id_within_blk * (Q.vcount() - begin_len) - begin_len;
    stk_elem_cand *first_cand = nullptr;

#ifndef STK_ELEM_CAND_ON_SHARED
    stk_elem_cand *this_stk_elem_cand = d_stk_elem_cand + warp_id * (Q.vcount() - begin_len) - begin_len;
    first_cand = &this_stk_elem_cand[begin_len];
#else
    first_cand = &this_stk_elem[begin_len].cand;
#endif

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

#ifdef BALANCE_CNT
        assigned = true;
#endif

        init_dfs_stacks(Q, cg, this_stk_len, this_stk_elem_fixed,
                        this_stk_elem, first_cand, d_MM,
                        this_offset, partial_matching_cnt,
                        lane_id, begin_len);

        while (*this_stk_len > begin_len) {
            int this_u = *this_stk_len - 1;
            stk_elem *e_ptr = &this_stk_elem[this_u];
#ifndef STK_ELEM_CAND_ON_SHARED
            stk_elem_cand *cand_ptr = &this_stk_elem_cand[this_u];
#else
            stk_elem_cand *cand_ptr = &e_ptr->cand;
#endif
            int lane_parent_idx = -1;
            int lane_idx_within_set = -1;
            int lane_v = -1;
            int real_lane_v = -this_u - 1;

            int prefix_sum = -e_ptr->start_idx_within_set;
            for (int i = e_ptr->start_set_idx; e_ptr->cand.cand_len[i] != -1; i++) {
                prefix_sum += cand_ptr->cand_len[i];
                if (prefix_sum > lane_id) {
                    lane_parent_idx = i;
                    lane_idx_within_set = lane_id - prefix_sum + cand_ptr->cand_len[i];
                    lane_v = (cand_ptr->cand_set[lane_parent_idx] + cg.d_cg_array)[lane_idx_within_set];
                    if (backward_mask & (1 << this_u)) {
                        real_lane_v = cg.d_get_mapped_v(this_u, lane_v);
                    }
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
                else if (lane_idx_within_set == cand_ptr->cand_len[lane_parent_idx] - 1) {
                    e_ptr->start_set_idx = lane_parent_idx + 1;
                    e_ptr->start_idx_within_set = 0;
                    if (e_ptr->cand.cand_len[e_ptr->start_set_idx] == -1) {
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
                int j = *this_stk_len - 2;
                unsigned bn_mask = Q.d_bknbrs_[this_u];
                while (bn_mask) {
                    int uu = 31 - __clz(bn_mask);
                    bn_mask &= ~(1u << uu);
                    int vv = -1;
                    int real_vv = -1;

#ifdef BITMAP_SET_INTERSECTION
                    if (e_ptr->cand_set_u[lane_parent_idx] == uu) {
                        continue;
                    }
#endif
                    if (real_lane_v >= 0) {
                        while (j > uu) {
                            if (j < begin_len) {
                                real_vv = this_stk_elem_fixed[j].mapped_v[cur_parent];
                            }
                            else {
                                real_vv = this_stk_elem[j].mapped_v[cur_parent];
                                cur_parent = this_stk_elem[j].parent_idx[cur_parent];
                            }
                            j--;

                            if (real_vv == real_lane_v) {
                                flag = false;
                                break;
                            }
                            else if (s_partial_order[j + 1] & (1 << this_u)) {
                                if (real_lane_v > real_vv) {
                                    flag = false;
                                    break;
                                }
                            }
                        }

                        if (flag == false) {
                            break;
                        }
                    }
                    else {
                        while (j > uu) {
                            if (j >= begin_len) {
                                cur_parent = this_stk_elem[j].parent_idx[cur_parent];
                            }
                            j--;
                        }
                    }

                    if (real_lane_v >= 0) {
                        if (j < begin_len) {
                            real_vv = this_stk_elem_fixed[j].mapped_v[cur_parent];
                            vv = this_stk_elem_fixed[j].mapped_idx[cur_parent];
                        }
                        else {
                            real_vv = this_stk_elem[j].mapped_v[cur_parent];
                            vv = this_stk_elem[j].mapped_idx[cur_parent];
                            cur_parent = this_stk_elem[j].parent_idx[cur_parent];
                        }
                        j--;

                        if (real_vv == real_lane_v) {
                            flag = false;
                            break;
                        }
                        else if (s_partial_order[uu] & (1 << this_u)) {
                            if (real_lane_v > real_vv) {
                                flag = false;
                                break;
                            }
                        }
                    }
                    else {
                        if (j < begin_len) {
                            vv = this_stk_elem_fixed[j].mapped_idx[cur_parent];
                        }
                        else {
                            vv = this_stk_elem[j].mapped_idx[cur_parent];
                            cur_parent = this_stk_elem[j].parent_idx[cur_parent];
                        }
                        j--;
                    }

#ifdef BITMAP_SET_INTERSECTION
                    if (false == cg.d_check_existence(uu, this_u, vv, lane_v)) {
                        flag = false;
                        break;
                    }
#else
                    CandLen_t this_len = 0;
                    int this_set = cg.d_get_candidates_offset(uu, this_u, vv, this_len);

                    if (this_set == cand_ptr->cand_set[lane_parent_idx]) {
                        continue;
                    }

                    if (false == binary_search<CandLen_t>(cg.d_cg_array + this_set, this_len, lane_v)) {
                        flag = false;
                        break;
                    }
#endif
                }

                if (flag && real_lane_v >= 0) {
                    while (j >= 0) {
                        int real_vv = -1;
                        if (j < begin_len) {
                            real_vv = this_stk_elem_fixed[j].mapped_v[cur_parent];
                        }
                        else {
                            real_vv = this_stk_elem[j].mapped_v[cur_parent];
                            cur_parent = this_stk_elem[j].parent_idx[cur_parent];
                        }
                        if (real_vv == real_lane_v) {
                            flag = false;
                            break;
                        }
                        else if (s_partial_order[j] & (1 << this_u)) {
                            if (real_lane_v > real_vv) {
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
                        if (this_stk_elem[*this_stk_len - 1].start_set_idx != -1) {
                            break;
                        }
                        else {
                            (*this_stk_len)--;
                        }
                    }
                    while (*this_stk_len > begin_len);
                }
                __syncwarp();
            }
            else {
                if (*this_stk_len == Q.vcount()) {
                    if (lane_id == 0) {
                        sum[warp_id] += (ull)__popc(flag_mask);
                        do {
                            if (this_stk_elem[*this_stk_len - 1].start_set_idx != -1) {
                                break;
                            }
                            else {
                                (*this_stk_len)--;
                            }
                        }
                        while (*this_stk_len > begin_len);
                    }
                    __syncwarp();
                }
                else {
                    if (lane_id == 0) {
                        this_stk_elem[*this_stk_len].start_set_idx = 0;
                        this_stk_elem[*this_stk_len].start_idx_within_set = 0;
                        (*this_stk_len)++;
                    }
                    __syncwarp();

                    int next_u = *this_stk_len - 1;
                    stk_elem *new_e_ptr = &this_stk_elem[next_u];
#ifndef STK_ELEM_CAND_ON_SHARED
                    stk_elem_cand *new_cand_ptr = &this_stk_elem_cand[next_u];
#else
                    stk_elem_cand *new_cand_ptr = &new_e_ptr->cand;
#endif

                    if (lane_id == 0) {
                        new_e_ptr->cand.cand_len[__popc(flag_mask)] = -1;
                    }
                    __syncwarp();

                    int chosen_index = -1;
                    if (flag_mask & (1 << lane_id)) {
                        unsigned mask_low = flag_mask & ((1 << (lane_id + 1)) - 1);
                        chosen_index = __popc(mask_low) - 1;
                    }

                    if (chosen_index >= 0) {
                        e_ptr->parent_idx[chosen_index] = lane_parent_idx;
                        e_ptr->mapped_idx[chosen_index] = lane_v;
                        if (real_lane_v < 0 && (label_mask & (1 << this_u))) {
                            real_lane_v = cg.d_get_mapped_v(this_u, lane_v);
                        }
                        e_ptr->mapped_v[chosen_index] = real_lane_v;

                        int j = *this_stk_len - 2;
                        int cur_parent = chosen_index;
                        int min_set = 0;
#ifdef SHORT_CANDIDATE_SET
                        CandLen_t min_len = SHRT_MAX;
#else
                        CandLen_t min_len = INT_MAX;
#endif
                        char min_set_u = 0;
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

                            if (j < begin_len) {
                                vv = this_stk_elem_fixed[j].mapped_idx[cur_parent];
                            }
                            else {
                                vv = this_stk_elem[j].mapped_idx[cur_parent];
                            }

                            CandLen_t this_len = 0;
                            int this_set = cg.d_get_candidates_offset(uu, next_u, vv, this_len);

                            if (min_len > this_len) {
                                min_len = this_len;
                                min_set = this_set;
                                min_set_u = uu;
                            }
                        }

                        new_cand_ptr->cand_set[chosen_index] = min_set;
                        new_cand_ptr->cand_len[chosen_index] = min_len;
#ifdef BITMAP_SET_INTERSECTION
                        new_e_ptr->cand_set_u[chosen_index] = min_set_u;
#endif
                    }
                    __syncwarp();
                }
            }
        }
    }

#ifdef BALANCE_CNT
    if (lane_id == 0) {
        if (assigned) {
            warp_time[warp_id] = (clock64() - warp_time[warp_id]) / 1e6;
        }
        else {
            warp_time[warp_id] = 0;
        }
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
    const unsigned label_mask,
    const unsigned backward_mask,
    const int expected_init_num
) {
#ifdef IDLE_CNT
    cudaCheck(cudaMemcpyToSymbol(sample_cnt, &Zero_ull, sizeof(ull)));
    cudaCheck(cudaMemcpyToSymbol(idle_cnt, &Zero_ull, sizeof(ull)));
#endif

    int partial_matching_cnt = 0;
    int *d_partial_matchings = set_beginning_partial_matchings(q, g, _cg, partial_matching_cnt);
    if (partial_matching_cnt == 0) {
        return 0;
    }

    MemManager h_MM;
    h_MM.init(d_partial_matchings, partial_matching_cnt, 0, partial_matching_cnt);
    MemManager *d_MM = nullptr;
    cudaCheck(cudaMalloc(&d_MM, sizeof(MemManager)));
    get_partial_init(&h_MM);
    h_MM.init_prev_head();
    cudaCheck(cudaMemcpy(d_MM, &h_MM, sizeof(MemManager), cudaMemcpyHostToDevice));

    printf("BFS kernel theoretical occupancy %.2f%%\n", calculateOccupancy((const void *)BFS_Extend, threadsPerBlock));

    int *d_error_flag = nullptr;
    cudaCheck(cudaMalloc(&d_error_flag, sizeof(int)));
    cudaCheck(cudaMemset(d_error_flag, 0, sizeof(int)));

    TIME_INIT();
    TIME_START();

    int l = 2;
    for (; partial_matching_cnt < expected_init_num && l < q.vcount(); l++) {
        if (is_trivial(l, partial_matching_cnt)) {
            break;
        }
        for (int offset = 0; offset < partial_matching_cnt; offset += warpNum) {
            BFS_Extend <<< maxBlocks, threadsPerBlock >>> (
                Q, G, cg, d_MM, l, offset, partial_matching_cnt, 0, d_error_flag
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
            Q, G, cg, d_MM, l, l + 1, partial_matching_cnt, 1, d_error_flag
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
    h_MM.mempool_to_read()->print_meta();

    if (l == q.vcount() || partial_matching_cnt == 0) {
        PRINT_TOTAL_TIME("Processing");
        return partial_matching_cnt;
    }

    check_gpu_memory();

    // BFS is finished. Preparing for DFS
    printf("Conducting DFS from level %d to %d\n", l + 1, Q.vcount());

    ull *d_sum = nullptr;
    cudaCheck(cudaMalloc(&d_sum, sizeof(ull) * warpNum));
    cudaCheck(cudaMemset(d_sum, 0, sizeof(ull) * warpNum));

    int *begin_offset = nullptr;
    cudaCheck(cudaMalloc(&begin_offset, sizeof(int)));
    cudaCheck(cudaMemset(begin_offset, 0, sizeof(int)));

    stk_elem_cand *d_stk_elem_cand = nullptr;
    int dynamic_shared_size = warpsPerBlock * (Q.vcount() - l) * sizeof(stk_elem)
                              + warpsPerBlock * l * sizeof(stk_elem_fixed);

#ifndef STK_ELEM_CAND_ON_SHARED
    cudaCheck(cudaMalloc(&d_stk_elem_cand, warpNum * (Q.vcount() - l) * sizeof(stk_elem_cand)));
#endif
    printf("Shared memory usage: %.2f KB per thread block\n", (dynamic_shared_size + (int)sizeof(int) * warpsPerBlock) / 1024.0);
    printf("DFS kernel theoretical occupancy %.2f%%\n", calculateOccupancy((const void *)dfs_kernel, threadsPerBlock, dynamic_shared_size));

    TIME_START();
    dfs_kernel <<< maxBlocks, threadsPerBlock, dynamic_shared_size >>> (
        Q, cg, d_sum,
        d_MM, partial_matching_cnt,
        begin_offset, l,
        d_stk_elem_cand,
        label_mask, backward_mask
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
    if (d_stk_elem_cand) {
        cudaCheck(cudaFree(d_stk_elem_cand));
    }

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
    const Graph &q,
    const Graph &g,
    const Graph_GPU &Q,
    const Graph_GPU &G,
    const candidate_graph &_cg,
    const candidate_graph_GPU &cg,
    unsigned label_mask,
    unsigned backward_mask,
    const int expected_init_num,
    const std::vector<uint32_t> &partial_order
) {
#ifdef IDLE_CNT
    cudaCheck(cudaMemcpyToSymbol(sample_cnt, &Zero_ull, sizeof(ull)));
    cudaCheck(cudaMemcpyToSymbol(idle_cnt, &Zero_ull, sizeof(ull)));
#endif

    int partial_matching_cnt = 0;
    int *d_partial_matchings = set_beginning_partial_matchings_sym(q, g, _cg, partial_matching_cnt, partial_order);
    if (partial_matching_cnt == 0) {
        return 0;
    }

    MemManager h_MM;
    h_MM.init(d_partial_matchings, partial_matching_cnt, 0, partial_matching_cnt);
    MemManager *d_MM = nullptr;
    cudaCheck(cudaMalloc(&d_MM, sizeof(MemManager)));
    get_partial_init(&h_MM);
    h_MM.init_prev_head();
    cudaCheck(cudaMemcpy(d_MM, &h_MM, sizeof(MemManager), cudaMemcpyHostToDevice));

    printf("BFS kernel theoretical occupancy %.2f%%\n", calculateOccupancy((const void *)BFS_Extend, threadsPerBlock));

    int *d_error_flag = nullptr;
    cudaCheck(cudaMalloc(&d_error_flag, sizeof(int)));
    cudaCheck(cudaMemset(d_error_flag, 0, sizeof(int)));

    TIME_INIT();
    TIME_START();

    int l = 2;

    int *d_partial_order = nullptr;
    cudaCheck(cudaMalloc(&d_partial_order, sizeof(uint32_t) * q.vcount()));
    cudaCheck(cudaMemcpy(d_partial_order, partial_order.data(), sizeof(uint32_t) * q.vcount(), cudaMemcpyHostToDevice));

    for (; partial_matching_cnt < expected_init_num && l < q.vcount(); l++) {
        if (is_trivial(l, partial_matching_cnt)) {
            break;
        }
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
    h_MM.mempool_to_read()->print_meta();

    if (l == q.vcount() || partial_matching_cnt == 0) {
        PRINT_TOTAL_TIME("Processing");
        return partial_matching_cnt;
    }

    check_gpu_memory();

    // BFS is finished. Preparing for DFS
    printf("Conducting DFS from level %d to %d\n", l + 1, Q.vcount());

    ull *d_sum = nullptr;
    cudaCheck(cudaMalloc(&d_sum, sizeof(ull) * warpNum));
    cudaCheck(cudaMemset(d_sum, 0, sizeof(ull) * warpNum));

    int *begin_offset = nullptr;
    cudaCheck(cudaMalloc(&begin_offset, sizeof(int)));
    cudaCheck(cudaMemset(begin_offset, 0, sizeof(int)));

    stk_elem_cand *d_stk_elem_cand = nullptr;
    int dynamic_shared_size = warpsPerBlock * (Q.vcount() - l) * sizeof(stk_elem)
                              + warpsPerBlock * l * sizeof(stk_elem_fixed);

#ifndef STK_ELEM_CAND_ON_SHARED
    cudaCheck(cudaMalloc(&d_stk_elem_cand, warpNum * (Q.vcount() - l) * sizeof(stk_elem_cand)));
#endif
    printf("Shared memory usage: %.2f KB per thread block\n", (dynamic_shared_size + (int)sizeof(int) * warpsPerBlock) / 1024.0);
    printf("DFS kernel theoretical occupancy %.2f%%\n", calculateOccupancy((const void *)dfs_kernel, threadsPerBlock, dynamic_shared_size));

    TIME_START();
    dfs_kernel_sym <<< maxBlocks, threadsPerBlock, dynamic_shared_size >>> (
        Q, cg, d_sum,
        d_MM, partial_matching_cnt,
        begin_offset, l,
        d_stk_elem_cand,
        label_mask,
        backward_mask,
        d_partial_order
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
    if (d_stk_elem_cand) {
        cudaCheck(cudaFree(d_stk_elem_cand));
    }

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