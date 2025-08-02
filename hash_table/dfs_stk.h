#ifndef DFS_STK_H
#define DFS_STK_H


// Allocated on shared memory
struct stk_elem_fixed {
    int mapped_v[UNROLL_MIN];
};


// Allocated on global memory (or shared memory)
struct stk_elem_cand {
    int cand_set[UNROLL_MAX];
    int cand_len[UNROLL_MAX];
};


// Allocated on shared memory
struct stk_elem {
    int mapped_v[UNROLL_MAX];

#ifdef STK_ELEM_CAND_ON_SHARED
    stk_elem_cand cand;
#endif

    int start_idx_within_set;
    char parent_idx[UNROLL_MAX];

    char start_set_idx;
    char unroll_size;
};

#endif