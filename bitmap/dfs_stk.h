#ifndef DFS_STK_H
#define DFS_STK_H


// Allocated on shared memory
struct stk_elem_fixed {
    int mapped_v[UNROLL_MIN];
    CandLen_t mapped_idx[UNROLL_MIN];
};


// Allocated on global memory (or shared memory)
struct stk_elem_cand {
    int cand_set[UNROLL_MAX + 1];
    CandLen_t cand_len[UNROLL_MAX + 1];
};


// Allocated on shared memory
struct stk_elem {
    int mapped_v[UNROLL_MAX + 1];

#ifdef STK_ELEM_CAND_ON_SHARED
    stk_elem_cand cand;
#endif

    CandLen_t mapped_idx[UNROLL_MAX + 1];
    CandLen_t start_idx_within_set;
    char parent_idx[UNROLL_MAX + 1];

#ifdef BITMAP_SET_INTERSECTION
    char cand_set_u[UNROLL_MAX + 1];
#endif

    char start_set_idx;
};

#endif