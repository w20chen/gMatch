#ifndef DFS_STK_H
#define DFS_STK_H


// Allocated on shared memory
struct stk_elem_fixed {
    int mapped_v[UNROLL_MIN];
};

// Allocated on shared memory
struct stk_elem {
    int *cand_set[UNROLL_MAX + 1];
    int cand_len[UNROLL_MAX + 1];

    int mapped_v[UNROLL_MAX + 1];

    int start_idx_within_set;
    char parent_idx[UNROLL_MAX + 1];

    char start_set_idx;
    char unroll_size;
};

#endif