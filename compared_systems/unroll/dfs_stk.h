#ifndef DFS_STK_H
#define DFS_STK_H

struct stk_elem_fixed {
    int mapped[UNROLL_MIN];
};

struct stk_elem {
    int *cand_set[UNROLL_MAX];
    short cand_len[UNROLL_MAX];

    unsigned warp_flag;

    char start_set_idx;
    short start_idx_within_set;

    char next_start_set_idx;
    short next_start_idx_within_set;

    char parent_idx[UNROLL_MAX];
    short mapped_idx[UNROLL_MAX];

    char unroll_size;
};

#endif