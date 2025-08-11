#ifndef DFS_STK_H
#define DFS_STK_H

#define BUFFER_SIZE 1000

#define QVMAX 16

#define UNROLL_SIZE 8

struct stk_elem {
    int selected_vertex;                                // Current selected vertex
    int cand_set[BUFFER_SIZE][UNROLL_SIZE];             // Candidate set
    int cand_len[UNROLL_SIZE];                          // Candidate set size
    int current_idx;                                    // Current candidate index
    int uiter;
};


class dfs_stk {
public:
    stk_elem stk[QVMAX];
};


#endif