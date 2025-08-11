#ifndef DFS_STK_H
#define DFS_STK_H

#define BUFFER_SIZE 1000

#define QVMAX 16

struct stk_elem {
    int selected_vertex;                // Current selected vertex
    int cand_set[BUFFER_SIZE];          // Candidate set
    int cand_len;                       // Candidate set size
    int current_idx;                    // Current candidate index
};


class dfs_stk {
public:
    stk_elem stk[QVMAX];
};


#endif