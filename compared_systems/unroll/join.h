#ifndef JOIN_BFS_H
#define JOIN_BFS_H

#include "candidate.h"
#include "mem_manager.h"

__host__ int *
set_beginning_partial_matchings(
    const Graph &Q,
    const Graph &G,
    const candidate_graph &cg,
    const std::vector<int> &matching_order,
    int &cnt
);

ull
join_bfs(
    const Graph &q,
    const Graph &g,
    const Graph_GPU &Q,
    const Graph_GPU &G,
    const candidate_graph &_cg,
    const candidate_graph_GPU &cg,
    const std::vector<int> &matching_order
);

void __global__
BFS_Extend(
    const Graph_GPU Q,
    const Graph_GPU G,
    const candidate_graph_GPU cg,
    MemManager *d_MM,
    int cur_query_vertex,
    int partial_offset,
    int last_flag,
    int *d_matching_order
);

ull
join_bfs_dfs(
    const Graph &q,
    const Graph &g,
    const Graph_GPU &Q,
    const Graph_GPU &G,
    const candidate_graph &_cg,
    const candidate_graph_GPU &cg,
    const std::vector<int> &matching_order
);

#endif