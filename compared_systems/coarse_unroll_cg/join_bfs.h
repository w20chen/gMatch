#ifndef JOIN_BFS_H
#define JOIN_BFS_H

#include "graph_gpu.h"
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

void __global__
BFS_Extend(
    const Graph_GPU Q,
    const Graph_GPU G,
    const candidate_graph_GPU cg,
    MemManager *d_MM,
    int cur_query_vertex,
    int *d_rank,
    int partial_offset,
    int last_flag
);

#endif