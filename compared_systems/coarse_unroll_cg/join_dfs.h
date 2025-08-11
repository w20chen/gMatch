#ifndef JOIN_DFS_H
#define JOIN_DFS_H

#include "graph_gpu.h"
#include "candidate.h"
#include "mem_manager.h"

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