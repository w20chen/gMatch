#ifndef JOIN_BFS_H
#define JOIN_BFS_H

#include "graph.h"
#include "helper.h"

int *match_first_edge_sym(
    Graph &q,
    Graph &g,
    Graph_GPU &Q,
    Graph_GPU &G,
    std::vector<int> &matching_order,
    int &cnt,
    std::vector<uint32_t> &partial_order
);

ull
join_bfs_dfs_sym(
    Graph &q,
    Graph &g,
    Graph_GPU Q,
    Graph_GPU G,
    std::vector<int> &matching_order,
    std::vector<uint32_t> &partial_order
);

#endif