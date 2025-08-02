#ifndef CANDIDATE_H
#define CANDIDATE_H

#include <queue>
#include <unordered_set>
#include "helper.h"
#include "graph.h"
#include "params.h"


class candidate_graph {
public:
    Graph &Q, &G;

    candidate_graph(Graph &q, Graph &g) : Q(q), G(g) {}

    ~candidate_graph() {}
};


class candidate_graph_GPU {
public:
    int query_vertex_cnt;
    int data_vertex_cnt;

    unsigned long long *nbr_offset;
    int *nbr_array;

    candidate_graph_GPU(const candidate_graph &cg);

    __device__ __forceinline__ int *get_nbr(int v, int &len) const {
        len = nbr_offset[v + 1] - nbr_offset[v];
        return nbr_array + nbr_offset[v];
    }
};

#endif