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

    ull *__restrict__ g_nbr_offset;
    int *__restrict__ g_nbr_array;

    candidate_graph_GPU(const candidate_graph &cg);

    __device__ __forceinline__ int *get_nbr(int v, int &len) const {
        ull start = __ldg(g_nbr_offset + v);
        ull end = __ldg(g_nbr_offset + v + 1);
        len = static_cast<int>(end - start);

        return g_nbr_array + start;
    }
};

#endif