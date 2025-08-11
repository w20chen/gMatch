#ifndef CANDIDATE_H
#define CANDIDATE_H

#include <queue>
#include <unordered_set>
#include "helper.h"
#include "graph.h"

extern __constant__ int const_eid[256];

class candidate_graph {
    Graph &Q, &G;
public:
    std::vector<std::vector<int>> cand;
    int tot_cand_cnt;

    int query_vertex_cnt;
    int query_edge_cnt;
    int data_vertex_cnt;

    int *h_eid;
    std::vector<int> h_cg_array;
    int *h_cg_offset;

    candidate_graph(Graph &, Graph &, bool);

    ~candidate_graph() {
        free(h_cg_offset);
        free(h_eid);
    }

    void
    refine(bool);

    void
    refine_bfs(int u, int v, std::unordered_set<ull> &refined);
};


class candidate_graph_GPU {
public:
    int *d_cand_set;
    int *d_cand_offset;

    int query_vertex_cnt;
    int data_vertex_cnt;

    int *d_cg_offset;
    int *d_cg_array;

public:
    candidate_graph_GPU(const candidate_graph &cg);

    __device__ int *
    d_get_candidates(int u1, int u2, int v, int &len) const {
        // When u1 is mapped to v, what are the candidates of u2?
        int *start = d_cg_offset + const_eid[u1 * query_vertex_cnt + u2] * data_vertex_cnt;
        len = start[v + 1] - start[v];
        return d_cg_array + start[v];
    }

    void
    deallocate() {
        cudaCheck(cudaFree(d_cand_set));
        cudaCheck(cudaFree(d_cand_offset));
        cudaCheck(cudaFree(d_cg_offset));
        cudaCheck(cudaFree(d_cg_array));
    }
};

#endif