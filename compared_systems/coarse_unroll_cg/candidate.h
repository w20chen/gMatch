#ifndef CANDIDATE_H
#define CANDIDATE_H


#include <queue>
#include <unordered_set>
#include "graph.h"
#include "helper.h"
#include "graph.h"


class Graph;


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

    float avg_len;

    candidate_graph(Graph &, Graph &);

    ~candidate_graph() {
        free(h_cg_offset);
        free(h_eid);
    }

    void
    refine();

    void
    refine_bfs(int u, int v, std::unordered_set<ull> &refined, int &r);

    void
    refine_neighbor_safety(int &r);

    void
    refine_2hop(int &r);
};


class candidate_graph_GPU {
public:
    int *d_cand_set;
    int *d_cand_offset;

    int query_vertex_cnt;
    int data_vertex_cnt;

    int *d_eid;
    int *d_cg_offset;
    int *d_cg_array;

public:
    candidate_graph_GPU(const candidate_graph &cg) {
        int V = cg.query_vertex_cnt;
        this->query_vertex_cnt = cg.query_vertex_cnt;
        this->data_vertex_cnt = cg.data_vertex_cnt;

        int *h_cand_offset = (int *)malloc(sizeof(int) * (V + 1));
        h_cand_offset[0] = 0;
        h_cand_offset[V] = cg.tot_cand_cnt;
        for (int i = 1; i < V; i++) {
            h_cand_offset[i] = h_cand_offset[i - 1] + cg.cand[i - 1].size();
        }
        cudaCheck(cudaMalloc(&d_cand_offset, (V + 1) * sizeof(int)));
        cudaCheck(cudaMemcpy(d_cand_offset, h_cand_offset, (V + 1) * sizeof(int),
                             cudaMemcpyHostToDevice));

        cudaCheck(cudaMalloc(&d_cand_set, cg.tot_cand_cnt * sizeof(int)));
        for (int i = 0; i < V; i++) {
            cudaCheck(cudaMemcpy(d_cand_set + h_cand_offset[i], cg.cand[i].data(),
                                 sizeof(int) * (h_cand_offset[i + 1] - h_cand_offset[i]),
                                 cudaMemcpyHostToDevice));
        }
        printf("Candidate set moved to GPU.\n");

        cudaCheck(cudaMalloc(&d_eid,
                             cg.query_vertex_cnt * cg.query_vertex_cnt * sizeof(int)));
        cudaCheck(cudaMemcpy(d_eid, cg.h_eid,
                             cg.query_vertex_cnt * cg.query_vertex_cnt * sizeof(int),
                             cudaMemcpyHostToDevice));

        int _offset_size = cg.query_edge_cnt * 2 * cg.data_vertex_cnt * sizeof(int);
        cudaCheck(cudaMalloc(&d_cg_array, sizeof(int) * cg.h_cg_array.size()));
        cudaCheck(cudaMemcpy(d_cg_array, cg.h_cg_array.data(),
                             sizeof(int) * cg.h_cg_array.size(), cudaMemcpyHostToDevice));

        cudaCheck(cudaMalloc(&d_cg_offset, _offset_size));
        cudaCheck(cudaMemcpy(d_cg_offset, cg.h_cg_offset, _offset_size,
                             cudaMemcpyHostToDevice));

        std::cout << "Total num of vertices in candidate_graph: " <<
                  cg.h_cg_array.size() << std::endl;
        free(h_cand_offset);
    }

    __device__ int *
    d_get_candidates(int u1, int u2, int v, int &len) const {
        // when u1 is mapped to v, what are the candidates of u2 ?
        int *start = d_cg_offset + d_eid[u1 * query_vertex_cnt + u2] *
                     this->data_vertex_cnt;
        len = start[v + 1] - start[v];
        assert(v >= 0);
        assert(start[v] >= 0);
        assert(start[v + 1] >= 0);
        assert(len >= 0);
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