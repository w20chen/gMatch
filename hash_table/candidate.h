#ifndef CANDIDATE_H
#define CANDIDATE_H

#include <queue>
#include <unordered_set>
#include "helper.h"
#include "graph.h"
#include "params.h"
#include "cuckoo.h"


class candidate_graph {
    std::vector<std::vector<bool>> B;

public:
    Graph &Q, &G;

    std::vector<std::vector<int>> cand;

    std::vector<int> h_cg_array;
    std::vector<std::vector<int>> h_cg_offset;

    candidate_graph(Graph &q, Graph &g, bool simple_filter);

    candidate_graph(Graph &q, Graph &g, bool simple_filter, const std::vector<int> &represent);

    ~candidate_graph() {}

    void
    refine();

    int *
    h_get_candidates(int vv, int u, int &len) const {
        len = h_cg_offset[u][vv + 1] - h_cg_offset[u][vv];
        return (int *)h_cg_array.data() + h_cg_offset[u][vv];
    }
};


class candidate_graph_GPU {
public:
    int query_vertex_cnt;
    int data_vertex_cnt;

    int *d_cg_offset;
    int *d_cg_array;

    CuckooHashGPU<IntTuple> *d_cuckoo_array;

public:
    candidate_graph_GPU(const candidate_graph &cg);

#ifdef CUCKOO_HASH
    __device__ __forceinline__ int *
    d_get_candidates(int vv, int u, int &len) const {
        IntTuple *p = d_cuckoo_array[u].find(vv);
        len = p->second;
        return d_cg_array + p->first;
    }

    __device__ __forceinline__ int
    d_get_candidates_offset(int vv, int u, int &len) const {
        IntTuple *p = d_cuckoo_array[u].find(vv);
        len = p->second;
        return p->first;
    }
#else
    __device__ __forceinline__ int *
    d_get_candidates(int vv, int u, int &len) const {
        int *p = &d_cg_offset[u * (data_vertex_cnt + 1) + vv];
        len = p[1] - p[0];
        return d_cg_array + p[0];
    }

    __device__ __forceinline__ int
    d_get_candidates_offset(int vv, int u, int &len) const {
        int *p = &d_cg_offset[u * (data_vertex_cnt + 1) + vv];
        len = p[1] - p[0];
        return p[0];
    }
#endif

    void
    deallocate() {
        // cudaCheck(cudaFree(d_cg_offset));
        // cudaCheck(cudaFree(d_cg_array));
    }

    int
    enable_cuckoo(const candidate_graph &cg) {
        cudaMalloc(&d_cuckoo_array, query_vertex_cnt * sizeof(CuckooHashGPU<IntTuple>));

        int memory_size = query_vertex_cnt * sizeof(CuckooHashGPU<IntTuple>);

        for (int u = 0; u < query_vertex_cnt; u++) {
            CuckooHash<IntTuple> C;
            for (int v = 0; v < data_vertex_cnt; v++) {
                if (cg.h_cg_offset[u][v] != -1 && cg.h_cg_offset[u][v + 1] != -1) {
                    C.insert(v, IntTuple(cg.h_cg_offset[u][v], cg.h_cg_offset[u][v + 1] - cg.h_cg_offset[u][v]));
                }
            }
            // d_cuckoo_array + u is the address of the u-th cuckoo hash table on the GPU
            memory_size += C.to_gpu(d_cuckoo_array + u);
        }
        return memory_size;
    }
};

#endif