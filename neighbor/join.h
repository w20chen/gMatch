#ifndef JOIN_BFS_H
#define JOIN_BFS_H

#include "candidate.h"
#include "mem_manager.h"

class cudaMutex {
    int* lock_;
public:

    __host__ cudaMutex() {
        cudaMalloc(&lock_, sizeof(int));
        cudaMemset(lock_, 0, sizeof(int));
    }

    __host__ ~cudaMutex() {
        cudaFree(lock_);
    }
    
    __device__ void lock() {
        while (atomicCAS(lock_, 0, 1) != 0) {
            __nanosleep(100);
        }
        __threadfence();
    }
    
    __device__ void unlock() {
        __threadfence();
        atomicExch(lock_, 0);
    }
};

__host__ int *
set_beginning_partial_matchings(
    Graph &Q,
    Graph &G,
    candidate_graph_GPU &cg,
    int &cnt
);

__host__ int *
set_beginning_partial_matchings_sym(
    Graph &Q,
    Graph &G,
    candidate_graph_GPU &cg,
    int &cnt,
    std::vector<uint32_t> &partial_order
);

void __global__
BFS_Extend(
    Graph_GPU Q,
    Graph_GPU G,
    candidate_graph_GPU cg,
    MemManager *d_MM,
    int cur_query_vertex,
    int partial_offset,
    int last_flag,
    int *d_error_flag
);

void __global__
BFS_Extend_sym(
    Graph_GPU Q,
    Graph_GPU G,
    candidate_graph_GPU cg,
    MemManager *d_MM,
    int cur_query_vertex,
    int partial_offset,
    int last_flag,
    int *d_error_flag,
    int *d_partial_order
);

ull
join_bfs_dfs(
    Graph &q,
    Graph &g,
    Graph_GPU &Q,
    Graph_GPU &G,
    candidate_graph &_cg,
    candidate_graph_GPU &cg,
    bool no_memory_pool
);

ull
join_bfs_dfs_sym(
    Graph &q,
    Graph &g,
    Graph_GPU &Q,
    Graph_GPU &G,
    candidate_graph &_cg,
    candidate_graph_GPU &cg,
    std::vector<uint32_t> &partial_order,
    bool no_memory_pool
);

ull
join_no_filtering(
    Graph &q,
    Graph &g,
    Graph_GPU &Q,
    Graph_GPU &G,
    candidate_graph &_cg,
    candidate_graph_GPU &cg,
    std::vector<uint32_t> &partial_order
);

ull
join_induced(
    Graph &q,
    Graph &g,
    Graph_GPU &Q,
    Graph_GPU &G,
    candidate_graph &_cg,
    candidate_graph_GPU &cg,
    std::vector<uint32_t> &partial_order
);

ull
join_induced_orientation(
    Graph &q,
    Graph &g,
    Graph_GPU &Q,
    Graph_GPU &G,
    candidate_graph &_cg,
    candidate_graph_GPU &cg
);

#endif