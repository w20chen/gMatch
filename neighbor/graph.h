#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <string>
#include <algorithm>
#include <queue>
#include "helper.h"
#include "params.h"

class Graph {
public:
    ull *h_offset;
    int *h_array;

private:
    int vcount_;
    int ecount_;

    std::vector<int> deg_;

    std::vector<int> vertex_label_;

    std::vector<unsigned> bknbrs_;

public:

    friend class Graph_GPU;

    std::vector<std::vector<int>> adj_;

#ifndef UNLABELED
#ifdef NLF_FILTER
    std::vector<std::unordered_map<int, int>> nlf_;
#endif
#endif

    int
    label(int u) const {
#ifndef UNLABELED
        return vertex_label_[u];
#else
        return 0;
#endif
    }

    int
    vcount() const {
        return vcount_;
    }

    int
    ecount() const {
        return ecount_;
    }

    bool
    is_clique() const {
        return ecount_ * 2 == vcount_ *  (vcount_ - 1);
    }

    int
    degree(int u) const {
        return deg_[u];
    }

    Graph(const std::string &file_path, bool is_csr = false);

    Graph(const std::string &file_path, std::vector<int> &matching_order);

    void
    parse_csr(const char* filename);

    void
    print_meta() const {
#ifndef UNLABELED
        std::cout << "|V|=" << vcount_ << " |E|=" << ecount_ << " |\u03A3|=" <<
                 *std::max_element(vertex_label_.begin(), vertex_label_.end()) + 1 << std::endl;
#else
        std::cout << "|V|=" << vcount_ << " |E|=" << ecount_ << " Unlabeled" << std::endl;
#endif
        std::cout << "Max Degree=" << *max_element(deg_.begin(), deg_.end()) << std::endl;
    }

    bool
    is_adjacent(int v1, int v2) const {
        if (adj_[v1].size() < adj_[v2].size()) {
            auto it = std::lower_bound(adj_[v1].begin(), adj_[v1].end(), v2);
            if (it == adj_[v1].end()) {
                return false;
            }
            else {
                return *it == v2;
            }
        }

        auto it = std::lower_bound(adj_[v2].begin(), adj_[v2].end(), v1);
        if (it == adj_[v2].end()) {
            return false;
        }
        else {
            return *it == v1;
        }
    }

    void
    generate_matching_order(std::vector<int> &matching_order) const;

    void
    generate_bfs_order(std::vector<int> &bfs_order, int start = 0) const;

private:

    void
    generate_backward_neighborhood() {
        // Only query graph can call this function.
        assert(vcount_ <= 32);
        bknbrs_.resize(vcount_);

        for (int u = 0; u < vcount_; u++) {
            unsigned mask = 0;
            for (int uu = 0; uu < u; uu++) {
                if (is_adjacent(u, uu)) {
                    mask |= (1u << uu);
                }
            }
            bknbrs_[u] = mask;
        }
    }

private:
    int
    find_automorphisms(std::vector<std::vector<uint32_t>> &embeddings) const;

public:
    int
    restriction_generation(std::vector<uint32_t> &partial_order) const;
};


class Graph_GPU {
public:
    int vcount_;
    int ecount_;

#ifndef UNLABELED
    int *d_label_;
#endif

    unsigned *d_bknbrs_;

// #ifdef DEGREE_FILTER_ON_THE_FLY
    int *d_degree_;
// #endif

public:
    __device__ __host__ __forceinline__ int
    vcount() const {
        return vcount_;
    }

    __device__ __host__ __forceinline__ int
    ecount() const {
        return ecount_;
    }

    Graph_GPU(const Graph &G, bool is_query) {
        this->vcount_ = G.vcount_;
        this->ecount_ = G.ecount_;

#ifndef UNLABELED
        cudaCheck(cudaMalloc(&d_label_, sizeof(int) * G.vcount_));
        printf("--- Allocating %d bytes (%d ints) for d_label_ @ %p\n", sizeof(int) * G.vcount_, G.vcount_, d_label_);
        cudaCheck(cudaMemcpy(d_label_, G.vertex_label_.data(), sizeof(int) * G.vcount_, cudaMemcpyHostToDevice));
#endif

        d_bknbrs_ = nullptr;

// #ifdef DEGREE_FILTER_ON_THE_FLY
        cudaCheck(cudaMalloc(&d_degree_, sizeof(int) * vcount_));
        cudaCheck(cudaMemcpy(d_degree_, G.deg_.data(), sizeof(int) * vcount_, cudaMemcpyHostToDevice));
// #endif

        if (is_query) {
            // Create index for backward neighbors.
            cudaCheck(cudaMalloc(&d_bknbrs_, sizeof(unsigned) * G.bknbrs_.size()));
            printf("--- Allocating %d bytes (%d unsigneds) for d_bknbrs_ @ %p\n", sizeof(unsigned) * G.bknbrs_.size(), G.bknbrs_.size(), d_bknbrs_);
            cudaCheck(cudaMemcpy(d_bknbrs_, G.bknbrs_.data(), 
                                 sizeof(unsigned) * G.bknbrs_.size(), cudaMemcpyHostToDevice));
        }
    }

    void __host__
    deallocate() {
        if (d_bknbrs_ != nullptr) {
            cudaCheck(cudaFree(d_bknbrs_));
        }
    }
};


#endif