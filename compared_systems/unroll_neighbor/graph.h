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
#include "helper.h"

class Graph {
private:
    bool is_query_;

    int vcount_;
    int ecount_;

    ull *h_offset;
    int *h_array;

    std::vector<int> deg_;
    std::vector<int> vertex_label_;

    std::vector<int> bknbrs_;
    std::vector<int> bknbrs_offset_;

public:

    friend class Graph_GPU;

    std::vector<std::vector<int>> adj_;

    bool
    is_query() const {
        return is_query_;
    }

    int
    max_label() const {
        return *std::max_element(vertex_label_.begin(), vertex_label_.end());
    }

    int
    label(int u) const {
        return vertex_label_[u];
    }

    int
    vcount() const {
        return vcount_;
    }

    int
    ecount() const {
        return ecount_;
    }

    int
    degree(int u) const {
        return deg_[u];
    }

    Graph(const std::string &file_path, bool is_q) {
        is_query_ = is_q;
        h_array = nullptr;
        h_offset = nullptr;

        if (is_q == false) {
            parseCSR(file_path.c_str());
            return;
        }

        std::ifstream infile(file_path);
        if (!infile.is_open()) {
            std::cout << "Cannot open graph file " << file_path << "." << std::endl;
            exit(-1);
        }

        char type = 0;
        infile >> type >> vcount_ >> ecount_;

        adj_.resize(vcount_);
        deg_.resize(vcount_);
        vertex_label_.resize(vcount_);

        while (infile >> type) {
            if (type == 'v') {
                int vid, label, deg;
                infile >> vid >> label >> deg;
                vertex_label_[vid] = label;
                deg_[vid] = deg;
            }
            else {
                break;
            }
        }

        if (type == 'e') {
            std::string next_str;
            while (true) {
                int v1, v2;
                infile >> v1 >> v2;

                adj_[v1].emplace_back(v2);
                adj_[v2].emplace_back(v1);

                if (!(infile >> next_str)) {
                    break;
                }
                if (next_str == "e") {
                    continue;
                }
                else if (!(infile >> next_str)) {
                    break;
                }
            }
        }

        infile.close();

        for (int v = 0; v < vcount_; v++) {
            if (adj_[v].size() != deg_[v]) {
                std::cout << "Degree Error: ";
                std::cout << v << ", " << adj_[v].size() << " != " << deg_[v] << std::endl;
            }
        }

        for (auto &l : adj_) {
            std::sort(l.begin(), l.end());
        }

        std::cout << "Query graph loaded from file " << file_path << "." << std::endl;

        print_meta();
    }

    void
    print_meta() const {
        std::cout << "|V|=" << vcount_ << " |E|=" << ecount_ << " |\u03A3|=" << 
            *max_element(vertex_label_.begin(), vertex_label_.end()) << std::endl;
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
    generate_backward_neighborhood(const std::vector<int> &matching_order) {
        // Only query graph can call this function.
        // "bknbrs_" stores the indices of backward neighbors in the matching order.

        bknbrs_offset_.resize(vcount_ + 1);
        for (int u = 0; u < vcount_; u++) {
            bknbrs_offset_[u] = bknbrs_.size();
            for (int i = 0; i < matching_order.size(); i++) {
                int uu = (int)matching_order[i];
                if (uu == u) {
                    break;
                }
                else if (is_adjacent(u, uu)) {
                    // Push back the index of uu.
                    // Backward neighbor indices will be in ascending order.
                    bknbrs_.push_back(i);
                }
            }
        }
        bknbrs_offset_[vcount_] = bknbrs_.size();
    }

    void
    generate_matching_order(std::vector<int> &matching_order) const;

    void
    parseCSR(const char* filename);

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

    int *d_adj_;
    ull *d_offset_;

    int *d_bknbrs_;
    int *d_bknbrs_offset_;

    int *d_label_;

public:
    __device__ __host__ __forceinline__ int
    vcount() const {
        return vcount_;
    }

    __device__ __host__ __forceinline__ int *
    d_get_nbrs(int v, int &len) const {
        len = (int)(d_offset_[v + 1] - d_offset_[v]);
        return d_adj_ + d_offset_[v];
    }

    __device__ __host__ __forceinline__ int
    label(int v) const {
        return d_label_[v];
    }

    Graph_GPU(const Graph &G, bool is_q) {
        d_bknbrs_ = nullptr;
        d_bknbrs_offset_ = nullptr;

        if (is_q == false) {
            load_data(G);
            return;
        }

        this->ecount_ = G.ecount_; 
        this->vcount_ = G.vcount_;

        int *h_adj_ = (int *)malloc(sizeof(int) * G.ecount() * 2);
        int *h_offset_ = (int *)malloc(sizeof(int) * (G.vcount() + 1));
        int off = 0;
        for (int i = 0; i < G.vcount(); i++) {
            h_offset_[i] = off;
            for (int j : G.adj_[i]) {
                h_adj_[off++] = j;
            }
        }
        h_offset_[G.vcount()] = G.ecount() * 2;

        cudaCheck(cudaMalloc(&d_adj_, sizeof(int) * G.ecount() * 2));
        cudaCheck(cudaMalloc(&d_offset_, sizeof(ull) * (G.vcount() + 1)));
        cudaCheck(cudaMemcpy(d_adj_, h_adj_, sizeof(int) * G.ecount() * 2, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_offset_, h_offset_, sizeof(ull) * (G.vcount() + 1), cudaMemcpyHostToDevice));

        cudaCheck(cudaMalloc(&d_label_, sizeof(int) * G.vcount()));
        cudaCheck(cudaMemcpy(d_label_, G.vertex_label_.data(), sizeof(int) * G.vcount(), cudaMemcpyHostToDevice));

        free(h_adj_);
        free(h_offset_);

        cudaCheck(cudaMalloc(&d_bknbrs_, sizeof(int) * G.bknbrs_.size()));
        cudaCheck(cudaMalloc(&d_bknbrs_offset_, sizeof(int) * G.bknbrs_offset_.size()));
        cudaCheck(cudaMemcpy(d_bknbrs_, G.bknbrs_.data(), sizeof(int) * G.bknbrs_.size(), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_bknbrs_offset_, G.bknbrs_offset_.data(), sizeof(int) * G.bknbrs_offset_.size(), cudaMemcpyHostToDevice));
    }

    void load_data(const Graph &G) {
        vcount_ = G.vcount_;
        ecount_ = G.ecount_;
        cudaCheck(cudaMalloc(&d_adj_, sizeof(int) * 2 * (ull)ecount_));
        cudaCheck(cudaMemcpy(d_adj_, G.h_array, sizeof(int) * 2 * (ull)ecount_, cudaMemcpyHostToDevice));
        cudaCheck(cudaMalloc(&d_offset_, sizeof(ull) * (vcount_ + 1)));
        cudaCheck(cudaMemcpy(d_offset_, G.h_offset, sizeof(ull) * (vcount_ + 1), cudaMemcpyHostToDevice));
        cudaCheck(cudaMalloc(&d_label_, sizeof(int) * vcount_));
        cudaCheck(cudaMemcpy(d_label_, G.vertex_label_.data(), sizeof(int) * vcount_, cudaMemcpyHostToDevice));

        printf("Data graph loaded to device\n");
    }
};


#endif