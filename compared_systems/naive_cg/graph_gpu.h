#ifndef GRAPH_GPU_H
#define GRAPH_GPU_H

#include "graph.h"
#include "helper.h"

class Graph_GPU {
public:
    int vcount_;

    int *d_adj_;
    int *d_offset_;

    int *d_bknbrs_;
    int *d_bknbrs_offset_;
    int *d_bknbrs_mask_;

    unsigned is_leaf;

    unsigned label_unique;

    int *d_same_label;

public:
    __device__ __host__ int
    vcount() const {
        return vcount_;
    }

    Graph_GPU(const Graph &G) {
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
        assert(off == G.ecount() * 2);

        cudaCheck(cudaMalloc(&d_adj_, sizeof(int) * G.ecount() * 2));
        cudaCheck(cudaMalloc(&d_offset_, sizeof(int) * (G.vcount() + 1)));
        cudaCheck(cudaMemcpy(d_adj_, h_adj_, sizeof(int) * G.ecount() * 2,
                             cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_offset_, h_offset_, sizeof(int) * (G.vcount() + 1),
                             cudaMemcpyHostToDevice));

        free(h_adj_);
        free(h_offset_);

        d_bknbrs_ = nullptr;
        d_bknbrs_offset_ = nullptr;
        label_unique = 0;
        d_same_label = nullptr;

        if (G.is_query()) {
            assert(G.bknbrs_.size() != 0);
            assert(G.bknbrs_offset_.size() != 0 &&
                   G.bknbrs_offset_.size() == G.vcount() + 1);
            cudaCheck(cudaMalloc(&d_bknbrs_, sizeof(int) * G.bknbrs_.size()));
            cudaCheck(cudaMalloc(&d_bknbrs_offset_, sizeof(int) * G.bknbrs_offset_.size()));
            cudaCheck(cudaMemcpy(d_bknbrs_, G.bknbrs_.data(),
                                 sizeof(int) * G.bknbrs_.size(), cudaMemcpyHostToDevice));
            cudaCheck(cudaMemcpy(d_bknbrs_offset_, G.bknbrs_offset_.data(),
                                 sizeof(int) * G.bknbrs_offset_.size(), cudaMemcpyHostToDevice));

            for (auto &p : G.label_vertex_mapping_) {
                auto &vec = p.second;
                if (vec.size() == 1) {
                    label_unique |= (1 << vec[0]);
                }
            }
            std::cout << "label unique: " << label_unique << std::endl;

            int *h_same_label = (int *)malloc(sizeof(int) * G.vcount());
            memset(h_same_label, -1, sizeof(int) * G.vcount());
            for (auto &p : G.label_vertex_mapping_) {
                auto &vec = p.second;
                if (vec.size() == 2) {
                    h_same_label[vec[0]] = vec[1];
                    h_same_label[vec[1]] = vec[0];
                }
            }
            cudaCheck(cudaMalloc(&d_same_label, sizeof(int) * G.vcount()));
            cudaCheck(cudaMemcpy(d_same_label, h_same_label, sizeof(int) * G.vcount(), cudaMemcpyHostToDevice));
            for (int i = 0; i < G.vcount(); i++) {
                printf("%d ", h_same_label[i]);
            }
            printf("\n");
            free(h_same_label);

            cudaCheck(cudaMalloc(&d_bknbrs_mask_, sizeof(int) * G.vcount()));
            std::vector<int> h_bknbrs_mask_(G.vcount(), 0);
            for (int u = 0; u < G.vcount(); u++) {
                for (int i = G.bknbrs_offset_[u]; i < G.bknbrs_offset_[u + 1]; i++) {
                    int uu = G.bknbrs_[i];
                    h_bknbrs_mask_[u] |= (1 << uu);
                }
                printf("%d:%p ", u, h_bknbrs_mask_[u]);
            }
            printf("\n");
            cudaCheck(cudaMemcpy(d_bknbrs_mask_, h_bknbrs_mask_.data(), sizeof(int) * G.vcount(), cudaMemcpyHostToDevice));
        }
    }

    void
    deallocate() {
        cudaCheck(cudaFree(d_adj_));
        cudaCheck(cudaFree(d_offset_));
        if (d_bknbrs_ != nullptr) {
            cudaCheck(cudaFree(d_bknbrs_));
        }
        if (d_bknbrs_offset_ != nullptr) {
            cudaCheck(cudaFree(d_bknbrs_offset_));
        }
    }

    __device__ __host__ __forceinline__ bool
    leaf(int u) const {
        return (is_leaf & (1 << u)) != 0;
    }
};

#endif