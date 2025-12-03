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
    bool is_dag;

private:
    int vcount_;
    unsigned int ecount_;

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

    unsigned int
    ecount() const {
        return ecount_;
    }

    bool
    is_clique() const {
        return (ull)ecount_ * 2 == (ull)vcount_ *  ((ull)vcount_ - 1);
    }

    int
    degree(int u) const {
        return deg_[u];
    }

    int
    min_degree() const {
        return *min_element(deg_.begin(), deg_.end());
    }

    int
    max_degree() const {
        return *max_element(deg_.begin(), deg_.end());
    }

    Graph(const std::string &file_path, bool is_csr = false);

    Graph(const std::string &file_path, std::vector<int> &matching_order);

    void
    parse_csr(const char* filename);

    void
    parse_g2miner_format(const std::string& prefix);

    void
    remove_degree_one_layer(int x = 2);

    void 
    write_csr(const char* filename);

    void
    reorder_by_degree(bool);

    void
    convert_to_degree_dag();

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

    static void
    save_largest_component(const char* input_filename, const char* output_filename);
};


class Graph_GPU {
public:
    int vcount_;
    unsigned int ecount_;

#ifndef UNLABELED
    int *d_label_;
#endif

    unsigned *d_bknbrs_;

    int *d_degree_;

public:
    __device__ __host__ __forceinline__ int
    vcount() const {
        return vcount_;
    }

    __device__ __host__ __forceinline__ unsigned int
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

        cudaCheck(cudaMalloc(&d_degree_, sizeof(int) * vcount_));
        cudaCheck(cudaMemcpy(d_degree_, G.deg_.data(), sizeof(int) * vcount_, cudaMemcpyHostToDevice));

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

#include <vector>
#include <queue>
#include <atomic>
#include <thread>
#include <algorithm>
#include <fstream>

class ConnectedComponentFinder {
private:
    Graph* graph_;
    std::vector<int> parent_;
    std::vector<int> component_size_;
    std::vector<bool> visited_;
    std::atomic<int> current_vertex_;

public:
    ConnectedComponentFinder(Graph* graph) : graph_(graph), current_vertex_(0) {
        int vcount = graph_->vcount();
        parent_.resize(vcount);
        component_size_.resize(vcount, 0);
        visited_.resize(vcount, false);

        for (int i = 0; i < vcount; i++) {
            parent_[i] = i;
        }
    }

    // 改进的find函数，确保完全路径压缩
    int find(int x) {
        int root = x;
        // 找到根节点
        while (parent_[root] != root) {
            root = parent_[root];
        }

        // 路径压缩
        while (x != root) {
            int next = parent_[x];
            parent_[x] = root;
            x = next;
        }
        return root;
    }

    // 改进的union操作，确保正确的合并
    void union_sets(int x, int y) {
        int root_x = find(x);
        int root_y = find(y);
        if (root_x != root_y) {
            // 总是合并到较小的根，确保一致性
            if (root_x < root_y) {
                parent_[root_y] = root_x;
            } else {
                parent_[root_x] = root_y;
            }
        }
    }

    // 修正的BFS函数
    void parallel_bfs(int start_vertex) {
        if (visited_[start_vertex]) return;

        std::queue<int> q;
        q.push(start_vertex);
        visited_[start_vertex] = true;

        while (!q.empty()) {
            int u = q.front();
            q.pop();

            // 遍历邻居
            ull start = graph_->h_offset[u];
            ull end = graph_->h_offset[u + 1];

            for (ull i = start; i < end; i++) {
                int v = graph_->h_array[i];

                // 合并当前顶点和邻居
                union_sets(start_vertex, v);

                if (!visited_[v]) {
                    visited_[v] = true;
                    q.push(v);
                }
            }
        }
    }

    // 修正的组件查找函数
    void find_components(int num_threads = std::thread::hardware_concurrency()) {
        int vcount = graph_->vcount();

        // 重置原子变量
        current_vertex_.store(0);

        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; t++) {
            threads.emplace_back([this, vcount]() {
                while (true) {
                    int start = current_vertex_.fetch_add(1);
                    if (start >= vcount) break;

                    // 直接调用BFS，不进行复杂的原子检查
                    parallel_bfs(start);
                }
            });
        }

        for (auto& t : threads) {
            t.join();
        }

        // 重新计算组件大小
        std::fill(component_size_.begin(), component_size_.end(), 0);
        for (int i = 0; i < vcount; i++) {
            int root = find(i);
            component_size_[root]++;
        }
    }

    std::vector<int> get_largest_component() {
        int max_size = 0;
        int max_root = -1;
        int vcount = graph_->vcount();

        for (int i = 0; i < vcount; i++) {
            if (component_size_[i] > max_size) {
                max_size = component_size_[i];
                max_root = i;
            }
        }

        if (max_root == -1) {
            return std::vector<int>();
        }

        std::vector<int> largest_component;
        for (int i = 0; i < vcount; i++) {
            if (find(i) == max_root) {
                largest_component.push_back(i);
            }
        }

        printf("Found largest component with %zu vertices\n", largest_component.size());
        return largest_component;
    }
};

#endif