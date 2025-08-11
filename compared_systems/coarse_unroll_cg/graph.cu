#include "graph.h"

void
Graph::generate_matching_order(std::vector<int> &matching_order) const {
    assert(is_query_);
    int n = vcount_;
    std::vector<bool> visited(n, false);

    int selected_vertex = 0;
    int selected_vertex_selectivity = deg_[selected_vertex];

    for (int u = 1; u < n; ++u) {
        int u_selectivity = deg_[u];
        if (u_selectivity > selected_vertex_selectivity) {
            selected_vertex = u;
            selected_vertex_selectivity = u_selectivity;
        }
    }

    matching_order.push_back(selected_vertex);
    visited[selected_vertex] = true;

    std::vector<int> tie_vertices;
    std::vector<int> temp;

    for (int _i = 1; _i < n; ++_i) {
        selected_vertex_selectivity = 0;
        for (int u = 0; u < n; ++u) {
            if (!visited[u]) {
                int u_selectivity = 0;
                for (auto uu : matching_order) {
                    if (is_adjacent(u, uu)) {
                        u_selectivity += 1;
                    }
                }
                if (u_selectivity > selected_vertex_selectivity) {
                    selected_vertex_selectivity = u_selectivity;
                    tie_vertices.clear();
                    tie_vertices.push_back(u);
                }
                else if (u_selectivity == selected_vertex_selectivity) {
                    tie_vertices.push_back(u);
                }
            }
        }

        if (tie_vertices.size() != 1) {
            temp.swap(tie_vertices);
            tie_vertices.clear();

            int count = 0;
            std::vector<int> u_fn;
            for (auto u : temp) {
                for (auto uu : adj_[u]) {
                    if (!visited[uu]) {
                        u_fn.push_back(uu);
                    }
                }

                int cur_count = 0;
                for (auto uu : matching_order) {
                    auto &uun = adj_[uu];
                    std::vector<int> uun_tmp;
                    uun_tmp.insert(uun_tmp.end(), uun.begin(), uun.end());

                    int common_neighbor_count = 0;
                    for (int ii : uun_tmp) {
                        for (int jj : u_fn) {
                            if (ii == jj) {
                                common_neighbor_count++;
                                break;
                            }
                        }
                        if (common_neighbor_count != 0) {
                            break;
                        }
                    }

                    if (common_neighbor_count > 0) {
                        cur_count += 1;
                    }
                }

                u_fn.clear();

                if (cur_count > count) {
                    count = cur_count;
                    tie_vertices.clear();
                    tie_vertices.push_back(u);
                }
                else if (cur_count == count) {
                    tie_vertices.push_back(u);
                }
            }
        }

        if (tie_vertices.size() != 1) {
            temp.swap(tie_vertices);
            tie_vertices.clear();

            int count = 0;
            std::vector<int> u_fn;
            for (auto u : temp) {
                for (auto uu : adj_[u]) {
                    if (!visited[uu]) {
                        u_fn.push_back(uu);
                    }
                }

                int cur_count = 0;
                for (auto uu : u_fn) {
                    bool valid = true;
                    for (auto uuu : matching_order) {
                        if (is_adjacent(uu, uuu)) {
                            valid = false;
                            break;
                        }
                    }
                    if (valid) {
                        cur_count += 1;
                    }
                }

                u_fn.clear();

                if (cur_count > count) {
                    count = cur_count;
                    tie_vertices.clear();
                    tie_vertices.push_back(u);
                }
                else if (cur_count == count) {
                    tie_vertices.push_back(u);
                }
            }
        }

        matching_order.push_back(tie_vertices[0]);
        visited[tie_vertices[0]] = true;
        tie_vertices.clear();
        temp.clear();
    }

    std::cout << "matching order: ";
    for (auto v : matching_order) {
        std::cout << v << " ";
    }
    std::cout << std::endl;
}


void
Graph::generate_matching_order(std::vector<int> &matching_order,
                               candidate_graph &CG) const {

    // TODO: make sure that leaf vertices are at the end of matching order
    // TODO: generate a better data-dependent matching order

    float mn = CG.cand[0].size() / (float)deg_[0];
    int root = 0;

    for (int u = 1; u < vcount_; u++) {
        int tmp = CG.cand[u].size() / (float)deg_[u];
        if (mn > tmp) {
            root = u;
            mn = tmp;
        }
    }

    matching_order.emplace_back(root);
    std::vector<int> visited(vcount_, 0);
    visited[root] = 1;

    while (matching_order.size() < vcount_) {
        int u = matching_order.back();
        int mn = 0x7fffffff;
        int index = -1;
        for (int uu : adj_[u]) {
            if (visited[uu] == 0 && mn > CG.cand[uu].size()) {
                mn = CG.cand[uu].size();
                index = uu;
            }
        }

        if (index == -1) {
            for (int uu = 0; uu < vcount_; uu++) {
                if (visited[uu] == 0 && mn > CG.cand[uu].size()) {
                    mn = CG.cand[uu].size();
                    index = uu;
                }
            }
        }
        matching_order.emplace_back(index);
        visited[index] = 1;
    }

    std::cout << "matching order: ";
    for (auto v : matching_order) {
        std::cout << v << " ";
    }
    std::cout << std::endl;
}