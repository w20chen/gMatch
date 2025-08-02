#include <vector>
#include <set>
#include "graph.h"
#include "candidate.h"


candidate_graph::candidate_graph(Graph &q, Graph &g, bool simple_filter) : Q(q), G(g) {
    cand.resize(Q.vcount());
    for (auto &vec : cand) {
        vec.reserve(1e6);
    }

    TIME_INIT();
    TIME_START();

    for (int u = 0; u < Q.vcount(); u++) {
        int label = Q.label(u);
        for (int v : G.label_vertex_mapping_[label]) {
            bool fail = false;
            for (auto p : Q.nlf_[u]) {
                int label = p.first;
                int count = p.second;
                if (count > G.nlf_[v][label]) {
                    fail = true;
                    break;
                }
            }
            if (!fail) {
                cand[u].emplace_back(v);
            }
        }
    }

    TIME_END();

    for (int i = 0; i < cand.size(); i++) {
        std::sort(cand[i].begin(), cand[i].end());
    }

    B.resize(Q.vcount());
    for (int u = 0; u < Q.vcount(); u++) {
        B[u].resize(G.vcount(), false);
        for (int v : cand[u]) {
            B[u][v] = true;
        }
    }

    TIME_START();

    // Refine candidate sets.
    if (simple_filter == false) {
        refine();
    }

    TIME_END();
    PRINT_TOTAL_TIME("Candidate Filtering");

    for (int i = 0; i < cand.size(); i++) {
        std::cout << "|C(" << i << ")| = " << cand[i].size() << std::endl;
    }

    // Build candidate graph on the CPU.
    h_cg_array.reserve(1e6);
    h_cg_offset.resize(Q.vcount());
    for (auto &vec : h_cg_offset) {
        vec.resize(G.vcount() + 1, -1);
    }

    TIME_START();

    for (int u1 = 0; u1 < Q.vcount(); u1++) {
        std::vector<int> possible_v;
        std::vector<bool> unique_v(G.vcount(), false);
        possible_v.reserve(1e6);

        for (int u2 : Q.adj_[u1]) {
            if (u1 > u2) {
                // for each query edge (u1, u2) where u1 > u2
                for (int v : cand[u2]) {
                    if (unique_v[v] == false) {
                        possible_v.push_back(v);
                        unique_v[v] = true;
                    }
                }
            }
            else {
                break;
            }
        }

        std::sort(possible_v.begin(), possible_v.end());

        for (int v : possible_v) {
            h_cg_offset[u1][v] = h_cg_array.size();

            for (int nbr : G.adj_[v]) {
                if (B[u1][nbr]) {
                    h_cg_array.push_back(nbr);
                }
            }

            h_cg_offset[u1][v + 1] = h_cg_array.size();
        }
    }

    TIME_END();
    PRINT_LOCAL_TIME("Build Candidate Graph");
}


candidate_graph::candidate_graph(Graph &q, Graph &g, bool simple_filter, const std::vector<int> &represent) : Q(q), G(g) {
    for (int i = 0; i < represent.size(); i++) {
        printf("%d:%d\n", i, represent[i]);
    }

    cand.resize(Q.vcount());
    for (auto &vec : cand) {
        vec.reserve(1e6);
    }

    TIME_INIT();
    TIME_START();

    for (int u = 0; u < Q.vcount(); u++) {
        if (represent[u] == u) {
            int label = Q.label(u);
            for (int v : G.label_vertex_mapping_[label]) {
                bool fail = false;
                for (auto p : Q.nlf_[u]) {
                    int label = p.first;
                    int count = p.second;
                    if (count > G.nlf_[v][label]) {
                        fail = true;
                        break;
                    }
                }
                if (!fail) {
                    cand[u].emplace_back(v);
                }
            }
        }
    }

    TIME_END();

    for (int i = 0; i < cand.size(); i++) {
        std::sort(cand[i].begin(), cand[i].end());
    }

    for (int i = 0; i < cand.size(); i++) {
        if (cand[i].size() == 0) {
            cand[i] = cand[represent[i]];
        }
    }

    B.resize(Q.vcount());
    for (int u = 0; u < Q.vcount(); u++) {
        B[u].resize(G.vcount(), false);
        for (int v : cand[u]) {
            B[u][v] = true;
        }
    }

    TIME_START();

    // Refine candidate sets.
    if (simple_filter == false) {
        refine();
    }

    TIME_END();
    PRINT_TOTAL_TIME("Candidate Filtering");

    for (int i = 0; i < cand.size(); i++) {
        std::cout << "|C(" << i << ")| = " << cand[i].size() << std::endl;
    }

    // Build candidate graph on the CPU.
    h_cg_array.reserve(1e6);
    h_cg_offset.resize(Q.vcount());
    for (auto &vec : h_cg_offset) {
        vec.resize(G.vcount() + 1, -1);
    }

    TIME_START();

    for (int u1 = 0; u1 < Q.vcount(); u1++) {
        std::vector<int> possible_v;
        std::vector<bool> unique_v(G.vcount(), false);
        possible_v.reserve(1e6);

        for (int u2 : Q.adj_[u1]) {
            if (u1 > u2) {
                // for each query edge (u1, u2) where u1 > u2
                for (int v : cand[u2]) {
                    if (unique_v[v] == false) {
                        possible_v.push_back(v);
                        unique_v[v] = true;
                    }
                }
            }
            else {
                break;
            }
        }

        std::sort(possible_v.begin(), possible_v.end());

        for (int v : possible_v) {
            h_cg_offset[u1][v] = h_cg_array.size();

            for (int nbr : G.adj_[v]) {
                if (B[u1][nbr]) {
                    h_cg_array.push_back(nbr);
                }
            }

            h_cg_offset[u1][v + 1] = h_cg_array.size();
        }
    }

    TIME_END();
    PRINT_LOCAL_TIME("Build Candidate Graph");
}


void candidate_graph::refine() {
    std::vector<int> bfs_order;
    Q.generate_bfs_order(bfs_order, rand() % Q.vcount());

    for (int u : bfs_order) {
        for (int i = cand[u].size() - 1; i >= 0; i--) {
            int v = cand[u][i];
            for (int uu : Q.adj_[u]) {
                // Intersect C(uu) and N(v)
                unsigned result_length = 0;

                for (int vv : G.adj_[v]) {
                    if (B[uu][vv]) {
                        result_length = 1;
                        break;
                    }
                }

                if (result_length == 0) {
                    // Remove v from C(u)
                    auto it = std::lower_bound(cand[u].begin(), cand[u].end(), v);
                    cand[u].erase(it);
                    B[u][v] = 0;
                    break;
                }
            }
        }
    }

    for (int index = Q.vcount() - 1; index >= 0; index--) {
        int u = bfs_order[index];
        for (int i = cand[u].size() - 1; i >= 0; i--) {
            int v = cand[u][i];
            for (int uu : Q.adj_[u]) {
                // Intersect C(uu) and N(v)
                unsigned result_length = 0;

                for (int vv : G.adj_[v]) {
                    if (B[uu][vv]) {
                        result_length = 1;
                        break;
                    }
                }

                if (result_length == 0) {
                    // Remove v from C(u)
                    auto it = std::lower_bound(cand[u].begin(), cand[u].end(), v);
                    cand[u].erase(it);
                    B[u][v] = 0;
                    break;
                }
            }
        }
    }
}


candidate_graph_GPU::candidate_graph_GPU(const candidate_graph &cg) {
    query_vertex_cnt = cg.Q.vcount();
    data_vertex_cnt = cg.G.vcount();

    int size_cuckoo = 0;
    int size_d_cg_offset = 0;
    int size_d_cg_array = sizeof(int) * cg.h_cg_array.size();

#ifdef CUCKOO_HASH
    size_cuckoo = enable_cuckoo(cg);
#else
    size_d_cg_offset = sizeof(int) * query_vertex_cnt * (data_vertex_cnt + 1);
    cudaCheck(cudaMalloc(&d_cg_offset, size_d_cg_offset));
    for (int u = 0; u < query_vertex_cnt; u++) {
        cudaCheck(cudaMemcpy(d_cg_offset + u * (data_vertex_cnt + 1), cg.h_cg_offset[u].data(), 
            (data_vertex_cnt + 1) * sizeof(int), cudaMemcpyHostToDevice));
    }
#endif

    cudaCheck(cudaMalloc(&d_cg_array, size_d_cg_array));
    cudaCheck(cudaMemcpy(d_cg_array, cg.h_cg_array.data(), size_d_cg_array, cudaMemcpyHostToDevice));

    printf("+------------------------+----------------+\n");
    printf("| Array Name             |   Size (KB)    |\n");
    printf("+------------------------+----------------+\n");
    printf("| Candidate Graph Offset | %-14.2lf |\n", size_d_cg_offset / 1024.0);
    printf("| Candidate Graph Array  | %-14.2lf |\n", size_d_cg_array / 1024.0);
    printf("| Cuckoo Hash Tables     | %-14.2lf |\n", size_cuckoo / 1024.0);
    printf("| Total                  | %-14.2lf |\n", (size_d_cg_offset + size_d_cg_array + size_cuckoo) / 1024.0);
    printf("+------------------------+----------------+\n");
}
