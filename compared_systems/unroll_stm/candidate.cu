#include <vector>
#include <unordered_set>
#include "graph.h"
#include "candidate.h"
#include "computesetintersection.h"

__constant__ int const_eid[256];

candidate_graph::candidate_graph(Graph &q, Graph &g, bool recursive_filtering) : Q(q), G(g) {
    tot_cand_cnt = 0;
    query_vertex_cnt = Q.vcount();
    query_edge_cnt = Q.ecount();
    data_vertex_cnt = G.vcount();
    cand.resize(query_vertex_cnt);

    for (auto &vec : cand) {
        vec.reserve(1000);
    }

    for (int u = 0; u < query_vertex_cnt; u++) {
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

    for (int i = 0; i < cand.size(); i++) {
        std::sort(cand[i].begin(), cand[i].end());
    }

    // Refine candidate sets.
    refine(recursive_filtering);

    tot_cand_cnt = 0;
    for (int i = 0; i < cand.size(); i++) {
        tot_cand_cnt += cand[i].size();
        std::cout << "|C(" << i << ")| = " << cand[i].size() << std::endl;
    }
    std::cout << "Total number of candidates: " << tot_cand_cnt << std::endl;

    // Generate edge id for query graph.
    h_eid = (int *)malloc(sizeof(int) * query_vertex_cnt * query_vertex_cnt);
    memset(h_eid, 0, sizeof(int) * query_vertex_cnt * query_vertex_cnt);

    int cnt = 0;
    for (int u = 0; u < query_vertex_cnt; u++) {
        for (int v : Q.adj_[u]) {
            h_eid[u * query_vertex_cnt + v] = cnt++;
        }
    }

    // Build candidate graph on the CPU.
    int _offset_size = query_edge_cnt * 2 * data_vertex_cnt * sizeof(int);
    h_cg_offset = (int *)malloc(_offset_size);
    memset(h_cg_offset, -1, _offset_size);

    h_cg_array.reserve(1e6);

    for (int u1 = 0; u1 < query_vertex_cnt; u1++) {
        for (int u2 : Q.adj_[u1]) {
            int row = h_eid[u1 * query_vertex_cnt + u2];
            int *start = h_cg_offset + row * data_vertex_cnt;
            for (int v : cand[u1]) {
                // When u1 is mapped to v, what are the candidates of u2?
                start[v] = h_cg_array.size();

                // Intersect C(u2) and N(v)
                h_cg_array.resize(start[v] + cand[u2].size());

                unsigned result_length = 0;
                ComputeSetIntersection::ComputeCandidates(
                    cand[u2].data(), cand[u2].size(),
                    G.adj_[v].data(), G.adj_[v].size(),
                    h_cg_array.data() + start[v], result_length
                );

                h_cg_array.resize(start[v] + result_length);

                if (result_length >= 32768) {
                    printf("error\n");
                    exit(-1);
                }

                // std::sort(h_cg_array.begin() + start[v], h_cg_array.end());
                start[v + 1] = h_cg_array.size();
            }
        }
    }
}


void
candidate_graph::refine(bool recursive_filtering) {
    TIME_INIT();
    TIME_START();
    std::unordered_set<ull> refined;

    for (int u = 0; u < query_vertex_cnt; u++) {
        for (int i = cand[u].size() - 1; i >= 0; i--) {
            int v = cand[u][i];
            for (int uu : Q.adj_[u]) {
                // Intersect C(uu) and N(v)
                unsigned result_length = 0;
                ComputeSetIntersection::ComputeCandidates(
                    cand[uu].data(), cand[uu].size(),
                    G.adj_[v].data(), G.adj_[v].size(),
                    result_length
                );
                if (result_length == 0) {
                    // Remove v from C(u)
                    auto it = std::lower_bound(cand[u].begin(), cand[u].end(), v);
                    cand[u].erase(it);

                    if (Q.vcount() > 6 && recursive_filtering) {
                        refined.emplace(((ull)u << 32) | (ull)v);
                    }
                    break;
                }
            }
        }
    }

    TIME_END();
    PRINT_LOCAL_TIME("First round candidate filtering");

    if (Q.vcount() > 6 && recursive_filtering) {
        TIME_START();
        while (refined.empty() == false) {
            auto it = refined.begin();
            ull p = *it;
            int u = (int)(p >> 32);
            int v = (int)p;
            refined.erase(it);
            refine_bfs(u, v, refined);
        }
        TIME_END();
        PRINT_LOCAL_TIME("Recursive filtering");
    }
}


void
candidate_graph::refine_bfs(int u, int v, std::unordered_set<ull> &refined) {
    // (u, v) has been removed
    for (int uu : Q.adj_[u]) {
        for (int i = cand[uu].size() - 1; i >= 0; i--) {
            int vv = cand[uu][i];
            // Intersect C(u) and N(vv)
            unsigned result_length = 0;
            ComputeSetIntersection::ComputeCandidates(
                cand[u].data(), cand[u].size(),
                G.adj_[vv].data(), G.adj_[vv].size(),
                result_length
            );
            if (result_length == 0) {
                auto it = std::lower_bound(cand[uu].begin(), cand[uu].end(), vv);
                cand[uu].erase(it);
                refined.emplace(((ull)uu << 32) | (ull)vv);
            }
        }
    }
}


candidate_graph_GPU::candidate_graph_GPU(const candidate_graph &cg) {
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

    cudaCheck(cudaMemcpyToSymbol(const_eid, cg.h_eid, cg.query_vertex_cnt * cg.query_vertex_cnt * sizeof(int)));

    int _offset_size = cg.query_edge_cnt * 2 * cg.data_vertex_cnt * sizeof(int);
    cudaCheck(cudaMalloc(&d_cg_array, sizeof(int) * cg.h_cg_array.size()));
    cudaCheck(cudaMemcpy(d_cg_array, cg.h_cg_array.data(),
                         sizeof(int) * cg.h_cg_array.size(), cudaMemcpyHostToDevice));

    cudaCheck(cudaMalloc(&d_cg_offset, _offset_size));
    cudaCheck(cudaMemcpy(d_cg_offset, cg.h_cg_offset, _offset_size,
                         cudaMemcpyHostToDevice));

    std::cout << "Total number of vertices in candidate graph on the GPU: " << cg.h_cg_array.size() << std::endl;
    free(h_cand_offset);
}
