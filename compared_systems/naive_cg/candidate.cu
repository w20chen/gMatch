#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include "graph.h"
#include "candidate.h"
#include "computesetintersection.h"

candidate_graph::candidate_graph(Graph &q, Graph &g) : Q(q), G(g) {
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
    refine();

    tot_cand_cnt = 0;
    for (int i = 0; i < cand.size(); i++) {
        tot_cand_cnt += cand[i].size();
        std::cout << "|C(" << i << ")| = " << cand[i].size() << std::endl;
    }
    std::cout << "Total num of candidates: " << tot_cand_cnt << std::endl;

    // Generate edge id for query graph.
    h_eid = (int *)malloc(sizeof(int) * query_vertex_cnt * query_vertex_cnt);
    memset(h_eid, 0, sizeof(int) * query_vertex_cnt * query_vertex_cnt);

    int cnt = 0;
    for (int u = 0; u < query_vertex_cnt; u++) {
        for (int v : Q.adj_[u]) {
            h_eid[u * query_vertex_cnt + v] = cnt++;
        }
    }
    assert(cnt == Q.ecount() * 2);

    // Build candidate graph on the CPU.
    int _offset_size = query_edge_cnt * 2 * data_vertex_cnt * sizeof(int);
    h_cg_offset = (int *)malloc(_offset_size);
    memset(h_cg_offset, -1, _offset_size);

    h_cg_array.reserve(1e6);

    int set_len_avg = 0;
    int set_len_cnt = 0;

    for (int u1 = 0; u1 < query_vertex_cnt; u1++) {
        for (int u2 : Q.adj_[u1]) {
            int row = h_eid[u1 * query_vertex_cnt + u2];
            int *start = h_cg_offset + row * data_vertex_cnt;
            for (int v : cand[u1]) {
                // when u1 is mapped to v, what are the candidates of u2?
                start[v] = h_cg_array.size();

                // intersect C(u2) and N(v)
                h_cg_array.resize(start[v] + cand[u2].size());

                unsigned result_length = 0;
                ComputeSetIntersection::ComputeCandidates(
                    cand[u2].data(), cand[u2].size(),
                    G.adj_[v].data(), G.adj_[v].size(),
                    h_cg_array.data() + start[v], result_length
                );

                h_cg_array.resize(start[v] + result_length);

                // std::sort(h_cg_array.begin() + start[v], h_cg_array.end());
                start[v + 1] = h_cg_array.size();

                if (start[v] != start[v + 1]) {
                    set_len_avg += start[v + 1] - start[v];
                    set_len_cnt += 1;
                }
            }
        }
    }

    avg_len = (float)set_len_avg / set_len_cnt;
    printf("average length of candidate set: %.2f\n", avg_len);
}


void
candidate_graph::refine() {
    TIME_INIT();
    TIME_START();
    int refine_cnt = 0;
    std::unordered_set<ull> refined;

    for (int u = 0; u < query_vertex_cnt; u++) {
        for (int i = 0; i < cand[u].size(); i++) {
            int v = cand[u][i];
            for (int uu : Q.adj_[u]) {
                // intersect C(uu) and N(v)
                unsigned result_length = 0;
                ComputeSetIntersection::ComputeCandidates(
                    cand[uu].data(), cand[uu].size(),
                    G.adj_[v].data(), G.adj_[v].size(),
                    result_length
                );
                if (result_length == 0) {
                    // remove v from C(u)
                    refine_cnt++;
                    auto it = std::find(cand[u].begin(), cand[u].end(), v);
                    cand[u].erase(it);

                    if (Q.vcount() > 6) {
                        refined.emplace(((ull)u << 32) | (ull)v);
                    }

                    i--;
                    break;
                }
            }
        }
    }

    TIME_END();
    PRINT_LOCAL_TIME("First round candidate filtering");

    if (Q.vcount() > 6) {
        TIME_START();
        int r1 = 0;
        while (refined.empty() == false) {
            auto it = refined.begin();
            ull p = *it;
            int u = (int)(p >> 32);
            int v = (int)p;
            refined.erase(it);
            refine_bfs(u, v, refined, r1);
        }
        refine_cnt += r1;
        TIME_END();
        PRINT_LOCAL_TIME("Recursive filtering");
        std::cout << "recursive refine cnt: " << r1 << std::endl;
    }

    std::cout << "total refine cnt: " << refine_cnt << std::endl;
}


void
candidate_graph::refine_bfs(int u, int v,
                            std::unordered_set<ull> &refined, int &r) {
    // (u, v) has been removed
    for (int uu : Q.adj_[u]) {
        for (int i = 0; i < cand[uu].size(); i++) {
            int vv = cand[uu][i];
            // intersect C(u) and N(vv)
            unsigned result_length = 0;
            ComputeSetIntersection::ComputeCandidates(
                cand[u].data(), cand[u].size(),
                G.adj_[vv].data(), G.adj_[vv].size(),
                result_length
            );
            if (result_length == 0) {
                r++;
                auto it = std::find(cand[uu].begin(), cand[uu].end(), vv);
                cand[uu].erase(it);
                i--;
                refined.emplace(((ull)uu << 32) | (ull)vv);
            }
        }
    }
}

void
candidate_graph::refine_2hop(int &r) {
    std::vector<std::unordered_map<int, int>> q_nlf_2hop(Q.vcount());
    std::vector<std::unordered_map<int, int>> g_nlf_2hop(G.vcount());

    for (int u = 0; u < query_vertex_cnt; u++) {
        for (int u1 : Q.adj_[u]) {
            for (int u2 : Q.adj_[u1]) {
                if (u != u2) {
                    int l1 = Q.label(u1);
                    int l2 = Q.label(u2);
                    int l = (l2 << 16) & l1;
                    q_nlf_2hop[u][l] += 1;
                }
            }
        }
    }

    for (int v = 0; v < data_vertex_cnt; v++) {
        for (int v1 : G.adj_[v]) {
            for (int v2 : G.adj_[v1]) {
                if (v != v2) {
                    int l1 = G.label(v1);
                    int l2 = G.label(v2);
                    int l = (l2 << 16) & l1;
                    g_nlf_2hop[v][l] += 1;
                }
            }
        }
    }

    for (int u = 0; u < query_vertex_cnt; u++) {
        for (int i = 0; i < cand[u].size(); i++) {
            int v = cand[u][i];
            for (auto p : q_nlf_2hop[u]) {
                int l = p.first;
                int num = p.second;
                if (num > g_nlf_2hop[v][l]) {
                    auto it = std::find(cand[u].begin(), cand[u].end(), v);
                    cand[u].erase(it);
                    r++;
                    i--;
                    break;
                }
            }
        }
    }
}

void
candidate_graph::refine_neighbor_safety(int &r) {
    for (int u = 0; u < query_vertex_cnt; u++) {
        for (int i = 0; i < cand[u].size(); i++) {
            int v = cand[u][i];
            // for each pair of u-v

            std::unordered_map<int, int> nlf_cs;
            for (int uu : Q.adj_[u]) {
                int l = Q.label(uu);
                unsigned result_length = 0;
                ComputeSetIntersection::ComputeCandidates(
                    cand[uu].data(), cand[uu].size(),
                    G.adj_[v].data(), G.adj_[v].size(),
                    result_length
                );
                if (nlf_cs.find(l) == nlf_cs.end()) {
                    nlf_cs[l] = 0;
                }
                nlf_cs[l] += result_length;
            }

            // for each label l
            for (auto p : Q.nlf_[u]) {
                int l = p.first;
                if (p.second > nlf_cs[l]) {
                    auto it = std::find(cand[u].begin(), cand[u].end(), v);
                    cand[u].erase(it);
                    r++;
                    i--;
                    break;
                }
            }
        }
    }
}