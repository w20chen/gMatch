#include <vector>
#include <set>
#include "graph.h"
#include "candidate.h"

candidate_graph_GPU::candidate_graph_GPU(const candidate_graph &cg) {
    ull *nbr_offset = nullptr;
    int *nbr_array = nullptr;

    query_vertex_cnt = cg.Q.vcount();
    data_vertex_cnt = cg.G.vcount();

    if (cg.G.h_offset != nullptr && cg.G.h_array != nullptr) {
        // Set the correct length for the CSR array
        // 注意：使用 (ull) 强制转换避免 unsigned int 溢出
        // 当 ecount_ 很大时（如 2,251,136,188），2 * ecount_ 会超过 unsigned int 的最大值
        ull h_array_size = cg.G.is_dag ? (ull)cg.G.ecount() : 2 * (ull)cg.G.ecount();

        cudaCheck(cudaMalloc(&nbr_offset, (data_vertex_cnt + 1) * sizeof(ull)));
        printf("--- Allocating %llu bytes (%lu unsigned long longs) for nbr_offset @ %p\n", (data_vertex_cnt + 1) * sizeof(ull), data_vertex_cnt + 1, nbr_offset);
        cudaCheck(cudaMalloc(&nbr_array, h_array_size * sizeof(int)));
        printf("--- Allocating %llu bytes (%llu ints) for nbr_array @ %p\n", h_array_size * sizeof(int), h_array_size, nbr_array);

        cudaCheck(cudaMemcpy(nbr_offset, cg.G.h_offset, (data_vertex_cnt + 1) * sizeof(ull), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(nbr_array, cg.G.h_array, h_array_size * sizeof(int), cudaMemcpyHostToDevice));
        printf("CSR edges copied to device!\n");

        // allocated in Graph::parse_csr
        delete[] cg.G.h_offset;
        delete[] cg.G.h_array;
        cg.G.h_offset = nullptr;
        cg.G.h_array = nullptr;

        this->g_nbr_offset = nbr_offset;
        this->g_nbr_array = nbr_array;
        return;
    }

    cudaCheck(cudaMalloc(&nbr_offset, (data_vertex_cnt + 1) * sizeof(ull)));
    printf("--- Allocating %llu bytes (%lu unsigned long longs) for nbr_offset @ %p\n", (data_vertex_cnt + 1) * sizeof(ull), data_vertex_cnt + 1, nbr_offset);
    ull *h_nbr_offset = (ull *)malloc((data_vertex_cnt + 1) * sizeof(ull));
    if (h_nbr_offset == NULL) {
        printf("Failed to allocate memory for h_nbr_offset\n");
        exit(1);
    }

    h_nbr_offset[0] = 0;
    for (int v = 1; v <= data_vertex_cnt; v++) {
        h_nbr_offset[v] = h_nbr_offset[v - 1] + cg.G.adj_[v - 1].size();
    }
    cudaCheck(cudaMemcpy(nbr_offset, h_nbr_offset, (data_vertex_cnt + 1) * sizeof(ull), cudaMemcpyHostToDevice));

    const ull total_nbrs = h_nbr_offset[data_vertex_cnt];

    std::vector<int> h_nbr_array(total_nbrs);

    for (int v = 0; v < data_vertex_cnt; v++) {
        const auto &adj_list = cg.G.adj_[v];
        const ull offset = h_nbr_offset[v];
        std::copy(adj_list.begin(), adj_list.end(), h_nbr_array.begin() + offset);
    }

    cudaCheck(cudaMalloc(&nbr_array, total_nbrs * sizeof(int)));
    printf("--- Allocating %llu bytes (%llu ints) for nbr_array @ %p\n", total_nbrs * sizeof(int), total_nbrs, nbr_array);

    cudaCheck(cudaMemcpy(nbr_array, h_nbr_array.data(), total_nbrs * sizeof(int), cudaMemcpyHostToDevice));

    free(h_nbr_offset);

    this->g_nbr_offset = nbr_offset;
    this->g_nbr_array = nbr_array;
}
