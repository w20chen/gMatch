#include "join.h"
#include "params.h"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>


__global__ void count_matches_kernel_sym(
    const int* d_cand_0, int cand_0_size,
    const ull* d_adj_index, const int* d_adj_array,
    const char* d_cand_1, int* d_counts,
    unsigned* partial_order
) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (partial_order[0] & (1 << 1)) {
        for (int i = thread_idx; i < cand_0_size; i += blockDim.x * gridDim.x) {
            int v0 = d_cand_0[i];
            long long start = d_adj_index[v0];
            long long end = d_adj_index[v0 + 1];
            int count = 0;

            // for each neighbor v1 of v0
            for (long long j = end - 1; j >= start; j--) {
                int v1 = d_adj_array[j];
                // condition of symmetry breaking
                if (v1 <= v0) {
                    break;
                }
                if (d_cand_1[v1]) {
                    count++;
                }
            }
            d_counts[i] = count;
        }
    }
    else {
        for (int i = thread_idx; i < cand_0_size; i += blockDim.x * gridDim.x) {
            int v0 = d_cand_0[i];
            long long start = d_adj_index[v0];
            long long end = d_adj_index[v0 + 1];
            int count = 0;

            for (long long j = start; j < end; j++) {
                int v1 = d_adj_array[j];
                if (d_cand_1[v1]) {
                    count++;
                }
            }
            d_counts[i] = count;
        }
    }
}

__global__ void fill_matches_kernel_sym(
    const int* d_cand_0, int cand_0_size,
    const ull* d_adj_index, const int* d_adj_array,
    const char* d_cand_1, const int* d_offsets,
    int2* d_output,
    unsigned* partial_order
) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (partial_order[0] & (1 << 1)) {
        for (int i = thread_idx; i < cand_0_size; i += blockDim.x * gridDim.x) {
            int v0 = d_cand_0[i];
            long long start = d_adj_index[v0];
            long long end = d_adj_index[v0 + 1];
            int pos = d_offsets[i];

            for (long long j = end - 1; j >= start; j--) {
                int v1 = d_adj_array[j];
                if (v1 <= v0) {
                    break;
                }
                if (d_cand_1[v1]) {
                    d_output[pos] = make_int2(v0, v1);
                    pos++;
                }
            }
        }
    }
    else {
        for (int i = thread_idx; i < cand_0_size; i += blockDim.x * gridDim.x) {
            int v0 = d_cand_0[i];
            long long start = d_adj_index[v0];
            long long end = d_adj_index[v0 + 1];
            int pos = d_offsets[i];

            for (long long j = start; j < end; j++) {
                int v1 = d_adj_array[j];

                if (d_cand_1[v1]) {
                    d_output[pos] = make_int2(v0, v1);
                    pos++;
                }
            }
        }
    }
}


int *match_first_edge_sym(
    Graph &q,
    Graph &g,
    Graph_GPU &Q,
    Graph_GPU &G,
    std::vector<int> &matching_order,
    int &cnt,
    std::vector<uint32_t> &partial_order
) {

    std::vector<int> cand_0;
    std::vector<char> cand_1(g.vcount(), 0);

    int u = matching_order[0];
    int label = q.label(u);
    int degree = q.degree(u);

    for (int v = 0; v < g.vcount(); v++) {
        if (g.label(v) == label) {
            if (degree <= g.degree(v)) {
                cand_0.emplace_back(v);
            }
        }
    }

    u = matching_order[1];
    label = q.label(u);
    degree = q.degree(u);

    for (int v = 0; v < g.vcount(); v++) {
        if (g.label(v) == label) {
            if (degree <= g.degree(v)) {
                cand_1[v] = true;
            }
        }
    }

    int *d_cand_0 = nullptr;
    char *d_cand_1 = nullptr;

    int2 *d_output = nullptr;
    int *d_counts = nullptr;

    int cand_0_size = cand_0.size();

    cudaCheck(cudaMalloc(&d_cand_0, cand_0_size * sizeof(int)));
    cudaCheck(cudaMemcpy(d_cand_0, cand_0.data(), cand_0_size * sizeof(int), cudaMemcpyHostToDevice));

    cudaCheck(cudaMalloc(&d_cand_1, cand_1.size() * sizeof(char)));
    cudaCheck(cudaMemcpy(d_cand_1, cand_1.data(), cand_1.size() * sizeof(char), cudaMemcpyHostToDevice));

    cudaCheck(cudaMalloc(&d_counts, cand_0_size * sizeof(int)));

    const int block_size = 256;
    int grid_size = (cand_0_size + block_size - 1) / block_size;

    unsigned *d_partial_order = nullptr;
    cudaCheck(cudaMalloc(&d_partial_order, partial_order.size() * sizeof(unsigned)));
    cudaCheck(cudaMemcpy(d_partial_order, partial_order.data(), partial_order.size() * sizeof(unsigned), cudaMemcpyHostToDevice));

    count_matches_kernel_sym<<<grid_size, block_size>>>(
        d_cand_0, cand_0_size, G.d_offset_, G.d_adj_, d_cand_1, d_counts, d_partial_order
    );
    cudaCheck(cudaDeviceSynchronize());

    int total_matches = 0;
    cudaCheck(cudaMemcpy(&total_matches, d_counts + cand_0_size - 1, sizeof(int), cudaMemcpyDeviceToHost));

    thrust::exclusive_scan(thrust::device, d_counts, d_counts + cand_0_size, d_counts);

    int tmp = 0;
    cudaCheck(cudaMemcpy(&tmp, d_counts + cand_0_size - 1, sizeof(int), cudaMemcpyDeviceToHost));
    total_matches += tmp;

    cudaCheck(cudaMalloc(&d_output, total_matches * sizeof(int2)));

    fill_matches_kernel_sym<<<grid_size, block_size>>>(
        d_cand_0, cand_0_size, G.d_offset_, G.d_adj_, d_cand_1, d_counts, d_output, d_partial_order
    );
    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaFree(d_cand_0));
    cudaCheck(cudaFree(d_cand_1));
    cudaCheck(cudaFree(d_counts));
    cudaCheck(cudaFree(d_partial_order));

    cnt = total_matches;

    return (int *)d_output;
}
