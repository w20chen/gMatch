#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <assert.h>
#include <string>
#include "helper.h"
#include "graph.h"
#include "candidate.h"
#include "join.h"


int
main(int argc, char **argv) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("#SM=%d, #warp per SM=%d, #total warp=%d\n",
    prop.multiProcessorCount,
    prop.maxThreadsPerMultiProcessor / prop.warpSize,
    prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / prop.warpSize));

#ifdef UNLABELED
    std::cout << "\033[41;37mThe data graph is unlabeled.\033[0m" << std::endl;
#else
    std::cout << "\033[41;37mThe data graph is labeled.\033[0m" << std::endl;
#endif

#ifdef SYMMETRY_BREAKING
    std::cout << "\033[41;37mSymmetry breaking is enabled.\033[0m" << std::endl;
#else
    std::cout << "\033[41;37mSymmetry breaking is disabled.\033[0m" << std::endl;
#endif

    InputParser cmd_parser(argc, argv);
    assert(cmd_parser.check_cmd_option_exists("-q"));
    assert(cmd_parser.check_cmd_option_exists("-d") ^ cmd_parser.check_cmd_option_exists("-b"));
    std::string input_query_graph_file = cmd_parser.get_cmd_option("-q");

    int dev_id = 0;
    if (cmd_parser.check_cmd_option_exists("--device")) {
        dev_id = atoi(cmd_parser.get_cmd_option("--device").c_str());
    }

    if (dev_id < 0) {
        dev_id = selectDeviceWithMaxFreeMemory();
    }
    cudaSetDevice(dev_id);
    check_gpu_props();

    bool no_memory_pool = true;
    if (cmd_parser.check_cmd_option_exists("--memory-pool")) {
        no_memory_pool = false;
    }

    bool no_filtering = false;
    if (cmd_parser.check_cmd_option_exists("--no-filtering")) {
        no_filtering = true;
    }

    Graph original_query(input_query_graph_file);
    std::vector<int> matching_order;
    original_query.generate_matching_order(matching_order);

    Graph Q(input_query_graph_file, matching_order);

    std::string input_data_graph_file;
    bool is_csr = false;

    if (cmd_parser.check_cmd_option_exists("-b")) {
        input_data_graph_file = cmd_parser.get_cmd_option("-b");
        is_csr = true;
    }
    else {
        input_data_graph_file = cmd_parser.get_cmd_option("-d");
        is_csr = false;
    }

    Graph G(input_data_graph_file, is_csr);

    std::cout << "Graph loaded to host memory." << std::endl;

#ifdef SYMMETRY_BREAKING
    int alpha = 0;
    std::vector<uint32_t> partial_order;
    alpha = Q.restriction_generation(partial_order);
    printf("Symmetry = %d\n", alpha);
#endif

    check_gpu_memory();

    TIME_INIT();
    TIME_START();

    candidate_graph CG(Q, G);
    candidate_graph_GPU CG_GPU(CG);

    check_gpu_memory();

    Graph_GPU Q_GPU(Q, true);
    Graph_GPU G_GPU(G, false);

    TIME_END();
    PRINT_TOTAL_TIME("Transfer time of the data graph");

    std::cout << "Graph loaded to device memory." << std::endl;

    check_gpu_memory();

    ull ret = 0;
#ifdef SYMMETRY_BREAKING
    if (alpha == 1) {
        ret = join_bfs_dfs(Q, G, Q_GPU, G_GPU, CG, CG_GPU, no_memory_pool);
    }
    else {
        if (no_filtering) {
            ret = alpha * join_no_filtering(Q, G, Q_GPU, G_GPU, CG, CG_GPU, partial_order);
        }
        else {
            ret = alpha * join_bfs_dfs_sym(Q, G, Q_GPU, G_GPU, CG, CG_GPU, partial_order, no_memory_pool);
        }
    }
#else
    ret = join_bfs_dfs(Q, G, Q_GPU, G_GPU, CG, CG_GPU, no_memory_pool);
#endif

    printf("\033[41;37m\nResult: %llu\n\033[0m\n", ret);

    Q_GPU.deallocate();
    G_GPU.deallocate();

    return 0;
}
