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
    InputParser cmd_parser(argc, argv);
    assert(cmd_parser.check_cmd_option_exists("-q"));
    assert(cmd_parser.check_cmd_option_exists("-d"));
    std::string input_query_graph_file = cmd_parser.get_cmd_option("-q");
    std::string input_data_graph_file = cmd_parser.get_cmd_option("-d");

    bool simple_filter = false;
    if (cmd_parser.check_cmd_option_exists("--simple-filter")) {
        simple_filter = true;
    }

    int dev_id = 0;
    if (cmd_parser.check_cmd_option_exists("--device")) {
        dev_id = atoi(cmd_parser.get_cmd_option("--device").c_str());
    }

    if (dev_id < 0) {
        dev_id = selectDeviceWithMaxFreeMemory();
    }
    cudaSetDevice(dev_id);
    check_gpu_props();

    Graph original_query(input_query_graph_file);
    std::vector<int> matching_order;
    original_query.generate_matching_order(matching_order);

    Graph Q(input_query_graph_file, matching_order);
    Graph G(input_data_graph_file);

#ifdef SYMMETRY_BREAKING
    int alpha = 0;
    std::vector<uint32_t> partial_order;
    std::vector<int> represent;
    alpha = Q.restriction_generation(partial_order, represent);
    printf("Symmetry = %d\n", alpha);
#endif

    TIME_INIT();
    TIME_START();

#ifdef SYMMETRY_BREAKING
    candidate_graph CG(Q, G, simple_filter, represent);
#else
    candidate_graph CG(Q, G, simple_filter);
#endif
    candidate_graph_GPU CG_GPU(CG);

    Graph_GPU Q_GPU(Q, true);
    Graph_GPU G_GPU(G, false);

    check_gpu_memory();

    TIME_END();
    PRINT_LOCAL_TIME("Preparation");
    TIME_START();

    int expected_init_num = 1e6;
    if (cmd_parser.check_cmd_option_exists("--initial")) {
        expected_init_num = atoi(cmd_parser.get_cmd_option("--initial").c_str());
    }

    ull ret = 0;
#ifdef SYMMETRY_BREAKING
    if (alpha == 1) {
        ret = join_bfs_dfs(Q, G, Q_GPU, G_GPU, CG, CG_GPU, expected_init_num);
    }
    else {
        ret = alpha * join_bfs_dfs_sym(Q, G, Q_GPU, G_GPU, CG, CG_GPU, expected_init_num, partial_order);
    }
#else
    ret = join_bfs_dfs(Q, G, Q_GPU, G_GPU, CG, CG_GPU, expected_init_num);
#endif

    printf("\033[41;37m\nResult: %llu\n\033[0m\n", ret);

    TIME_END();
    PRINT_TOTAL_TIME("Subgraph Matching Finished");

    Q_GPU.deallocate();
    G_GPU.deallocate();
    CG_GPU.deallocate();

    return 0;
}
