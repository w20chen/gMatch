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

    int dev_id = 0;
    if (cmd_parser.check_cmd_option_exists("--device")) {
        dev_id = atoi(cmd_parser.get_cmd_option("--device").c_str());
    }

    cudaSetDevice(dev_id);
    check_gpu_props();

    bool recursive_filtering = true;
    if (cmd_parser.check_cmd_option_exists("--recursive-filtering")) {
        recursive_filtering = atoi(cmd_parser.get_cmd_option("--recursive-filtering").c_str());
    }

    TIME_INIT();
    TIME_START();

    Graph Q(input_query_graph_file, true);
    Graph G(input_data_graph_file, false);

    candidate_graph CG(Q, G, recursive_filtering);
    candidate_graph_GPU CG_GPU(CG);

    std::vector<int> matching_order;
    Q.generate_matching_order(matching_order);
    Q.generate_backward_neighborhood(matching_order);

    Graph_GPU Q_GPU(Q);
    Graph_GPU G_GPU(G);

    TIME_END();
    PRINT_LOCAL_TIME("Preparation");
    TIME_START();

    ull ret = join_bfs_dfs(Q, G, Q_GPU, G_GPU, CG, CG_GPU, matching_order);

    printf("\033[41;37m\nResult: %llu\n\033[0m\n", ret);

    TIME_END();
    PRINT_LOCAL_TIME("Processing");
    PRINT_TOTAL_TIME("Subgraph Matching Finished");

    Q_GPU.deallocate();
    G_GPU.deallocate();
    CG_GPU.deallocate();

    return 0;
}
