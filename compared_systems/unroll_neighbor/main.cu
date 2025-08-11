#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <assert.h>
#include <string>
#include "helper.h"
#include "graph.h"
#include "join.h"


int
main(int argc, char **argv) {
    InputParser cmd_parser(argc, argv);
    assert(cmd_parser.check_cmd_option_exists("-q"));
    assert(cmd_parser.check_cmd_option_exists("-b"));
    std::string input_query_graph_file = cmd_parser.get_cmd_option("-q");
    std::string input_data_graph_file = cmd_parser.get_cmd_option("-b");

    int dev_id = 1;
    if (cmd_parser.check_cmd_option_exists("--device")) {
        dev_id = atoi(cmd_parser.get_cmd_option("--device").c_str());
    }

    cudaSetDevice(dev_id);
    check_gpu_props();

    Graph Q(input_query_graph_file, true);
    Graph G(input_data_graph_file, false);

    std::vector<int> matching_order;
    Q.generate_matching_order(matching_order);
    Q.generate_backward_neighborhood(matching_order);

    Graph_GPU Q_GPU(Q, true);
    Graph_GPU G_GPU(G, false);

    std::vector<uint32_t> partial_order;
    int alpha = Q.restriction_generation(partial_order);
    printf("### Symmetry breaking is enabled.\n");
    printf("Symmetry = %d\n", alpha);

    TIME_INIT();
    TIME_START();

    ull ret = alpha * join_bfs_dfs_sym(Q, G, Q_GPU, G_GPU, matching_order, partial_order);

    printf("\033[41;37m\nResult: %llu\n\033[0m\n", ret);

    TIME_END();
    PRINT_LOCAL_TIME("Total");

    return 0;
}
