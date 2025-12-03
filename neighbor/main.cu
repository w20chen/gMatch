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

    if (is_csr) {
        G.remove_degree_one_layer(Q.min_degree());
    }

#ifdef SYMMETRY_BREAKING
    if (is_csr && cmd_parser.check_cmd_option_exists("--reorder")) {
        bool ascending = cmd_parser.get_cmd_option("--reorder") == "1";
        G.reorder_by_degree(ascending);
        if (cmd_parser.check_cmd_option_exists("--dump")) {
            if (ascending) input_data_graph_file += ".1.cache";
            else input_data_graph_file += ".0.cache";
            G.write_csr(input_data_graph_file.c_str());
        }
    }

    if (is_csr && Q.is_clique() && !cmd_parser.check_cmd_option_exists("--no-orientation")) {
        G.convert_to_degree_dag();
    }
#endif

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

    ull edge_list_bound = (ull)G.ecount() * 2 * sizeof(int) * 2;
    ull free_memory = GPUFreeMemory();
    if (G.is_dag) {
        edge_list_bound /= 2;
    }
    printf("edge_list_bound: %llu bytes, free_memory: %llu bytes\n", edge_list_bound, free_memory);
    if (edge_list_bound > free_memory) {
        printf("# no-flitering mode is activated.\n");
        no_filtering = true;
    }

    ull ret = 0;
    bool induced_flag = (Q.is_clique() && Q.vcount() > 3);
    if(cmd_parser.check_cmd_option_exists("--no-induced")) {
        induced_flag = false;
    }

    if (G.max_degree() > MAX_DEGREE) {
        induced_flag = false;
    }

#ifdef SYMMETRY_BREAKING
    if (induced_flag) {
        if (!G.is_dag) {
            printf("# join_induced\n");
            ret = alpha * join_induced(Q, G, Q_GPU, G_GPU, CG, CG_GPU, partial_order);
        }
        else {
            printf("# join_induced_orientation\n");
            ret = join_induced_orientation(Q, G, Q_GPU, G_GPU, CG, CG_GPU);
        }
    }
    else if ((alpha == 1 || G.is_dag) && !no_filtering) {
        printf("# join_bfs_dfs\n");
        ret = join_bfs_dfs(Q, G, Q_GPU, G_GPU, CG, CG_GPU, no_memory_pool);
    }
    else {
        if (no_filtering) {
            printf("# join_no_filtering\n");
            ret = alpha * join_no_filtering(Q, G, Q_GPU, G_GPU, CG, CG_GPU, partial_order);
        }
        else {
            printf("# join_bfs_dfs_sym\n");
            ret = alpha * join_bfs_dfs_sym(Q, G, Q_GPU, G_GPU, CG, CG_GPU, partial_order, no_memory_pool);
        }
    }
#else
    printf("# join_bfs_dfs\n");
    ret = join_bfs_dfs(Q, G, Q_GPU, G_GPU, CG, CG_GPU, no_memory_pool);
#endif

    if (!G.is_dag || (G.is_dag && no_filtering)) {
        printf("\033[41;37m\nResult: %llu\n\033[0m\n", ret);
        printf("unique count: %llu\n", ret / alpha);
    }
    else {
        printf("\033[41;37m\nResult: %llu\n\033[0m\n", ret * alpha);
        printf("unique count: %llu\n", ret);
    }

    Q_GPU.deallocate();
    G_GPU.deallocate();

    return 0;
}
