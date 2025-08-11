import sys
import os
import glob
import argparse
from subprocess import Popen, PIPE
import time


def generate_args(binary, *params):
    arguments = [binary]
    arguments.extend(list(params))
    return arguments


def execute_binary(args):
    process = Popen(' '.join(args), shell=True, stdout=PIPE, stderr=PIPE)
    (std_output, std_error) = process.communicate()
    process.wait()
    rc = process.returncode
    return rc, std_output, std_error


def check_correctness(binary_path, data_graph_path, query_graph_path, expected_embedding_num):
    execution_args = generate_args(binary_path, '-d', data_graph_path, '-q', query_graph_path)

    start_time = time.time()
    (rc, std_output, std_error) = execute_binary(execution_args)
    end_time = time.time()

    elapsed_time = end_time - start_time

    if rc == 0:
        embedding_num = 0
        std_output_list = std_output.decode().split('\n')
        for line in std_output_list:
            if 'Result:' in line:
                embedding_num = int(line.split(':')[1].strip())
                break
        if embedding_num != expected_embedding_num:
            print('Query {0} is wrong. Expected {1}, Output {2}'.format(query_graph_path, expected_embedding_num,
                                                              embedding_num))
            return
        print('Pass {0}:{1}'.format(query_graph_path, embedding_num))
        print('Time taken: {:.2f} seconds'.format(elapsed_time))
    else:
        print('Query {0} execution error.'.format(query_graph_path))
        return


def check(binary_path, data_graph_path, query_graph_path):
    execution_args = generate_args(binary_path, '-d', data_graph_path, '-q', query_graph_path)

    start_time = time.time()
    (rc, std_output, std_error) = execute_binary(execution_args)
    end_time = time.time()

    elapsed_time = end_time - start_time

    if rc == 0:
        embedding_num = 0
        std_output_list = std_output.decode().split('\n')
        for line in std_output_list:
            if 'Result:' in line:
                embedding_num = int(line.split(':')[1].strip())
                break
        print('Query {0} result: {1}'.format(query_graph_path, embedding_num))
        print('Time taken: {:.2f} seconds'.format(elapsed_time))
    else:
        print('Query {0} execution error.'.format(query_graph_path))
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Subgraph Matching Test')
    parser.add_argument('--binary', type=str, default='../bitmap/build/SubgraphMatching',
                      help='Path to the subgraph matching binary')
    args = parser.parse_args()

    binary_path = args.binary
    if not os.path.isfile(binary_path):
        print('The binary {0} does not exist.'.format(binary_path))
        exit(-1)

    input_expected_results = []
    input_expected_results_file = 'expected_output.txt'
    with open(input_expected_results_file, 'r') as f:
        for line in f:
            if line:
                input_expected_results.append(int(line.strip()))

    data_graph_path = 'data.graph'
    query_graph_files = ['query_graph/Q_{}.graph'.format(i) for i in range(22)]

    for i, query_graph_file in enumerate(query_graph_files):
        check_correctness(binary_path, data_graph_path, query_graph_file, input_expected_results[i])