from subprocess import Popen, PIPE, TimeoutExpired
import argparse
import os

def generate_args(binary, *params):
    arguments = [binary]
    arguments.extend(list(params))
    return arguments

def execute_binary(args, timeout=600):  # timeout 600 sec
    try:
        process = Popen(args, stdout=PIPE, stderr=PIPE, text=True)
        (std_output, std_error) = process.communicate(timeout=timeout)
        rc = process.returncode
        return rc, std_output, std_error
    except TimeoutExpired:
        process.kill()
        std_output, std_error = process.communicate()
        print("Process timed out")
        return -1, "Process timed out", std_error

def check(binary_path, data_graph_path, query_graph_path):
    execution_args = generate_args(binary_path, '-d', data_graph_path, '-q', query_graph_path)
    (rc, std_output, std_error) = execute_binary(execution_args)

    if rc == 0:
        embedding_num = 0
        elapsed_time = 0
        std_output_list = std_output.split('\n')
        for line in std_output_list:
            if 'Result:' in line:
                embedding_num = int(line.split(':')[1].strip())
            if 'Processing' in line:
                elapsed_time = int(line.split(':')[1].split('(')[0].strip())
        query_id = os.path.basename(query_graph_path).split('_')[-1]
        print(f'{query_id},{embedding_num},{elapsed_time}')
    else:
        print(f'Query {query_graph_path} execution error.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Execute multiple subgraph queries on the data graph')
    parser.add_argument('--dataset', required=True, type=str, help='Dataset name')
    parser.add_argument('--label', required=True, type=int, help='Number of labels')
    parser.add_argument('--query', required=True, type=str, help='Name of query set (e.g., 12, 12-dense, 12-sparse)')
    parser.add_argument('--start', required=False, type=str, default='0', help='Start query ID')
    args = parser.parse_args()

    binary_path = './hash_table/build/SubgraphMatching'
    data_path = os.path.join('dataset', args.dataset, f'label_{args.label}', 'data.graph')
    query_dir = os.path.join('dataset', args.dataset, f'label_{args.label}', 'query_graph', str(args.query))

    filenames = []
    for filename in os.listdir(query_dir):
        if int(filename.split('_')[-1]) < int(args.start):
            continue
        file_path = os.path.join(query_dir, filename)
        if os.path.isfile(file_path):
            filenames.append(file_path)

    filenames = sorted(filenames, key=lambda x: int(x.split('_')[-1]))

    for file_path in filenames:
        check(binary_path, data_path, file_path)