#  gMatch: Fine-Grained and Hardware-Efficient Subgraph Matching on GPUs
An efficient subgraph matching algorithm optimized for NVIDIA GPUs using CUDA, featuring high GPU utilization and designed for load-balanced parallel execution.

## Introduction
This project implements a GPU-accelerated subgraph matching algorithm featuring:

- Parallel BFS and DFS kernels for efficient subgraph isomorphism enumeration
- An efficient DFS backtracking stack that optimizes GPU thread utilization
- Candidate filtering techniques to generate minimal candidate sets
- Symmetry breaking techniques to ensure each unique subgraph is detected exactly once
- A GPU-optimized data structure for retrieving candidate edges
- A bitmap-based data structure to accelerate set intersection
- Load balancing techniques for ensuring inter-warp workload balance via queuing and memory pool utilization

## Compile
Our program requires CMake (version 3.30.1), Make (version 4.3), GCC (version 10.5.0), and NVCC (version 12.5). One can compile the code by executing the following commands.
```bash
bash compile.sh
```
After a successful compilation, the binary files are created under `hash_table/build`, `bitmap/build`, `neighbor/build`, and `work_stealing/build` directory.


## Graph File Format
Our program only allows a specific format as the input. Invalid format is likely to cause error. A valid graph file follows a specific format with the following structure:

1. **Header Line** (`t`):
    - Starts with `t`, followed by two integers representing:
        - Total number of vertices
        - Total number of edges

2. **Vertex Lines** (`v`):
    - Each line starts with `v`, followed by three integers per vertex:
        - Vertex ID (0-based index)
        - Label (arbitrary numeric value)
        - Degree (number of edges connected to this vertex)

3. **Edge Lines** (`e`):
    - Each line starts with `e`, followed by two vertex IDs (0-based) representing an undirected edge between them.
    - Edges are unordered (e.g., `e 0 1` is the same as `e 1 0`).

This format defines an undirected graph where vertices are labeled and degrees are explicitly listed for each node. For a graph $g$, the first line should be `t` $|V(g)|$ $|E(g)|$, followed by $|V(g)|$ vertex lines and $|E(g)|$ edge lines. Each vertex $v\in V(g)$ owns an unique vertex ID from 0 to $|V(g)|-1$.

## Test
The algorithm's correctness can be verified via the following command:

```bash
cd test/naive
python3 test.py --binary ../../hash_table/build/SubgraphMatching
python3 test.py --binary ../../bitmap/build/SubgraphMatching
python3 test.py --binary ../../neighbor/build/SubgraphMatching
python3 test.py --binary ../../work_stealing/build/SubgraphMatching
```

More test cases can be found in `test/hprd`.

## Execute
After a successful compilation, there are three binary files created under `hash_table/build`, `bitmap/build`, `neighbor/build`, and `work_stealing/build` directory respectively. They all solves the subgraph matching problem but might have different performance on different scenarios.

### 1. Run with Hash Table

One can execute the program `hash_table/build/SubgraphMatching` using the following command.

```bash
./hash_table/build/SubgraphMatching -d <data_graph_path> -q <query_graph_path>
```

Here is an example:
```bash
./hash_table/build/SubgraphMatching -d test/naive/data.graph -q test/naive/query_graph/Q_0.graph
```

`test.py` is a script for running batches of queries on datasets. By using

```bash
python3 test.py --dataset dblp --label 16 --query 12
```
we are running all queries in the directory `dataset/dblp/label_16/query_graph/12` on the data graph `dataset/dblp/label_16/data.graph`.

### 2. Run with Bitmap

`bitmap/build/SubgraphMatching` provides another version of our program, using a bitmap-based data structure to accelerate set intersection.

`test_bitmap.py` works in the same way with `test.py`, but it is for `bitmap/build/SubgraphMatching`.

### 3. Run with Work Stealing

`work_stealing/build/SubgraphMatching` provides another version of our program, using work stealing technique and the bitmap-based data structure. By using the parameter `--initial`, one can set the size of memory pool for initial tasks (i.e., number of initial partial matchings for DFS enumeration). The work stealing technique is necessary when `--initial` is small (e.g., $10^4$) and not useful when `--initial` is large (e.g., $10^7$).

### 4. Run with Neighborhood

`neighbor/build/SubgraphMatching` is designed for large data graph where candidate filtering is deactivated and local candidate sets are computed by intersecting neighborhoods of backward neighbors. This program processes unlabeled graph by default.

### 5. Adaptively Make the Best Choice

Alternatively, you can use `run.sh` to choose from all four versions adaptively. You can run a query on the data graph by using
```bash
python3 run.py -d <data_graph_path> -q <query_graph_path>
```
Set the parameter `--initial` if you want to see the performance under limited memory consumption.
