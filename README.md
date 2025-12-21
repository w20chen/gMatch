# gMatch: Fine-Grained and Hardware-Efficient Subgraph Matching on GPUs
An efficient subgraph matching algorithm optimized for NVIDIA GPUs using CUDA, featuring *high GPU utilization*, *low GPU memory consumption*, and *load-balanced parallel execution*.

## Introduction
This project implements a GPU-accelerated subgraph matching algorithm featuring:

- Parallel BFS and DFS kernels for efficient subgraph isomorphism enumeration
- An efficient DFS backtracking stack that optimizes GPU thread utilization
- Candidate filtering techniques to generate minimal candidate sets
- Symmetry breaking techniques to ensure each unique subgraph is detected exactly once
- A GPU-optimized data structure for retrieving candidate edges
- A bitmap-based data structure to accelerate set intersection
- Load balancing techniques for ensuring inter-warp workload balance

## Compile
This project requires CMake (version 3.30.1), Make (version 4.3), GCC (version 10.5.0), and NVCC (version 12.5). You can compile the code by executing the following commands.
```bash
bash compile.sh
```
After successful compilation, the binary files are created in the `bitmap/build`, `hash_table/build`, and `neighbor/build` directories.


## Graph File Format
### Text Format

gMatch only accepts specific input formats. An invalid format is likely to cause errors. A valid graph file follows the structure below:

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

This format defines an undirected graph where vertices are labeled and degrees are explicitly listed for each node. For a graph $g$, the first line should be `t` $|V(g)|$ $|E(g)|$, followed by $|V(g)|$ vertex lines and $|E(g)|$ edge lines. Each vertex $v\in V(g)$ has a unique vertex ID from 0 to $|V(g)|-1$.

### Binary Format

gMatch also supports a **CSR (Compressed Sparse Row) binary format** for large data graphs, structured as follows: The file begins with two **32-bit integers** representing the **vertex count** $|V(G)|$ and **edge count** $|E(G)|$. Next comes the **offset array**, consisting of $|V(G)| + 1$ **64-bit unsigned integers (unsigned long long)**, where the $i$-th entry points to the start of vertex $i$'s edges in the edge array (0-based), and the last entry equals $2 \times |E(G)|$. Following this is the **vertex label array**, storing $|V(G)|$ **32-bit integers**, with the $i$-th value representing the label of vertex $i$. Finally, the **edge data** consists of $2 \times |E(G)|$ **32-bit integers**, organized as consecutive destination vertices for each edge, with all edges sorted by their source vertex as defined by the offset array.

**To specify the input format, use the command-line parameter `-b` for CSR binary format and `-d` for text format.**

## Datasets

| Dataset | Download Link |
|--------|----------|
| dblp, enron, gowalla, github, wikitalk | https://zenodo.org/records/17990755 |
| pokec, friendster, orkut, livejournal, ldbc_sf10 | https://zenodo.org/records/17994634 |
| rmat | https://zenodo.org/records/17998026 |
| ldbc_sf3, ldbc_sf10, ldbc_sf30, ldbc_sf100 | https://zenodo.org/records/17996944 |

## Test
The correctness of the algorithm can be verified using the following commands:

```bash
cd test
python3 test.py --binary ../bitmap/build/SubgraphMatching
python3 test.py --binary ../hash_table/build/SubgraphMatching
```

## Execute
After successful compilation, three binary files are created in the `bitmap/build`, `hash_table/build`, and `neighbor/build` directories, respectively. They all solve the subgraph matching problem but are designed for different scenarios.

### 1. Run with Bitmap-Based Candidate Graph

You can execute `bitmap/build/SubgraphMatching` using the following command.

```bash
./bitmap/build/SubgraphMatching -d <data_graph_path> -q <query_graph_path>
```

Here is an example:
```bash
./bitmap/build/SubgraphMatching -d test/naive/data.graph -q test/naive/query_graph/Q_0.graph
```

`test_bitmap.py` is a script for running batches of queries on datasets.

```bash
python3 test_bitmap.py --dataset dblp --label 16 --query 12
```
This runs all queries in the directory `dataset/dblp/label_16/query_graph/12` on the data graph `dataset/dblp/label_16/data.graph`.

### 2. Run with Hash Table-Based Candidate Graph

`hash_table/build/SubgraphMatching` provides another version of gMatch, using a hash table-based data structure for candidate set retrieval. `test_hash_table.py` works in the same way as `test_bitmap.py`. Hash table-based candidate graph is generally slower than bitmap-based but is able to maintain larger candidate sets.

### 3. Run with Neighborhood Intersection

`neighbor/build/SubgraphMatching` is designed for large data graphs (i.e., LiveJournal, LDBC, Orkut, Friendster, RMAT, and Pokec) where candidate graph is deactivated, and local candidate sets are computed by intersecting the neighborhoods of backward neighbors. This program processes **unlabeled** graphs by default.

```bash
 ./neighbor/build/SubgraphMatching -b dataset/orkut/label_1/orkut.bin -q patterns/p2
```

## Compared Systems Setup
### EGSM
This algorithm is from the paper: "Xibo Sun and Qiong Luo. Efficient GPU-Accelerated Subgraph Matching. SIGMOD 2023." We use EGSM as a comparison system for our experiments. The code is from https://github.com/RapidsAtHKUST/EGSM.

You can use the following commands to compile EGSM.
```bash
cd compared_system/EGSM
mkdir build
cd build
cmake ..
make
```

Run `compared_system/test_egsm.py` for experimental results of EGSM.

### STMatch

This algorithm is from the paper "Yihua Wei, Peng Jiang. STMatch: Accelerating Graph Pattern Matching on GPU with Stack-Based Loop Optimizations. SC 2022."

We use STMatch as a comparison system for our experiments. The code is from https://github.com/HPC-Research-Lab/STMatch. For fairness, we modify the code so that it uses the same matching order for subgraph matching. We also modify the code for graph file parsing so that it can process our graph format correctly.

You can use the following commands to compile STMatch.

```bash
cd compared_system/STMatch
mkdir bin
make
```

### T-DFS

This algorithm is from the paper "Lyuheng Yuan, et al. Faster Depth-First Subgraph Matching on GPUs. ICDE 2024." The code is from https://github.com/lyuheng/tdfs. We modify the code so that it uses the same matching order for subgraph matching. We also modify the code for graph file parsing so that it can process our graph format correctly.

You can use the following commands to compile T-DFS.

```bash
cd compared_system/TDFS
mkdir bin
make
```
