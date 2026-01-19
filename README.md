# gMatch: Fine-Grained and Hardware-Efficient Subgraph Matching on GPUs

An efficient subgraph matching algorithm optimized for NVIDIA GPUs using CUDA, featuring *high GPU utilization*, *low GPU memory consumption*, and *load-balanced parallel execution*.

<div align="left">
  <a href="#"><kbd>📄 Paper</kbd></a>
  <a href="https://github.com/w20chen/gMatch"><kbd>💻 Code</kbd></a>
  <a href="#"><kbd>📊 Report</kbd></a>
  <a href="https://deepwiki.com/w20chen/gMatch"><kbd>🌐 DeepWiki</kbd></a>
</div>

## Table of Contents
- [Introduction](#introduction)
- [Build](#build)
- [Testing](#testing)
- [Data Format](#data-format)
- [Datasets](#datasets)
- [Usage](#usage)
- [Reproducing Results](#reproducing-results)
- [Comparison Setup](#comparison-setup)


## Introduction
This project implements a GPU-accelerated subgraph matching algorithm featuring:

- Parallel BFS and DFS kernels for efficient subgraph isomorphism enumeration
- An optimized DFS backtracking stack that improves GPU thread utilization
- Candidate filtering techniques to produce reduced candidate sets
- Symmetry breaking techniques to ensure each unique subgraph is enumerated exactly once
- A GPU-optimized data structure for retrieving candidate edges
- A bitmap-based data structure to accelerate set intersection
- Load balancing techniques for inter-warp workload distribution

## Build
This project requires CMake (version 3.30.1), Make (version 4.3), GCC (version 10.5.0), and NVCC (version 12.5). One can compile the code using the following command.
```bash
bash compile.sh
```
After successful compilation, the binary files `bitmap/build/SubgraphMatching`, `hash_table/build/SubgraphMatching`, `neighbor/build/SubgraphMatching_unlabeled`, and `neighbor/build/SubgraphMatching_labeled` are created.

## Testing
The correctness of the algorithm can be verified using the following commands:

```bash
cd test
python3 test.py --binary ../bitmap/build/SubgraphMatching
python3 test.py --binary ../hash_table/build/SubgraphMatching
python3 test.py --binary ../neighbor/build/SubgraphMatching_labeled
```

## Data Format
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
|:--------:|:----------:|
| [dblp](https://snap.stanford.edu/data/com-DBLP.html), [enron](https://snap.stanford.edu/data/email-Enron.html), [gowalla](https://snap.stanford.edu/data/loc-Gowalla.html), [github](https://snap.stanford.edu/data/github-social.html), [wikitalk](https://snap.stanford.edu/data/wiki-Talk.html) | https://zenodo.org/records/17990755 |
| [pokec](https://snap.stanford.edu/data/soc-Pokec.html), [friendster](https://snap.stanford.edu/data/com-Friendster.html), [orkut](https://snap.stanford.edu/data/com-Orkut.html), [livejournal](https://snap.stanford.edu/data/com-LiveJournal.html), [ldbc_sf10](https://github.com/ldbc/ldbc_snb_datagen_hadoop) | https://zenodo.org/records/17994634 |
| [rmat](https://github.com/farkhor/PaRMAT) | https://zenodo.org/records/17998026 |
| [ldbc_sf3, ldbc_sf10, ldbc_sf30, ldbc_sf100](https://github.com/ldbc/ldbc_snb_datagen_hadoop) | https://zenodo.org/records/17996944 |


## Usage
After successful compilation, four binary files are created in the `bitmap/build`, `hash_table/build`, and `neighbor/build` directories. They all solve the subgraph matching problem but are designed for different scenarios.

### Run with Candidate Graph

This is designed for *large queries over medium-sized graphs* (e.g., dblp, enron, gowalla, github, and wikitalk).

#### Bitmap-Based

One can execute `bitmap/build/SubgraphMatching` using the following command.

```bash
./bitmap/build/SubgraphMatching -d <data_graph_path> -q <query_graph_path>
```

Here is an example:
```bash
./bitmap/build/SubgraphMatching -d test/naive/data.graph -q test/naive/query_graph/Q_0.graph
```

| Argument | Description |
|:------:|------|
| `-q` | Path to the query graph (required). |
| `-d` | Path to the data graph (text format only, required). |
| `--device` | GPU device ID to use (0 by default). |
| `--initial` | Threshold for the number of partial matches at which BFS ends (default: $10^6$). |


#### Hash Table-Based

`hash_table/build/SubgraphMatching` provides another version of gMatch, using a hash table-based data structure for candidate set retrieval. Hash table-based candidate graph is generally slower than bitmap-based but is able to maintain larger candidate sets.

### Run with Neighborhood Intersection

This is designed for *small queries over large-scale graphs* (e.g., pokec, friendster, orkut, livejournal, ldbc, and rmat).

`neighbor/build/SubgraphMatching_unlabeled` is designed for large data graphs (e.g., pokec, friendster, orkut, livejournal, ldbc, and rmat) where candidate graph is deactivated, and local candidate sets are computed by intersecting the neighborhoods of backward neighbors. This program processes *unlabeled* graphs only. For labeled graphs, please use `neighbor/build/SubgraphMatching_labeled`.

```bash
./neighbor/build/SubgraphMatching_unlabeled -b dataset/orkut/label_1/orkut.bin -q patterns/p2
```

| Argument | Description |
|:----------:|-------------|
| `-q` | Path to the query graph file (required). |
| `-d` | Path to the data graph file in text format (deprecated). |
| `-b` | Path to the data graph file in binary CSR format. |
| `--device` | GPU device ID to use (0 by default). |
| `--memory-pool` | Enable memory pooling for BFS (deprecated, disabled by default). |
| `--no-filtering` | Manually disable candidate filtering (automatically added when the data graph is too large). |
| `--reorder [0/1]` | Reorder vertices by degree for CSR graphs (disabled by default; 0/1 for descending/ascending). |
| `--dump` | Save reordered CSR graph to a cache file for future use (requires `--reorder`) |
| `--no-orientation` | Disable the conversion of the data graph to a DAG (an optimization for cliques, enabled by default). |
| `--no-induced` | Disable induced subgraph optimization (enabled by default). |


## Reproducing Results
To reproduce the experimental results of gMatch in *Figure 9* of the paper, please use `test_bitmap.py`. For example, `python3 test_bitmap.py --dataset enron --label 16 --query 12`. This will run a batch of 12-vertex queries on the data graph enron. 

**Note**: It is necessary to prepare the data graph as `dataset/enron/label_16/data.graph`, and all the query graphs in the directory `dataset/enron/label_16/query_graph/12/` with file names such as `Q_0`, `Q_1`, `Q_2`, etc.

To reproduce the results of gMatch in *Table 4* of the paper, please use `neighbor/build/SubgraphMatching_unlabeled`. For example, ` ./neighbor/build/SubgraphMatching_unlabeled -b dataset/livejournal/label_1/livejournal.bin -q patterns/p2`.


## Comparison Setup
### EGSM
This algorithm is from the paper: "Xibo Sun and Qiong Luo. Efficient GPU-Accelerated Subgraph Matching. SIGMOD 2023." We use EGSM as a comparison system for our experiments. The code is from https://github.com/RapidsAtHKUST/EGSM.

One can use the following commands to compile EGSM.
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

One can use the following commands to compile STMatch.

```bash
cd compared_system/STMatch
mkdir bin
make
```

### T-DFS

This algorithm is from the paper "Lyuheng Yuan, et al. Faster Depth-First Subgraph Matching on GPUs. ICDE 2024." The code is from https://github.com/lyuheng/tdfs. We modify the code so that it uses the same matching order for subgraph matching. We also modify the code for graph file parsing so that it can process our graph format correctly.

One can use the following commands to compile T-DFS.

```bash
cd compared_system/TDFS
mkdir bin
make
```
