#include "graph.h"

void
Graph::generate_bfs_order(std::vector<int> &bfs_order, int start) const {
    std::queue<int> Q;
    std::vector<int> visited(vcount_, 0);
    Q.push(start);
    visited[start] = 1;
    bfs_order.push_back(start);

    while (!Q.empty()) {
        int v = Q.front();
        Q.pop();
        for (int vv : adj_[v]) {
            if (visited[vv] == 0) {
                Q.push(vv);
                visited[vv] = 1;
                bfs_order.push_back(vv);
            }
        }
    }
}

void
Graph::generate_matching_order(std::vector<int> &matching_order) const {
    if (vcount_ == 4 && ecount_ == 4 && *max_element(deg_.begin(), deg_.end()) == 2) {
        // Rectangle (4-cycle)
        matching_order.resize(4);
        matching_order[0] = 0;
        matching_order[1] = 1;
        matching_order[2] = 3;
        matching_order[3] = 2;
        return;
    }

    int n = vcount_;
    std::vector<bool> visited(n, false);

    int selected_vertex = 0;
    int selected_vertex_selectivity = deg_[selected_vertex];

    for (int u = 1; u < n; ++u) {
        int u_selectivity = deg_[u];
        if (u_selectivity > selected_vertex_selectivity) {
            selected_vertex = u;
            selected_vertex_selectivity = u_selectivity;
        }
    }

    matching_order.push_back(selected_vertex);
    visited[selected_vertex] = true;

    std::vector<int> tie_vertices;
    std::vector<int> temp;

    for (int _i = 1; _i < n; ++_i) {
        selected_vertex_selectivity = 0;
        for (int u = 0; u < n; ++u) {
            if (!visited[u]) {
                int u_selectivity = 0;
                for (auto uu : matching_order) {
                    if (is_adjacent(u, uu)) {
                        u_selectivity += 1;
                    }
                }
                if (u_selectivity > selected_vertex_selectivity) {
                    selected_vertex_selectivity = u_selectivity;
                    tie_vertices.clear();
                    tie_vertices.push_back(u);
                }
                else if (u_selectivity == selected_vertex_selectivity) {
                    tie_vertices.push_back(u);
                }
            }
        }

        if (tie_vertices.size() != 1) {
            temp.swap(tie_vertices);
            tie_vertices.clear();

            int count = 0;
            std::vector<int> u_fn;
            for (auto u : temp) {
                for (auto uu : adj_[u]) {
                    if (!visited[uu]) {
                        u_fn.push_back(uu);
                    }
                }

                int cur_count = 0;
                for (auto uu : matching_order) {
                    auto &uun = adj_[uu];
                    std::vector<int> uun_tmp;
                    uun_tmp.insert(uun_tmp.end(), uun.begin(), uun.end());

                    int common_neighbor_count = 0;
                    for (int ii : uun_tmp) {
                        for (int jj : u_fn) {
                            if (ii == jj) {
                                common_neighbor_count++;
                                break;
                            }
                        }
                        if (common_neighbor_count != 0) {
                            break;
                        }
                    }

                    if (common_neighbor_count > 0) {
                        cur_count += 1;
                    }
                }

                u_fn.clear();

                if (cur_count > count) {
                    count = cur_count;
                    tie_vertices.clear();
                    tie_vertices.push_back(u);
                }
                else if (cur_count == count) {
                    tie_vertices.push_back(u);
                }
            }
        }

        if (tie_vertices.size() != 1) {
            temp.swap(tie_vertices);
            tie_vertices.clear();

            int count = 0;
            std::vector<int> u_fn;
            for (auto u : temp) {
                for (auto uu : adj_[u]) {
                    if (!visited[uu]) {
                        u_fn.push_back(uu);
                    }
                }

                int cur_count = 0;
                for (auto uu : u_fn) {
                    bool valid = true;
                    for (auto uuu : matching_order) {
                        if (is_adjacent(uu, uuu)) {
                            valid = false;
                            break;
                        }
                    }
                    if (valid) {
                        cur_count += 1;
                    }
                }

                u_fn.clear();

                if (cur_count > count) {
                    count = cur_count;
                    tie_vertices.clear();
                    tie_vertices.push_back(u);
                }
                else if (cur_count == count) {
                    tie_vertices.push_back(u);
                }
            }
        }

        matching_order.push_back(tie_vertices[0]);
        visited[tie_vertices[0]] = true;
        tie_vertices.clear();
        temp.clear();
    }

    std::cout << "Matching order: ";
    for (auto v : matching_order) {
        std::cout << v << " ";
    }
    std::cout << std::endl;
}


Graph::Graph(const std::string &file_path, bool is_csr) {
    is_dag = false;

    if (is_csr) {
        if (file_path.find(".bin") != std::string::npos) {
            parse_csr(file_path.c_str());
        }
        else {
            parse_g2miner_format(file_path);
        }
        return;
    }

    h_offset = nullptr;
    h_array = nullptr;

    FILE* infile = fopen(file_path.c_str(), "r");
    if (!infile) {
        std::cout << "Cannot open graph file " << file_path << "." << std::endl;
        exit(-1);
    }

    char type = 0;
    fscanf(infile, " %c %d %u", &type, &vcount_, &ecount_);

    adj_.resize(vcount_);
    deg_.resize(vcount_);

#ifndef UNLABELED
    vertex_label_.resize(vcount_);
#ifdef NLF_FILTER
    nlf_.resize(vcount_);
#endif
#endif

    while (fscanf(infile, " %c", &type) == 1) {
        if (type == 'v') {
            int vid, label, deg;
            fscanf(infile, "%d %d %d", &vid, &label, &deg);
#ifndef UNLABELED
            vertex_label_[vid] = label;
#endif
            deg_[vid] = deg;

            if (vid % (int)1e6 == 0) {
                std::cout << "Loading graph " << vid / (int)1e6 << " / " << vcount_ / (int)1e6 << " 1M vertices loaded." << std::endl;
            }
        }
        else {
            break;
        }
    }

    unsigned int edge_count = 0;

    for (int v = 0; v < vcount_; v++) {
        adj_[v].reserve(deg_[v]);
    }

    if (type == 'e') {
        int v1, v2;
        while (fscanf(infile, "%d %d", &v1, &v2) == 2) {
            adj_[v1].emplace_back(v2);
            adj_[v2].emplace_back(v1);

#ifndef UNLABELED
#ifdef NLF_FILTER
            nlf_[v1][vertex_label_[v2]]++;
            nlf_[v2][vertex_label_[v1]]++;
#endif
#endif
            edge_count++;
            if (fscanf(infile, " %c", &type) != 1) {
                break;
            }
            if (edge_count % (int)1e7 == 0) {
                std::cout << "Loading graph " << edge_count / (int)1e7 << " / " << ecount_ / (int)1e7 << " 10M edges loaded." << std::endl;
            }
        }
    }

    fclose(infile);

    if (edge_count != ecount_) {
        std::cout << "Edge count error: " << edge_count << " != " << ecount_ << std::endl;
        exit(-1);
    }

    for (int v = 0; v < vcount_; v++) {
        if (adj_[v].size() != deg_[v]) {
            std::cout << "Degree Error: ";
            std::cout << v << ", " << adj_[v].size() << " != " << deg_[v] << std::endl;
            exit(-1);
        }
    }

    for (auto &l : adj_) {
        std::sort(l.begin(), l.end());
    }

    std::cout << "Graph loaded from file " << file_path << "." << std::endl;

    print_meta();
}


Graph::Graph(const std::string &file_path, std::vector<int> &matching_order) {
    is_dag = false;

    h_offset = nullptr;
    h_array = nullptr;

    std::vector<int> rename(matching_order.size());
    for (int i = 0; i < matching_order.size(); i++) {
        rename[matching_order[i]] = i;
    }

    std::ifstream infile(file_path);
    if (!infile.is_open()) {
        std::cout << "Cannot open graph file " << file_path << "." << std::endl;
        exit(-1);
    }

    char type = 0;
    infile >> type >> vcount_ >> ecount_;

    adj_.resize(vcount_);
    deg_.resize(vcount_);

#ifndef UNLABELED
    vertex_label_.resize(vcount_);
#ifdef NLF_FILTER
    nlf_.resize(vcount_);
#endif
#endif

    while (infile >> type) {
        if (type == 'v') {
            int vid, label, deg;
            infile >> vid >> label >> deg;

            vid = rename[vid];

#ifndef UNLABELED
            vertex_label_[vid] = label;
#endif
            deg_[vid] = deg;
        }
        else {
            break;
        }
    }

    if (type == 'e') {
        std::string next_str;
        while (true) {
            int v1, v2;
            infile >> v1 >> v2;

            v1 = rename[v1];
            v2 = rename[v2];

            adj_[v1].emplace_back(v2);
            adj_[v2].emplace_back(v1);

#ifndef UNLABELED
#ifdef NLF_FILTER
            nlf_[v1][vertex_label_[v2]]++;
            nlf_[v2][vertex_label_[v1]]++;
#endif
#endif

            if (!(infile >> next_str)) {
                break;
            }
            if (next_str == "e") {
                continue;
            }
            else if (!(infile >> next_str)) {
                break;
            }
        }
    }

    infile.close();

    for (int v = 0; v < vcount_; v++) {
        if (adj_[v].size() != deg_[v]) {
            std::cout << "Degree Error: ";
            std::cout << v << ", " << adj_[v].size() << " != " << deg_[v] << std::endl;
        }
    }

    for (auto &l : adj_) {
        std::sort(l.begin(), l.end());
    }

    generate_backward_neighborhood();

    std::cout << "Query vertices are renamed according to the matching order." << std::endl;

    for (int v = 0; v < vcount_; v++) {
        printf("N(%d): ", v);
        for (int w : adj_[v]) {
            printf("%d ", w);
        }
        printf("\n");
    }
}


void
Graph::parse_csr(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    if (fread(&vcount_, sizeof(int), 1, file) != 1 || fread(&ecount_, sizeof(unsigned int), 1, file) != 1) {
        fclose(file);
        fprintf(stderr, "Error reading vertex/edge counts\n");
        exit(EXIT_FAILURE);
    }

    printf("|V|=%g |E|=%g\n", (double)vcount_, (double)ecount_);

    h_offset = new ull[vcount_ + 1];
    if (fread(h_offset, sizeof(ull), vcount_ + 1, file) != static_cast<size_t>(vcount_ + 1)) {
        fclose(file);
        delete[] h_offset;
        fprintf(stderr, "Error reading offsets array\n");
        exit(EXIT_FAILURE);
    }

    vertex_label_.resize(vcount_, 0);
    if (fread(vertex_label_.data(), sizeof(int), vcount_, file) != static_cast<size_t>(vcount_)) {
        fclose(file);
        delete[] h_offset;
        fprintf(stderr, "Error reading offsets array\n");
        exit(EXIT_FAILURE);
    }

    h_array = new int[2 * (ull)ecount_];
    if (fread(h_array, sizeof(int), 2 * (ull)ecount_, file) != 2 * (ull)ecount_) {
        fclose(file);
        delete[] h_offset;
        delete[] h_array;
        fprintf(stderr, "Error reading edges array\n");
        exit(EXIT_FAILURE);
    }

    deg_.resize(vcount_, 0);
    for (int i = 0; i < vcount_; i++) {
        deg_[i] = h_offset[i + 1] - h_offset[i];
    }

    printf("Data graph loaded from CSR format file %s\n|V|=%d |E|=%u\n", filename, vcount_, ecount_);
    std::cout << "Max Degree=" << *max_element(deg_.begin(), deg_.end()) << std::endl;
    std::cout << "Min Degree=" << *min_element(deg_.begin(), deg_.end()) << std::endl;
    std::cout << "Avg Degree=" << (static_cast<double>(ecount_) * 2 / vcount_) << std::endl;
    std::cout << "#labels=" << *max_element(vertex_label_.begin(), vertex_label_.end()) + 1 << std::endl;

    fclose(file);
}

void Graph::parse_g2miner_format(const std::string& prefix) {
    printf("Loading graph from g2miner format: %s\n", prefix.c_str());

    std::ifstream f_meta(prefix + ".meta.txt");
    if (!f_meta) {
        fprintf(stderr, "Error opening meta file: %s\n", (prefix + ".meta.txt").c_str());
        exit(EXIT_FAILURE);
    }

    ull n_vertices, n_edges;
    f_meta >> n_vertices >> n_edges;
    f_meta.close();

    vcount_ = (int)n_vertices;
    ecount_ = (unsigned int)(n_edges / 2);

    printf("|V|=%d |E|=%u\n", vcount_, ecount_);

    std::ifstream f_vertex(prefix + ".vertex.bin", std::ios::binary);
    if (!f_vertex) {
        fprintf(stderr, "Error opening vertex file: %s\n", (prefix + ".vertex.bin").c_str());
        exit(EXIT_FAILURE);
    }

    f_vertex.seekg(0, std::ios::end);
    size_t vertex_file_size = f_vertex.tellg();
    f_vertex.seekg(0, std::ios::beg);
    printf("Vertex file size: %zu bytes\n", vertex_file_size);

    h_offset = new ull[vcount_ + 1];
    f_vertex.read(reinterpret_cast<char*>(h_offset), sizeof(ull) * (vcount_ + 1));
    f_vertex.close();

    std::ifstream f_edge(prefix + ".edge.bin", std::ios::binary);
    if (!f_edge) {
        fprintf(stderr, "Error opening edge file: %s\n", (prefix + ".edge.bin").c_str());
        delete[] h_offset;
        exit(EXIT_FAILURE);
    }


    f_edge.seekg(0, std::ios::end);
    size_t edge_file_size = f_edge.tellg();
    f_edge.seekg(0, std::ios::beg);
    printf("Edge file size: %zu bytes\n", edge_file_size);

    h_array = new int[2 * (ull)ecount_];

    f_edge.read(reinterpret_cast<char*>(h_array), sizeof(int) * 2 * (ull)ecount_);
    f_edge.close();

    vertex_label_.resize(vcount_, 0);

    deg_.resize(vcount_, 0);
    for (int i = 0; i < vcount_; i++) {
        deg_[i] = h_offset[i + 1] - h_offset[i];
    }

    printf("Data graph loaded from original format\n");
    std::cout << "Max Degree=" << *max_element(deg_.begin(), deg_.end()) << std::endl;
    std::cout << "Min Degree=" << *min_element(deg_.begin(), deg_.end()) << std::endl;
    std::cout << "Avg Degree=" << (static_cast<double>(ecount_) * 2 / vcount_) << std::endl;
    std::cout << "#labels=" << *max_element(vertex_label_.begin(), vertex_label_.end()) + 1 << std::endl;
}

void Graph::remove_degree_one_layer(int x) {
    printf("Removing one layer of degree < %d vertices...\n", x);

    std::vector<bool> keep(vcount_, false);
    for (int u = 0; u < vcount_; u++) {
        if (deg_[u] >= x) {
            keep[u] = true;
        }
    }

    std::vector<int> new_deg(vcount_, 0);
    ull new_edge_count = 0;

    for (int u = 0; u < vcount_; u++) {
        if (!keep[u]) continue;

        ull start = h_offset[u];
        ull end = h_offset[u + 1];

        for (ull j = start; j < end; j++) {
            int v = h_array[j];
            if (keep[v]) {
                new_deg[u]++;
                new_edge_count++;
            }
        }
    }

    new_edge_count /= 2;

    ull* new_h_offset = new ull[vcount_ + 1];
    int* new_h_array = new int[2 * (ull)new_edge_count];

    new_h_offset[0] = 0;
    for (int i = 0; i < vcount_; i++) {
        new_h_offset[i + 1] = new_h_offset[i] + new_deg[i];
    }

    ull edge_index = 0;
    for (int u = 0; u < vcount_; u++) {
        if (!keep[u]) continue;

        ull start = h_offset[u];
        ull end = h_offset[u + 1];

        for (ull j = start; j < end; j++) {
            int v = h_array[j];
            if (keep[v]) {
                new_h_array[edge_index++] = v;
            }
        }
    }

    ull old_ecount = ecount_;
    ecount_ = (unsigned int)new_edge_count;

    for (int i = 0; i < vcount_; i++) {
        deg_[i] = new_deg[i];
    }

    delete[] h_offset;
    delete[] h_array;
    h_offset = new_h_offset;
    h_array = new_h_array;

    printf("Removed %llu edges connected to degree < %d vertices.\n", old_ecount - ecount_, x);
    printf("Graph after removing degree < %d vertices:\n", x);
    printf("|V|=%d |E|=%u\n", vcount_, ecount_);

    int max_deg = 0, min_deg = INT_MAX;
    ull total_deg = 0;
    for (int i = 0; i < vcount_; i++) {
        max_deg = std::max(max_deg, deg_[i]);
        min_deg = std::min(min_deg, deg_[i]);
        total_deg += deg_[i];
    }

    printf("Max Degree=%d\n", max_deg);
    printf("Min Degree=%d\n", min_deg);
    printf("Avg Degree=%.2f\n", static_cast<double>(total_deg) / vcount_);
}

void 
Graph::write_csr(const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error opening file for writing: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    if (fwrite(&vcount_, sizeof(int), 1, file) != 1 || fwrite(&ecount_, sizeof(unsigned int), 1, file) != 1) {
        fclose(file);
        fprintf(stderr, "Error writing vertex/edge counts\n");
        exit(EXIT_FAILURE);
    }

    if (fwrite(h_offset, sizeof(ull), vcount_ + 1, file) != static_cast<size_t>(vcount_ + 1)) {
        fclose(file);
        fprintf(stderr, "Error writing offsets array\n");
        exit(EXIT_FAILURE);
    }

    if (fwrite(vertex_label_.data(), sizeof(int), vcount_, file) != static_cast<size_t>(vcount_)) {
        fclose(file);
        fprintf(stderr, "Error writing vertex labels\n");
        exit(EXIT_FAILURE);
    }

    if (fwrite(h_array, sizeof(int), 2 * (ull)ecount_, file) != 2 * (ull)ecount_) {
        fclose(file);
        fprintf(stderr, "Error writing edges array\n");
        exit(EXIT_FAILURE);
    }

    fclose(file);
    printf("Data graph saved to CSR format file %s\n|V|=%d |E|=%u\n", filename, vcount_, ecount_);
}

int
Graph::find_automorphisms(std::vector<std::vector<uint32_t>> &embeddings) const {
    // Note that this method is working on small graphs (tens of vertices) only.

    // This method conducts subgraph matching from Q to Q.
    std::vector<bool> visited(vcount_, false);
    std::vector<uint32_t> idx(vcount_);
    std::vector<uint32_t> mapping(vcount_);
    std::vector<std::vector<uint32_t>> local_candidates(vcount_);
    std::vector<std::vector<uint32_t>> global_candidates(vcount_);
    std::vector<std::vector<uint32_t>> backward_neighbors(vcount_);

    // Initialize global candidates.
    for (uint32_t u = 0; u < vcount_; u++) {
        uint32_t u_label = label(u);
        uint32_t u_degree = degree(u);

        for (uint32_t v = 0; v < vcount_; v++) {
            uint32_t v_label = label(v);
            uint32_t v_degree = degree(v);

            if (v_label == u_label && v_degree == u_degree) {
                global_candidates[u].push_back(v);
            }
        }
    }

    // Set backward neighbors to compute local candidates.
    for (uint32_t i = 1; i < vcount_; i++) {
        for (uint32_t j = 0; j < i; j++) {
            if (is_adjacent(i, j)) {
                backward_neighbors[i].push_back(j);
            }
        }
    }

    // Recursive search along the matching order (0, 1, ..., n).
    int cur_level = 0;
    local_candidates[cur_level] = global_candidates[0];

    while (true) {
        while (idx[cur_level] < local_candidates[cur_level].size()) {
            uint32_t u = cur_level;
            uint32_t v = local_candidates[cur_level][idx[cur_level]];
            idx[cur_level] += 1;

            if (cur_level == vcount_ - 1) {
                // Find an embedding
                mapping[u] = v;
                embeddings.push_back(mapping);
            }
            else {
                mapping[u] = v;
                visited[v] = true;
                cur_level += 1;
                idx[cur_level] = 0;

                {
                    // Compute local candidates.
                    u = cur_level;
                    for (auto temp_v : global_candidates[u]) {
                        if (!visited[temp_v]) {
                            bool is_feasible = true;
                            for (auto uu : backward_neighbors[u]) {
                                uint32_t temp_vv = mapping[uu];
                                if (!is_adjacent(temp_v, temp_vv)) {
                                    is_feasible = false;
                                    continue;
                                }
                            }
                            if (is_feasible) {
                                local_candidates[cur_level].push_back(temp_v);
                            }
                        }
                    }
                }
            }
        }

        local_candidates[cur_level].clear();
        cur_level -= 1;
        if (cur_level < 0) {
            break;
        }
        visited[mapping[cur_level]] = false;
    }

    std::cout << "Found " << embeddings.size() << " automorphisms of the query graph." << std::endl;
    return embeddings.size();
}


int
Graph::restriction_generation(std::vector<uint32_t> &partial_order) const {
    assert(vcount_ <= 32);

    std::vector<std::vector<uint32_t>> automorphisms;
    int alpha = find_automorphisms(automorphisms);

    bool relation[32][32];
    memset(relation, 0, sizeof(relation));

    for (int u = 0; u < vcount_; u++) {
        std::vector<std::vector<uint32_t>> stablized_aut;
        for (auto &m : automorphisms) {
            if (m[u] == u) {
                stablized_aut.emplace_back(m);
            }
            else {
                relation[u][m[u]] = 1;
                if (u > m[u]) {
                    printf("Error\n");
                    exit(-1);
                }
            }
        }
        automorphisms = stablized_aut;
    }

    // Find all path longer than 1
    bool M[32][32];

    // Compute R^2
    for (int u = 0; u < vcount_; u++) {
        for (int v = 0; v < vcount_; v++) {
            M[u][v] = 0;
            for (int w = 0; w < vcount_; w++) {
                if (relation[u][w] && relation[w][v]) {
                    M[u][v] = 1;
                    break;
                }
            }
        }
    }

    for (int k = 2; k < vcount_ - 1; k++) {
        for (int u = 0; u < vcount_; u++) {
            for (int v = 0; v < vcount_; v++) {
                if (M[u][v]) {
                    continue;
                }
                for (int w = 0; w < vcount_; w++) {
                    if ((relation[u][w] && M[w][v]) || (M[u][w] && relation[w][v])) {
                        M[u][v] = 1;
                        break;
                    }
                }
            }
        }
    }

    // Only save the Hasse Graph
    partial_order.resize(vcount_, 0);

    for (int u = 0; u < vcount_; u++) {
        for (int v = 0; v < vcount_; v++) {
            if (relation[u][v] && !M[u][v]) {
                // u < v
                partial_order[v] |= (1 << u);
                printf("%d -> %d\n", u, v);
            }
        }
    }

    return alpha;
}


void Graph::reorder_by_degree(bool ascending) {
    if (ascending) {
        printf("Reordering graph by degree in ascending order.\n");
    }
    else {
        printf("Reordering graph by degree in descending order.\n");
    }

    std::vector<std::pair<int, int>> vertex_degree_pairs(vcount_);
    for (int i = 0; i < vcount_; i++) {
        vertex_degree_pairs[i] = std::make_pair(i, deg_[i]);
    }
    
    if (ascending) {
        // Sort vertices by degree in ascending order.
        std::sort(vertex_degree_pairs.begin(), vertex_degree_pairs.end(),
            [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                return a.second < b.second;
            });
    }
    else {
        std::sort(vertex_degree_pairs.begin(), vertex_degree_pairs.end(),
            [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                return a.second > b.second;
            });
    }
    
    std::vector<int> old_to_new(vcount_);
    for (int new_id = 0; new_id < vcount_; new_id++) {
        int old_id = vertex_degree_pairs[new_id].first;
        old_to_new[old_id] = new_id;
    }
    
    std::vector<int> new_vertex_label(vcount_);
    for (int i = 0; i < vcount_; i++) {
        new_vertex_label[old_to_new[i]] = vertex_label_[i];
    }
    vertex_label_ = std::move(new_vertex_label);
    
    std::vector<int> new_deg(vcount_);
    for (int i = 0; i < vcount_; i++) {
        new_deg[old_to_new[i]] = deg_[i];
    }
    deg_ = std::move(new_deg);
    
    std::vector<std::vector<int>> new_adj_list(vcount_);
    
    for (int old_u = 0; old_u < vcount_; old_u++) {
        int new_u = old_to_new[old_u];
        ull start = h_offset[old_u];
        ull end = h_offset[old_u + 1];
        
        for (ull j = start; j < end; j++) {
            int old_v = h_array[j];
            int new_v = old_to_new[old_v];
            new_adj_list[new_u].push_back(new_v);
        }
    }
    
    for (int i = 0; i < vcount_; i++) {
        std::sort(new_adj_list[i].begin(), new_adj_list[i].end());
    }
    
    delete[] h_offset;
    delete[] h_array;
    
    h_offset = new ull[vcount_ + 1];
    h_offset[0] = 0;
    for (int i = 0; i < vcount_; i++) {
        h_offset[i + 1] = h_offset[i] + new_adj_list[i].size();
    }
    
    h_array = new int[2 * (ull)ecount_];
    ull edge_index = 0;
    for (int i = 0; i < vcount_; i++) {
        for (int neighbor : new_adj_list[i]) {
            h_array[edge_index++] = neighbor;
        }
    }
    
    if (edge_index != 2 * (ull)ecount_) {
        fprintf(stderr, "Error: Edge count mismatch after reordering\n");
        exit(EXIT_FAILURE);
    }
    
    printf("Graph has been reordered by degree.\n");
}


void Graph::convert_to_degree_dag() {
    if (is_dag) {
        printf("Graph is already a DAG. Skipping conversion.\n");
        return;
    }
    is_dag = true;
    std::vector<std::vector<int>> new_adj_list(vcount_);

    for (int u = 0; u < vcount_; u++) {
        ull start = h_offset[u];
        ull end = h_offset[u + 1];

        for (ull j = start; j < end; j++) {
            int v = h_array[j];
            if (u >= v) continue;

            if (deg_[u] < deg_[v]) {
                new_adj_list[u].push_back(v);
            }
            else if (deg_[u] > deg_[v]) {
                new_adj_list[v].push_back(u);
            }
            else {
                if (u < v) {
                    new_adj_list[u].push_back(v);
                }
                else {
                    new_adj_list[v].push_back(u);
                }
            }
        }
    }

    for (int i = 0; i < vcount_; i++) {
        std::sort(new_adj_list[i].begin(), new_adj_list[i].end());
    }

    delete[] h_offset;
    delete[] h_array;

    h_offset = new ull[vcount_ + 1];
    h_offset[0] = 0;
    for (int i = 0; i < vcount_; i++) {
        h_offset[i + 1] = h_offset[i] + new_adj_list[i].size();
    }

    ull dag_edge_count = h_offset[vcount_];
    h_array = new int[dag_edge_count];
    ull edge_index = 0;
    for (int i = 0; i < vcount_; i++) {
        for (int neighbor : new_adj_list[i]) {
            h_array[edge_index++] = neighbor;
        }
    }

    // We keep the original degree as this will ensure correct candidate generation
    
    ecount_ = dag_edge_count;
    printf("Converted to DAG: |V|=%d |E|=%u\n", vcount_, ecount_);
}


void
Graph::save_largest_component(const char* input_filename, const char* output_filename) {
    std::string string_input_filename = input_filename;
    Graph original_graph(string_input_filename, true);

    ConnectedComponentFinder finder(&original_graph);
    finder.find_components();
    std::vector<int> largest_component = finder.get_largest_component();

    printf("Largest component size: %zu vertices\n", largest_component.size());

    std::vector<int> vertex_map(original_graph.vcount_, -1);
    for (size_t i = 0; i < largest_component.size(); i++) {
        vertex_map[largest_component[i]] = i;
    }

    ull new_ecount = 0;
    for (int old_vertex : largest_component) {
        ull start = original_graph.h_offset[old_vertex];
        ull end = original_graph.h_offset[old_vertex + 1];

        for (ull j = start; j < end; j++) {
            int neighbor = original_graph.h_array[j];
            if (vertex_map[neighbor] != -1) {
                new_ecount++;
            }
        }
    }

    new_ecount /= 2;

    int new_vcount = largest_component.size();
    ull* new_offset = new ull[new_vcount + 1];
    int* new_array = new int[2 * new_ecount];

    new_offset[0] = 0;
    for (int i = 0; i < new_vcount; i++) {
        int old_vertex = largest_component[i];
        ull start = original_graph.h_offset[old_vertex];
        ull end = original_graph.h_offset[old_vertex + 1];

        int edge_count = 0;
        for (ull j = start; j < end; j++) {
            int neighbor = original_graph.h_array[j];
            if (vertex_map[neighbor] != -1) {
                edge_count++;
            }
        }
        new_offset[i + 1] = new_offset[i] + edge_count;
    }

    std::vector<int> current_pos(new_vcount, 0);
    for (int i = 0; i < new_vcount; i++) {
        int old_vertex = largest_component[i];
        ull start = original_graph.h_offset[old_vertex];
        ull end = original_graph.h_offset[old_vertex + 1];

        for (ull j = start; j < end; j++) {
            int old_neighbor = original_graph.h_array[j];
            if (vertex_map[old_neighbor] != -1) {
                int new_neighbor = vertex_map[old_neighbor];
                ull pos = new_offset[i] + current_pos[i];
                new_array[pos] = new_neighbor;
                current_pos[i]++;
            }
        }
    }

    std::vector<int> new_vertex_label(new_vcount);
    for (int i = 0; i < new_vcount; i++) {
        new_vertex_label[i] = original_graph.vertex_label_[largest_component[i]];
    }

    FILE* file = fopen(output_filename, "wb");
    if (!file) {
        fprintf(stderr, "Error opening output file: %s\n", output_filename);
        exit(EXIT_FAILURE);
    }

    fwrite(&new_vcount, sizeof(int), 1, file);
    fwrite(&new_ecount, sizeof(unsigned int), 1, file);

    fwrite(new_offset, sizeof(ull), new_vcount + 1, file);

    fwrite(new_vertex_label.data(), sizeof(int), new_vcount, file);

    fwrite(new_array, sizeof(int), 2 * new_ecount, file);

    fclose(file);

    delete[] new_offset;
    delete[] new_array;

    printf("Largest component saved to %s\n", output_filename);
    printf("New graph: |V|=%d |E|=%llu\n", new_vcount, new_ecount);
}
