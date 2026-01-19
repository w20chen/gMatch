#!/usr/bin/env python3
import struct
import os
import sys
from pathlib import Path
from collections import defaultdict

def check_csr_file(filename, check_undirected=True):
    print(f"Checking file: {filename}")
    print("=" * 60)
    
    try:
        file_size = os.path.getsize(filename)
        print(f"File size: {file_size} bytes")
        
        with open(filename, 'rb') as f:
            if file_size < 8:
                print("❌ Error: File too small to read number of vertices and edges")
                return False
            
            try:
                n_vertices, n_edges = struct.unpack('<II', f.read(8))
                print(f"Number of vertices |V|: {n_vertices}")
                print(f"Number of edges |E|: {n_edges}")
                
                if n_vertices == 0 and n_edges > 0:
                    print("❌ Error: Number of vertices is 0 but number of edges is not 0")
                    return False
                    
            except struct.error as e:
                print(f"❌ Error: Failed to read number of vertices and edges - {e}")
                return False
            
            expected_size = (
                8 +  # |V| and |E|
                (n_vertices + 1) * 8 +  # offset array
                n_vertices * 4 +  # vertex label array
                2 * n_edges * 4  # edge data array
            )
            
            print(f"Expected file size: {expected_size} bytes")
            print(f"Actual file size: {file_size} bytes")
            
            if file_size != expected_size:
                print(f"❌ Error: File size mismatch!")
                print(f"   Expected: {expected_size} bytes")
                print(f"   Actual: {file_size} bytes")
                return False
            
            try:
                offsets = struct.unpack(f'<{n_vertices + 1}Q', f.read((n_vertices + 1) * 8))
                print(f"Offset array size: {len(offsets)}")
                
                if offsets[0] != 0:
                    print("❌ Error: First element of offset array must be 0")
                    return False
                
                if offsets[-1] != 2 * n_edges:
                    print(f"❌ Error: Last element of offset array must equal 2*|E| = {2 * n_edges}, but got {offsets[-1]}")
                    return False
                
                for i in range(1, len(offsets)):
                    if offsets[i] < offsets[i - 1]:
                        print(f"❌ Error: Offset array decreases at position {i-1}->{i}: {offsets[i-1]} -> {offsets[i]}")
                        return False
                
                for i, offset in enumerate(offsets):
                    if offset > 2 * n_edges:
                        print(f"❌ Error: Offset value {offset} for vertex {i} out of range [0, {2 * n_edges}]")
                        return False
                
                print("✅ Offset array check passed")
                
            except struct.error as e:
                print(f"❌ Error: Failed to read offset array - {e}")
                return False
            
            try:
                vertex_labels = struct.unpack(f'<{n_vertices}I', f.read(n_vertices * 4))
                print(f"Vertex label array size: {len(vertex_labels)}")
                print("✅ Vertex label array check passed")
                
            except struct.error as e:
                print(f"❌ Error: Failed to read vertex label array - {e}")
                return False
            
            try:
                edges = struct.unpack(f'<{2 * n_edges}I', f.read(2 * n_edges * 4))
                print(f"Edge data array size: {len(edges)} elements")
                
                self_loops = 0
                duplicate_edges = 0
                unsorted_neighbors = 0
                
                adjacency_list = [[] for _ in range(n_vertices)]
                
                for vertex_idx in range(n_vertices):
                    start_idx = offsets[vertex_idx]
                    end_idx = offsets[vertex_idx + 1]
                    
                    neighbors = []
                    for edge_idx in range(start_idx, end_idx, 1):
                        if edge_idx >= len(edges):
                            print(f"❌ Error: Edge index {edge_idx} out of edge array range")
                            return False
                        
                        dest_vertex = edges[edge_idx]
                        neighbors.append(dest_vertex)
                        adjacency_list[vertex_idx].append(dest_vertex)
                        
                        if dest_vertex >= n_vertices:
                            print(f"❌ Error: Edge from vertex {vertex_idx} points to invalid vertex {dest_vertex}")
                            return False
                        
                        if dest_vertex == vertex_idx:
                            self_loops += 1
                            if self_loops <= 5:  # Show only first 5 self-loops
                                print(f"⚠️  Warning: Vertex {vertex_idx} has a self-loop")
                    
                    if len(neighbors) > 1:
                        for i in range(1, len(neighbors)):
                            if neighbors[i] < neighbors[i - 1]:
                                unsorted_neighbors += 1
                                if unsorted_neighbors <= 3:  # Show only first 3 sorting errors
                                    print(f"⚠️  Warning: Neighbors of vertex {vertex_idx} are not sorted: {neighbors[i-1]} -> {neighbors[i]}")
                                break
                    
                    neighbor_set = set()
                    for neighbor in neighbors:
                        if neighbor in neighbor_set:
                            duplicate_edges += 1
                            if duplicate_edges <= 5:  # Show only first 5 duplicate edges
                                print(f"⚠️  Warning: Vertex {vertex_idx} has duplicate edge to vertex {neighbor}")
                        neighbor_set.add(neighbor)
                
                asymmetric_edges = 0
                if check_undirected:
                    print("\nChecking undirected graph symmetry...")
                    for u in range(n_vertices):
                        for v in adjacency_list[u]:
                            if u == v:
                                continue
                            
                            if u not in adjacency_list[v]:
                                asymmetric_edges += 1
                                if asymmetric_edges <= 10:  # Show only first 10 asymmetric edges
                                    print(f"❌ Undirected graph is asymmetric: edge {u}->{v} exists but reverse edge {v}->{u} does not")
                    
                    if asymmetric_edges > 0:
                        print(f"❌ Found {asymmetric_edges} asymmetric edges")
                    else:
                        print("✅ Undirected graph symmetry check passed")
                
                print("\n" + "=" * 40)
                print("Graph structure statistics:")
                print(f"Total vertices: {n_vertices}")
                print(f"Total edges: {n_edges}")
                
                if self_loops > 0:
                    print(f"❌ Number of self-loops: {self_loops}")
                else:
                    print("✅ No self-loops")
                
                if duplicate_edges > 0:
                    print(f"❌ Number of duplicate edges: {duplicate_edges}")
                else:
                    print("✅ No duplicate edges")
                
                if unsorted_neighbors > 0:
                    print(f"❌ Number of vertices with unsorted neighbors: {unsorted_neighbors}")
                else:
                    print("✅ All vertices have sorted neighbors")
                
                if check_undirected:
                    if asymmetric_edges > 0:
                        print(f"❌ Number of asymmetric edges: {asymmetric_edges}")
                    else:
                        print("✅ Graph is undirected (all edges have corresponding reverse edges)")
                
                total_edges_from_offsets = 0
                for i in range(n_vertices):
                    start_idx = offsets[i]
                    end_idx = offsets[i + 1]
                    total_edges_from_offsets += (end_idx - start_idx)
                
                if total_edges_from_offsets != 2 * n_edges:
                    print(f"❌ Error: Edge count computed from offset array ({total_edges_from_offsets}) does not equal declared edge count ({2 * n_edges})")
                    return False
                
                print("✅ Edge data basic check passed")
                
            except struct.error as e:
                print(f"❌ Error: Failed to read edge data array - {e}")
                return False
            
            remaining_data = f.read()
            if remaining_data:
                print(f"❌ Warning: File has {len(remaining_data)} bytes of extra data at the end")
                return False
            
            print("\n" + "=" * 60)
            issues_found = (self_loops > 0 or duplicate_edges > 0 or unsorted_neighbors > 0 or 
                          (check_undirected and asymmetric_edges > 0))
            
            if not issues_found:
                print("🎉 All checks passed! File format is completely correct")
                if check_undirected:
                    print("✅ Graph is strictly undirected (no self-loops, no duplicate edges, neighbors sorted, fully symmetric)")
                return True
            else:
                print("⚠️  Basic format is correct, but the following issues were found:")
                if self_loops > 0:
                    print(f"   - Found {self_loops} self-loops")
                if duplicate_edges > 0:
                    print(f"   - Found {duplicate_edges} duplicate edges")
                if unsorted_neighbors > 0:
                    print(f"   - {unsorted_neighbors} vertices have unsorted neighbors")
                if check_undirected and asymmetric_edges > 0:
                    print(f"   - Found {asymmetric_edges} asymmetric edges (not a strictly undirected graph)")
                return False
            
    except FileNotFoundError:
        print(f"❌ Error: File {filename} does not exist")
        return False
    except Exception as e:
        print(f"❌ Error: Exception occurred while reading file - {e}")
        return False

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python check_dataset.py <csr_file> [--skip-undirected-check]")
        print("Example: python check_dataset.py graph.csr")
        print("         python check_dataset.py graph.csr --skip-undirected-check")
        sys.exit(1)
    
    filename = sys.argv[1]
    check_undirected = True
    
    if len(sys.argv) == 3 and sys.argv[2] == "--skip-undirected-check":
        check_undirected = False
        print("Note: Skipping undirected graph symmetry check")
    
    if not Path(filename).exists():
        print(f"Error: File '{filename}' does not exist")
        sys.exit(1)
    
    success = check_csr_file(filename, check_undirected)
    
    print("=" * 60)
    if success:
        print("🎉 File format is completely correct!")
        sys.exit(0)
    else:
        print("💥 File format has issues!")
        sys.exit(1)

if __name__ == "__main__":
    main()