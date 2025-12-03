#!/usr/bin/env python3
import struct
import os
import sys
from pathlib import Path
from collections import defaultdict

def check_csr_file(filename, check_undirected=True):
    print(f"正在检测文件: {filename}")
    print("=" * 60)
    
    try:
        file_size = os.path.getsize(filename)
        print(f"文件大小: {file_size} 字节")
        
        with open(filename, 'rb') as f:
            if file_size < 8:
                print("❌ 错误: 文件太小，无法读取顶点数和边数")
                return False
            
            try:
                n_vertices, n_edges = struct.unpack('<II', f.read(8))
                print(f"顶点数 |V|: {n_vertices}")
                print(f"边数 |E|: {n_edges}")
                
                if n_vertices == 0 and n_edges > 0:
                    print("❌ 错误: 顶点数为0但边数不为0")
                    return False
                    
            except struct.error as e:
                print(f"❌ 错误: 读取顶点数和边数失败 - {e}")
                return False
            
            expected_size = (
                8 +  # |V| 和 |E|
                (n_vertices + 1) * 8 +  # 偏移数组
                n_vertices * 4 +  # 顶点标签数组
                2 * n_edges * 4  # 边数据数组
            )
            
            print(f"期望文件大小: {expected_size} 字节")
            print(f"实际文件大小: {file_size} 字节")
            
            if file_size != expected_size:
                print(f"❌ 错误: 文件大小不匹配!")
                print(f"   期望: {expected_size} 字节")
                print(f"   实际: {file_size} 字节")
                return False
            
            try:
                offsets = struct.unpack(f'<{n_vertices + 1}Q', f.read((n_vertices + 1) * 8))
                print(f"偏移数组大小: {len(offsets)}")
                
                if offsets[0] != 0:
                    print("❌ 错误: 偏移数组第一个元素必须为0")
                    return False
                
                if offsets[-1] != 2 * n_edges:
                    print(f"❌ 错误: 偏移数组最后一个元素必须等于 2*|E| = {2 * n_edges}, 实际为 {offsets[-1]}")
                    return False
                
                for i in range(1, len(offsets)):
                    if offsets[i] < offsets[i - 1]:
                        print(f"❌ 错误: 偏移数组在位置 {i-1}->{i} 处递减: {offsets[i-1]} -> {offsets[i]}")
                        return False
                
                for i, offset in enumerate(offsets):
                    if offset > 2 * n_edges:
                        print(f"❌ 错误: 顶点 {i} 的偏移值 {offset} 超出范围 [0, {2 * n_edges}]")
                        return False
                
                print("✅ 偏移数组检查通过")
                
            except struct.error as e:
                print(f"❌ 错误: 读取偏移数组失败 - {e}")
                return False
            
            try:
                vertex_labels = struct.unpack(f'<{n_vertices}I', f.read(n_vertices * 4))
                print(f"顶点标签数组大小: {len(vertex_labels)}")
                print("✅ 顶点标签数组检查通过")
                
            except struct.error as e:
                print(f"❌ 错误: 读取顶点标签数组失败 - {e}")
                return False
            
            try:
                edges = struct.unpack(f'<{2 * n_edges}I', f.read(2 * n_edges * 4))
                print(f"边数据数组大小: {len(edges)} 个元素")
                
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
                            print(f"❌ 错误: 边索引 {edge_idx} 超出边数组范围")
                            return False
                        
                        dest_vertex = edges[edge_idx]
                        neighbors.append(dest_vertex)
                        adjacency_list[vertex_idx].append(dest_vertex)
                        
                        if dest_vertex >= n_vertices:
                            print(f"❌ 错误: 顶点 {vertex_idx} 的边指向无效顶点 {dest_vertex}")
                            return False
                        
                        if dest_vertex == vertex_idx:
                            self_loops += 1
                            if self_loops <= 5:  # 只显示前5个自环
                                print(f"⚠️  警告: 顶点 {vertex_idx} 存在自环")
                    
                    if len(neighbors) > 1:
                        for i in range(1, len(neighbors)):
                            if neighbors[i] < neighbors[i - 1]:
                                unsorted_neighbors += 1
                                if unsorted_neighbors <= 3:  # 只显示前3个排序错误
                                    print(f"⚠️  警告: 顶点 {vertex_idx} 的邻居未排序: {neighbors[i-1]} -> {neighbors[i]}")
                                break
                    
                    neighbor_set = set()
                    for neighbor in neighbors:
                        if neighbor in neighbor_set:
                            duplicate_edges += 1
                            if duplicate_edges <= 5:  # 只显示前5个重边
                                print(f"⚠️  警告: 顶点 {vertex_idx} 存在重边指向顶点 {neighbor}")
                        neighbor_set.add(neighbor)
                
                asymmetric_edges = 0
                if check_undirected:
                    print("\n正在检查无向图对称性...")
                    for u in range(n_vertices):
                        for v in adjacency_list[u]:
                            if u == v:
                                continue
                            
                            if u not in adjacency_list[v]:
                                asymmetric_edges += 1
                                if asymmetric_edges <= 10:  # 只显示前10个不对称边
                                    print(f"❌ 无向图不对称: 存在边 {u}->{v} 但不存在反向边 {v}->{u}")
                    
                    if asymmetric_edges > 0:
                        print(f"❌ 发现 {asymmetric_edges} 个不对称边")
                    else:
                        print("✅ 无向图对称性检查通过")
                
                print("\n" + "=" * 40)
                print("图结构统计:")
                print(f"总顶点数: {n_vertices}")
                print(f"总边数: {n_edges}")
                
                if self_loops > 0:
                    print(f"❌ 自环数量: {self_loops}")
                else:
                    print("✅ 无自环")
                
                if duplicate_edges > 0:
                    print(f"❌ 重边数量: {duplicate_edges}")
                else:
                    print("✅ 无重边")
                
                if unsorted_neighbors > 0:
                    print(f"❌ 邻居未排序的顶点数: {unsorted_neighbors}")
                else:
                    print("✅ 所有顶点的邻居都已排序")
                
                if check_undirected:
                    if asymmetric_edges > 0:
                        print(f"❌ 不对称边数量: {asymmetric_edges}")
                    else:
                        print("✅ 图是无向的（所有边都有对应的反向边）")
                
                total_edges_from_offsets = 0
                for i in range(n_vertices):
                    start_idx = offsets[i]
                    end_idx = offsets[i + 1]
                    total_edges_from_offsets += (end_idx - start_idx)
                
                if total_edges_from_offsets != 2 * n_edges:
                    print(f"❌ 错误: 从偏移数组计算的边数 ({total_edges_from_offsets}) 不等于声明的边数 ({2 * n_edges})")
                    return False
                
                print("✅ 边数据基本检查通过")
                
            except struct.error as e:
                print(f"❌ 错误: 读取边数据数组失败 - {e}")
                return False
            
            remaining_data = f.read()
            if remaining_data:
                print(f"❌ 警告: 文件末尾还有 {len(remaining_data)} 字节的多余数据")
                return False
            
            print("\n" + "=" * 60)
            issues_found = (self_loops > 0 or duplicate_edges > 0 or unsorted_neighbors > 0 or 
                          (check_undirected and asymmetric_edges > 0))
            
            if not issues_found:
                print("🎉 所有检查通过! 文件格式完全正确")
                if check_undirected:
                    print("✅ 图是严格的无向图（无自环、无重边、邻居排序、完全对称）")
                return True
            else:
                print("⚠️  基本格式正确，但发现以下问题:")
                if self_loops > 0:
                    print(f"   - 存在 {self_loops} 个自环")
                if duplicate_edges > 0:
                    print(f"   - 存在 {duplicate_edges} 个重边")
                if unsorted_neighbors > 0:
                    print(f"   - 有 {unsorted_neighbors} 个顶点的邻居未排序")
                if check_undirected and asymmetric_edges > 0:
                    print(f"   - 存在 {asymmetric_edges} 个不对称边（不是严格的无向图）")
                return False
            
    except FileNotFoundError:
        print(f"❌ 错误: 文件 {filename} 不存在")
        return False
    except Exception as e:
        print(f"❌ 错误: 读取文件时发生异常 - {e}")
        return False

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("用法: python check_dataset.py <csr_file> [--skip-undirected-check]")
        print("示例: python check_dataset.py graph.csr")
        print("       python check_dataset.py graph.csr --skip-undirected-check")
        sys.exit(1)
    
    filename = sys.argv[1]
    check_undirected = True
    
    if len(sys.argv) == 3 and sys.argv[2] == "--skip-undirected-check":
        check_undirected = False
        print("注意: 跳过无向图对称性检查")
    
    if not Path(filename).exists():
        print(f"错误: 文件 '{filename}' 不存在")
        sys.exit(1)
    
    success = check_csr_file(filename, check_undirected)
    
    print("=" * 60)
    if success:
        print("🎉 文件格式完全正确!")
        sys.exit(0)
    else:
        print("💥 文件格式存在问题!")
        sys.exit(1)

if __name__ == "__main__":
    main()