cmake_minimum_required(VERSION 3.26)
project(SubgraphMatching LANGUAGES CUDA CXX)
set(CMAKE_VERBOSE_MAKEFILE on)

set(CMAKE_CXX_COMPILER g++)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -pthread -march=native")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

add_library(graph_objects SHARED graph.cu graph.h)
target_include_directories(graph_objects PUBLIC ${CUDA_INCLUDE_DIRS})

add_library(join_objects SHARED join_bfs.cu join_bfs_dfs.cu join.h)
target_include_directories(join_objects PUBLIC ${CUDA_INCLUDE_DIRS})

set_target_properties(graph_objects join_objects PROPERTIES
    CUDA_ARCHITECTURES "70"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(SubgraphMatching main.cu)
target_link_libraries(SubgraphMatching PUBLIC graph_objects join_objects ${CUDA_LIBRARIES})