cmake_minimum_required(VERSION 3.26)
set(CMAKE_CUDA_ARCHITECTURES "native")
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

add_library(graph_objects_unlabeled SHARED graph.cu graph.h)
target_compile_definitions(graph_objects_unlabeled PUBLIC UNLABELED)
target_include_directories(graph_objects_unlabeled PUBLIC ${CUDA_INCLUDE_DIRS})

add_library(graph_objects_labeled SHARED graph.cu graph.h)
target_include_directories(graph_objects_labeled PUBLIC ${CUDA_INCLUDE_DIRS})

add_library(join_objects_unlabeled SHARED candidate.cu candidate.h join_bfs.cu join_bfs_dfs.cu join_no_filtering.cu join_induced.cu join.h)
target_compile_definitions(join_objects_unlabeled PUBLIC UNLABELED)
target_include_directories(join_objects_unlabeled PUBLIC ${CUDA_INCLUDE_DIRS})

add_library(join_objects_labeled SHARED candidate.cu candidate.h join_bfs.cu join_bfs_dfs.cu join_no_filtering.cu join_induced.cu join.h)
target_include_directories(join_objects_labeled PUBLIC ${CUDA_INCLUDE_DIRS})

set_target_properties(graph_objects_unlabeled graph_objects_labeled join_objects_unlabeled join_objects_labeled PROPERTIES
    CUDA_ARCHITECTURES "native"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(SubgraphMatching_unlabeled main.cu)
target_link_libraries(SubgraphMatching_unlabeled PUBLIC graph_objects_unlabeled join_objects_unlabeled ${CUDA_LIBRARIES})

add_executable(SubgraphMatching_labeled main.cu)
target_link_libraries(SubgraphMatching_labeled PUBLIC graph_objects_labeled join_objects_labeled ${CUDA_LIBRARIES})