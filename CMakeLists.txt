cmake_minimum_required(VERSION 3.18)

project(Athena CXX)

include(cmake/config.cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_CXX_FLAGS "-O3 -Wall")

# openmp
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# cuda
set(CMAKE_CUDA_COMPILER ${CUDAToolkit_ROOT}/bin/nvcc)
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

include(FetchContent) # download third_party

add_subdirectory(${CMAKE_SOURCE_DIR}/ps-lite)
add_subdirectory(${CMAKE_SOURCE_DIR}/src)
add_subdirectory(${CMAKE_SOURCE_DIR}/geometric)