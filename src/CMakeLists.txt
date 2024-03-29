include_directories(header common memory_pool)
include_directories(SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

aux_source_directory(common COMMON_SRC)
aux_source_directory(cuda_common CUDA_COMMON_SRC)
aux_source_directory(ops OPS_SRC)
aux_source_directory(dnnl_ops DNNL_OPS_SRC)

add_library(c_runtime_api SHARED)

add_custom_target(athena DEPENDS c_runtime_api)

if(${ATHENA_VERSION} STREQUAL "all")
    message(STATUS "athena mkl+gpu version")
    find_package(CUDNN REQUIRED)
    add_definitions(-DDEVICE_GPU)
    FetchContent_Declare(mkl URL https://github.com/intel/mkl-dnn/archive/v1.6.1.tar.gz)
    FetchContent_MakeAvailable(mkl)
    target_sources(c_runtime_api PUBLIC ${COMMON_SRC} ${CUDA_COMMON_SRC} ${OPS_SRC} ${DNNL_OPS_SRC})
    target_link_libraries(c_runtime_api cudart cublas cusparse curand)
    target_link_libraries(c_runtime_api ${CUDNN_LIBRARY_PATH})
    target_link_libraries(c_runtime_api dnnl)
    target_include_directories(c_runtime_api PUBLIC ${CUDNN_INCLUDE_PATH})
    target_include_directories(c_runtime_api PUBLIC cuda_common)
    set_property(TARGET c_runtime_api PROPERTY CUDA_ARCHITECTURES 30 35 50 52)
elseif(${ATHENA_VERSION} STREQUAL "gpu")
    message(STATUS "athena gpu version")
    find_package(CUDNN REQUIRED)
    add_definitions(-DDEVICE_GPU)
    target_sources(c_runtime_api PUBLIC ${COMMON_SRC} ${CUDA_COMMON_SRC} ${OPS_SRC})
    target_link_libraries(c_runtime_api cudart cublas cusparse curand)
    target_link_libraries(c_runtime_api ${CUDNN_LIBRARY_PATH})
    target_include_directories(c_runtime_api PUBLIC ${CUDNN_INCLUDE_PATH})
    target_include_directories(c_runtime_api PUBLIC cuda_common)
    set_property(TARGET c_runtime_api PROPERTY CUDA_ARCHITECTURES 30 35 50 52)
elseif(${ATHENA_VERSION} STREQUAL "mkl")
    message(STATUS "athena mkl version")
    FetchContent_Declare(mkl URL https://github.com/intel/mkl-dnn/archive/v1.6.1.tar.gz)
    message(STATUS "Preparing mkl-dnn ...")
    FetchContent_MakeAvailable(mkl)
    target_sources(c_runtime_api PUBLIC ${COMMON_SRC} ${DNNL_OPS_SRC})
    target_link_libraries(c_runtime_api dnnl)
else()
    message(FATAL_ERROR "unknown athena version")
endif()

add_subdirectory(communication)
add_subdirectory(hetu)
