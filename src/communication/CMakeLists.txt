# compile nccl and mpi communication

find_package(NCCL REQUIRED)

if(MPI_AUTO_DOWNLOAD)
    FetchContent_Declare(openmpi URL https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.3.tar.gz)
    message(STATUS "Fetching openmpi source code ...")
    FetchContent_MakeAvailable(openmpi)
    set(MPI_CXX_LIBRARIES ${openmpi_BINARY_DIR}/lib/libmpi.so)
    set(MPI_CXX_INCLUDE_DIRS ${openmpi_BINARY_DIR}/include)
    add_custom_command(OUTPUT ${MPI_CXX_INCLUDE_DIRS}/mpi.h
        COMMAND ./configure --prefix=${openmpi_BINARY_DIR}
        COMMAND make -j8
        COMMAND make install
        COMMENT "configure, compile and install openmpi"
        WORKING_DIRECTORY ${openmpi_SOURCE_DIR}
    )
    add_custom_target(compile_openmpi DEPENDS ${MPI_CXX_INCLUDE_DIRS}/mpi.h)
else()
    find_package(MPI REQUIRED)
endif()

# compile nccl communication
add_library(_nccl_runtime SHARED nccl_communication.cu)
target_include_directories(_nccl_runtime PUBLIC ${NCCL_INCLUDE_DIRS})
target_link_libraries(_nccl_runtime PUBLIC ${NCCL_LIBRARIES})
add_custom_target(nccl DEPENDS _nccl_runtime)

# compile nccl+mpi communication
add_library(_mpi_nccl_runtime_api SHARED mpi_nccl_communication.cu)
target_include_directories(_mpi_nccl_runtime_api PUBLIC ${NCCL_INCLUDE_DIRS})
target_include_directories(_mpi_nccl_runtime_api PUBLIC ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(_mpi_nccl_runtime_api PUBLIC ${MPI_CXX_LIBRARIES})
target_link_libraries(_mpi_nccl_runtime_api PUBLIC ${NCCL_LIBRARIES})
add_custom_target(mpi_nccl DEPENDS _mpi_nccl_runtime_api)

# compile mpi communication
add_library(_mpi_runtime_api SHARED mpi_communication.cc)
target_include_directories(_mpi_runtime_api PUBLIC ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(_mpi_runtime_api PUBLIC ${MPI_CXX_LIBRARIES})
add_custom_target(mpi DEPENDS _mpi_runtime_api)

if(MPI_AUTO_DOWNLOAD)
    add_dependencies(_mpi_runtime_api compile_openmpi)
    add_dependencies(_mpi_nccl_runtime_api compile_openmpi)
endif()