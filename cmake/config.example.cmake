set(ATHENA_VERSION "all")

set(CUDAToolkit_ROOT /usr/local/cuda)

set(NCCL_ROOT)

set(CUDNN_ROOT)

set(MPI_HOME /usr/local/mpi)

# if MPI_AUTO_DOWNLOAD is on, we will not try to locate MPI
# instead we will download and compile it in time (openmpi-4.0.3)
set(MPI_AUTO_DOWNLOAD ON)