from athena.communicator.mpi_nccl_comm import ncclDataType_t, ncclRedOp_t, mpi_nccl_communicator
from athena import ndarray
import numpy as np
import time 


def test_allreduce(comm = None):
    shape = (24, 24)
    size = 4
    for val in shape:
        size *= val
    input_arr = np.ones(shape)*comm.localRank.value
    input_arr = ndarray.array(input_arr, ctx = ndarray.gpu(comm.localRank.value))
    # input_arr = ndarray.array(input_arr, ctx = ndarray.cpu())
 
    start = time.time()
    comm.dlarrayNcclAllReduce(input_arr, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclSum)
    comm.stream.sync()
    end = time.time()

    secs = end - start

    return size, secs

def test_p2p(comm = None, src = 0, target = 1):
    shape = (1000, 30, 224, 224)  
    size = 4
    for val in shape:
        size *= val
    print("MyRank: ", comm.myRank.value)
    arr = np.ones(shape)*comm.localRank.value
    arr = ndarray.array(arr, ctx = ndarray.gpu(comm.localRank.value))
    # arr = ndarray.array(arr, ctx = ndarray.cpu())
    start = time.time()
    if comm.myRank.value == 0:
        comm.dlarraySend(arr, ncclDataType_t.ncclFloat32, 1)
    else:
        comm.dlarrayRecv(arr, ncclDataType_t.ncclFloat32, 0)    
    comm.stream.sync()
    end = time.time()

    secs = end - start
    # size: /Bytes
    # dur_time: /s
    return size, secs

# t.dlarrayBroadcast(arr, ncclDataType_t.ncclFloat32, 0)
# t.dlarrayAllGather(arr, output_arr, ncclDataType_t.ncclFloat32)
if __name__ == "__main__":
    comm = mpi_nccl_communicator()
    comm.ncclInit()
    # size, secs = test_p2p(comm)
    size, secs = test_allreduce(comm)
    print("band width: %.2f MB/s"%(size/(2**20)/secs))
    # test_allreduce(comm)
    comm.ncclFinish()
