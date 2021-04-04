from __future__ import absolute_import
from .Node import Op
from .. import ndarray
from .._base import _LIB, check_call
from ..stream import create_event_handle
from ..communicator.mpi_nccl_comm import ncclDataType_t,\
                                         ncclRedOp_t, mpi_nccl_communicator


class AllReduceCommunicateOp(Op):
    def __init__(self, nodeA):
        super().__init__(AllReduceCommunicateOp, [nodeA], nodeA.ctx)
        self.on_gpu = ndarray.is_gpu_ctx(self.ctx)
        self.on_cpu = not self.on_gpu

    def compute(self, input_vals, output_val, comm = None, stream_handle = None):
        if self.on_cpu:
            assert not isinstance(input_vals[0], (ndarray.IndexedSlices, ndarray.ND_Sparse_Array))
            comm.dlarrayNcclAllReduce(input_vals[0], ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclSum)
            input_vals[0].copyto(output_val)
        else:
            if self.event == None:
                self.event = create_event_handle(input_vals[0].ctx)
            if isinstance(input_vals[0], ndarray.NDArray):
                input_vals[0].copyto(output_val)
                comm.dlarrayNcclAllReduce(output_val, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclSum, stream_handle)
                self.event.record(stream_handle) 
            elif isinstance(input_vals[0], ndarray.IndexedSlices):
                input_vals[0].indices.copyto(output_val.indices)
                input_vals[0].values.copyto(output_val.values)
                comm.dlarrayNcclAllReduce(output_val.indices, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclSum, stream_handle)
                comm.dlarrayNcclAllReduce(output_val.values, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclSum, stream_handle)
                self.event.record(stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return input_shapes[0]
    
    def forward_hook(self, config):
        self.ctx = self.inputs[0].ctx
        self.on_gpu = ndarray.is_gpu_ctx(self.ctx)
        self.on_cpu = not self.on_gpu
        if self.on_gpu and self.inputs[0].event is None:
            self.inputs[0].event = create_event_handle(self.ctx)
            
        # disable inplace if not lazy execution
        # previously we use array reshape lazy callback to do this, which is deprecated (not efficient)
        self.inputs[0].inplace = False


def allreduceCommunicate_op(node):
    """Make a new instance of AllReduceCommunicateOp and call the instance.

    Parameters:
    ----
    node : Node
        The Node to do allreduce

    Returns:
    ----
    A new Node instance created by Op.

    """
    return AllReduceCommunicateOp(node)
