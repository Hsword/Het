""" library to take autodiff and execute a computation graph """
from __future__ import absolute_import
import numpy as np
# import scipy.sparse
from scipy.sparse import spmatrix, coo_matrix
from .. import ndarray
from .._base import DNNL_LIB
from ..cpu_links import array_set as cpu_array_set
from .Variable import PlaceholderOp  # add for optimizer
from ..dataloader import DataloaderOp, GNNDataLoaderOp
from .AllReduceCommunicate import AllReduceCommunicateOp
from .ParameterServerCommunicate import ParameterServerCommunicateOp, ParameterServerSparsePullOp, parameterServerSparsePull_op
from .DataTransfer import DataH2DOp, DataD2HOp, DataD2HSparseOp
from .EmbeddingLookUp import EmbeddingLookUp, EmbeddingLookUp_Gradient
from . import OnesLike
from ..stream import *
from ..communicator.mpi_nccl_comm import ncclDataType_t, ncclRedOp_t, mpi_nccl_communicator
from operator import add
from functools import reduce
import ctypes
import os
from time import time
FLAG_SHOW_GRAPH = False
G_NODE_ID = 0


def path_to_lib(name):
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    lib_path = os.path.join(curr_path, '../../../build/lib/')
    return os.path.join(lib_path, name)

def mpi_nccl_init():
    global nccl_comm
    nccl_comm = mpi_nccl_communicator()
    nccl_comm.ncclInit()
    device_id = nccl_comm.device_id.value
    return nccl_comm, device_id

def mpi_nccl_finish(comm = None):
    comm.ncclFinish()

def get_nccl_communicate():
    global nccl_comm
    return nccl_comm

def get_worker_communicate():
    global ps_comm
    return ps_comm

def worker_init():
    global ps_comm
    ll = ctypes.cdll.LoadLibrary
    ps_comm = ll(path_to_lib("libps.so"))
    ps_comm.Init()
    os.environ['HEAPPROFILE'] = "./W" + str(ps_comm.rank())

def worker_finish():
    ps_comm.Finalize()

def server_init():
    global ps_comm
    ll = ctypes.cdll.LoadLibrary
    ps_comm = ll(path_to_lib("libps.so"))
    ps_comm.Init()
    ps_comm.StartServer()
    os.environ['HEAPPROFILE'] = "./S"+ str(ps_comm.rank())

def server_finish():
    ps_comm.Finalize()

def scheduler_init():
    global ps_comm
    ll = ctypes.cdll.LoadLibrary
    ps_comm = ll(path_to_lib("libps.so"))
    ps_comm.Init()

def scheduler_finish():
    ps_comm.Finalize()

class AthenaConfig(object):
    __slots__ = [
        'eval_node_list',
        'context',
        'seed',
        'np_rand',
        'comm_mode',
        'stream_mode',
        'ps_comm',
        'nccl_comm',
        'ctx_infer_mode',
        'worker_id',
        'worker_num',
        'comp_stream',
        'nccl_stream',
        'h2d_stream',
        'd2h_stream',
        'h2d_ops',
        'd2h_ops',
        'ps_map',
        'dataloader_name',
        'dataloader_ops',
        'use_sparse_pull',
        'cstable_policy',
        'inference',
        'enable_lazy',
        'bsp',
        'prefetch',
        'cache_bound',
        'log_path',
    ]

    def __init__(
        self,
        eval_node_list,
        ctx=ndarray.cpu(0),
        seed=None,
        comm_mode=None,
        stream_mode='AllStreams',
        ctx_infer_mode='use_default',
        dataloader_name='',
        use_sparse_pull=False,
        cstable_policy=None,
        inference = False,
        bsp=False,
        prefetch=True,
        enable_lazy=True,
        cache_bound=100,
        log_path=None,
    ):
        '''
        context: default device context
        comm_mode: communication mode, should be one of the following
            None       -> Single GPU
            PS         -> Parameter Server
            AllRedeuce -> MPI AllReduce
            Hybrid     -> Parameter Server for Sparse Parameter and MPI AllReduce for Dense Parameter
        stream_mode: None or ComputeStream or AllStreams
            None          -> do not use any streams (deprecated, bugs exist)
            ComputeStream -> only use stream for computation (deprecated, bugs exist)
            AllStreams    -> use 3 streams for h2d, d2h and computation
            streams should be used only when is_gpu_ctx(context) is True
        ctx_infer_mode: use_default or from_prev for nodes that not specified context
            use_default   -> use default context
            from_prev     -> use inputs nodes' context if possible, else use default
        '''
        self.eval_node_list = eval_node_list

        # check context
        assert ctx, 'Default context should be determined.'
        self.context = ctx

        # variables initialization
        self.seed = seed if seed else np.int64(time())
        self.np_rand = np.random.RandomState(self.seed)

        # get attribute of communication mode
        self.comm_mode = comm_mode
        self.ps_comm = None
        self.nccl_comm = None
        if self.comm_mode == 'PS' or self.comm_mode == 'Hybrid':
            worker_init()
            self.ps_comm = get_worker_communicate()
            self.worker_id = os.getenv('HEAPPROFILE')
            self.worker_num = int(os.environ['DMLC_NUM_WORKER']) if 'DMLC_NUM_WORKER' in os.environ else 1
        self.nccl_stream = None
        if self.comm_mode == "Hybrid" or self.comm_mode == "AllReduce":
            if ndarray.is_gpu_ctx(ctx):
                self.nccl_stream = create_stream_handle(ctx)
            self.nccl_comm = get_nccl_communicate()

        # check stream mode
        if stream_mode is not None:
            if not ndarray.is_gpu_ctx(ctx):
                stream_mode = None
            assert stream_mode in (None, 'ComputeStream', 'AllStreams'), \
                'Stream mode should be None, ComputeStream or AllStreams'
        self.stream_mode = stream_mode
        # define streams
        self.comp_stream = None if stream_mode is None else create_stream_handle(ctx)
        if stream_mode == 'AllStreams':
            self.h2d_stream = create_stream_handle(ctx)
            self.d2h_stream = create_stream_handle(ctx)
        else:
            self.h2d_stream = None
            self.d2h_stream = None

        # check ctx infer mode
        assert ctx_infer_mode in ('from_prev', 'use_default'), \
            'Context inference mode should be from_prev or use_default.'
        self.ctx_infer_mode = ctx_infer_mode

        self.use_sparse_pull = use_sparse_pull if self.comm_mode == 'PS' or self.comm_mode == "Hybrid" else False
        self.cstable_policy = cstable_policy if self.comm_mode == 'PS' or self.comm_mode == "Hybrid" else None
        self.prefetch = prefetch if self.comm_mode == 'PS' or self.comm_mode == 'Hybrid' else False
        if self.cstable_policy is not None:
            self.cstable_policy = self.cstable_policy.upper()
            self.use_sparse_pull = False

        self.h2d_ops = {}
        self.d2h_ops = {}
        self.ps_map = {}
        self.dataloader_name = dataloader_name
        self.inference = inference
        self.enable_lazy = (not inference) and enable_lazy # in inference(actually in PS) now we don't use lazy
        self.bsp = bsp
        self.cache_bound = int(cache_bound)

        self.log_path = log_path
        if log_path is not None and (self.comm_mode == 'PS' or self.comm_mode == "Hybrid"):
            assert os.path.isdir(log_path), 'Need to specify a work directory to save logs.'
            self.ps_comm.startRecord(ctypes.c_char_p(bytes(log_path, 'utf-8')))


class Executor(object):
    """Executor computes values for given set of nodes in computation graph."""

    def __init__(self, eval_node_list, config=None, **kargs):
        """
        Parameters
        ----------
        eval_node_list: list of nodes whose values need to be computed.
        topo_order: list of nodes in topological order
        node_to_shape_map: dict from node to shape of the node
        node_to_arr_map: dict from node to ndarray.NDArray allocated for node
        feed_shapes: shapes of feed_dict from last run(...)
        """
        if config is None:
            config = AthenaConfig(eval_node_list=eval_node_list, **kargs)
        assert isinstance(config, AthenaConfig), 'Config type %s invalid.' % str(type(config))
        self.eval_node_list = eval_node_list
        self.config = config

        # In this topo sort, the backward_hook will be called in backward phase;
        # when previous nodes finish, the forward hook will be called.
        # Can be used to add ops (if added in backward_hook, the added ops will be searched; not true in forward_hook).
        # Can be used to determine context (now in forward_hook).
        # Now the data transfer ops are added in forward_hook, the communicator ops (ps, allreduce) are added in backward_hook.

        if config.inference == False:
            topo_sort_with_hook(self.eval_node_list, self.config)
            # the real topo order, considering all ops
            self.topo_order = find_topo_sort(self.eval_node_list)
        else: # in inference phase
            if self.config.use_sparse_pull == True or self.config.cstable_policy is not None:
                # topo_sort_with_hook(self.eval_node_list, self.config)
                # insert ps_sparse_pull_op
                self.topo_order = find_topo_sort_inference(self.eval_node_list)
                # fetch sparse parameter
                fetch_sparse_parameter_value(self.topo_order, self.config)
            else:
                self.topo_order = find_topo_sort(self.eval_node_list)
            # fetch dense parameter
            # fetch_dense_parameter_value(self.topo_order, self.config)
        # main structures, nodes' shapes and arrays
        self.node_to_shape_map = {}
        self.node_to_arr_map = {}

        # inherit from configurations
        self.comm_mode = self.config.comm_mode
        self.ps_comm = self.config.ps_comm
        self.nccl_comm = self.config.nccl_comm
        self.comp_stream = self.config.comp_stream
        self.h2d_stream = self.config.h2d_stream
        self.d2h_stream = self.config.d2h_stream
        self.nccl_stream = self.config.nccl_stream
        self.param_psval_map = self.config.ps_map
        self.dataloader_name = self.config.dataloader_name
        self.use_sparse_pull = self.config.use_sparse_pull
        self.cstable_policy = self.config.cstable_policy

        # assisting structures, improve performance
        self.need_feed_nodes = []
        self.param_nodes = []
        self.dataloader_nodes = []
        self.computing_nodes = []
        for node in self.topo_order:
            if isinstance(node, DataloaderOp) or isinstance(node , GNNDataLoaderOp):
                self.dataloader_nodes.append(node)
            elif isinstance(node, PlaceholderOp):
                if node.shape is None:
                    self.need_feed_nodes.append(node)
                elif node.trainable:
                    self.param_nodes.append(node)
            elif not ((self.use_sparse_pull or self.cstable_policy) and isinstance(node, EmbeddingLookUp) and self.config.prefetch):
                self.computing_nodes.append(node)
        self.batch_num = set([node.get_batch_num(self.dataloader_name) for node in self.dataloader_nodes])
        assert len(self.batch_num) <= 1, 'Batch num not conform.'
        self.batch_num = None if len(self.batch_num) == 0 else self.batch_num.pop()
        self.init_need_allocation = (self.need_feed_nodes == []) and (self.dataloader_nodes == [])

    def infer_shape(self, feed_shapes):
        """Given shapes of feed_dict nodes, infer shape for all nodes in graph.

        Implementation note:
        Iteratively calls node.infer_shape to infer shapes.
        Node shapes stored in self.node_to_shape_map.

        Parameters
        ----------
        feed_shapes: node->shapes mapping for feed_dict nodes.
        """
        self.node_to_shape_map = {}
        for node in self.topo_order:
            if node in feed_shapes:
                self.node_to_shape_map[node] = tuple(feed_shapes[node])
            else:
                input_shapes = [self.node_to_shape_map[n] for n in node.inputs]
                cur_shape = node.infer_shape(input_shapes)
                self.node_to_shape_map[node] = cur_shape if cur_shape is None else tuple(cur_shape)
            # print(node.name, self.node_to_shape_map[node])

    def memory_plan(self):
        """Allocates ndarray.NDArray for every node except feed_dict nodes.
        Parameters
        ----------
        """
        for node, shape in self.node_to_shape_map.items():
            if isinstance(node, PlaceholderOp):
                if node.tensor_value is not None:
                    self.node_to_arr_map[node] = node.tensor_value
                elif node not in self.node_to_arr_map:
                    self.node_to_arr_map[node] = None
            elif not isinstance(node, DataloaderOp) and not isinstance(node, GNNDataLoaderOp):
                # add for OptimizerOp and ParameterServerOp
                if shape is None:
                    self.node_to_arr_map[node] = None
                    continue
                if isinstance(node, (EmbeddingLookUp_Gradient, DataD2HSparseOp)):
                    self.node_to_arr_map[node] = ndarray.IndexedSlices(dense_shape=shape)
                    continue
                if isinstance(node, EmbeddingLookUp) and (self.use_sparse_pull or self.cstable_policy) and self.config.prefetch:
                    self.node_to_arr_map[node] = self.param_psval_map[node.inputs[0]]
                    continue
                if node.on_gpu:
                    if node.inplace:
                        self.node_to_arr_map[node] = ndarray.NDArray(None)
                    else:
                        self.node_to_arr_map[node] = ndarray.empty(shape, ctx=node.ctx)
                else:
                    self.node_to_arr_map[node] = ndarray.empty(shape, ctx=node.ctx)

    def run(self, feed_dict = {}, convert_to_numpy_ret_vals=False):
        """
        Parameters
        ----------
        feed_dict: a dictionary of node->np.ndarray supplied by user.
        convert_to_numpy_ret_vals: whether to convert ret vals to np.array

        Returns
        -------
        A list of values for nodes in eval_node_list. NDArray or np.ndarray.
        """
        assert len(feed_dict) == len(self.need_feed_nodes), 'Feed dict invalid.'
        feed_shapes = {}
        need_reallocation = self.init_need_allocation

        # get feed in values
        for node, value in feed_dict.items():
            assert node in self.need_feed_nodes, 'Only allow feed in PlaceholderOp with no values, here got %s:%s.' % (str(type(node)), node.name)
            local_shape = tuple(value.shape)
            local_realloc = node not in self.node_to_shape_map or \
                local_shape != self.node_to_shape_map[node]
            need_reallocation = need_reallocation or local_realloc
            if node.on_cpu:
                assert isinstance(value, (np.ndarray, spmatrix, ndarray.NDArray)), \
                    "feed_dict value type not supported"
                if isinstance(value, np.ndarray):
                    if local_realloc:
                        self.node_to_arr_map[node] = ndarray.empty(local_shape, ctx=node.ctx)
                    self.node_to_arr_map[node][:] = value
                else:
                    self.node_to_arr_map[node] = value
            else:
                if isinstance(value, np.ndarray):
                    if local_realloc:
                        self.node_to_arr_map[node] = ndarray.array(value, ctx=node.ctx)
                    else:
                        self.node_to_arr_map[node][:] = value
                elif isinstance(value, spmatrix):
                    value = coo_matrix(value)
                    value = ndarray.sparse_array(value.data,
                            (value.row, value.col), shape = local_shape, ctx=node.ctx)
                    self.node_to_arr_map[node] = value
                elif isinstance(value, ndarray.NDArray):
                    if value.ctx == node.ctx:
                        self.node_to_arr_map[node] = value
                    else:
                        if local_realloc:
                            self.node_to_arr_map[node] = ndarray.empty(local_shape, ctx=node.ctx)
                        else:
                            self.node_to_arr_map[node][:] = value
                elif isinstance(value, ndarray.ND_Sparse_Array):
                    self.node_to_arr_map[node] = value
                else:
                    assert False, "feed_dict value type not supported"
            feed_shapes[node] = local_shape

        # get dataloader values
        for node in self.dataloader_nodes:
            local_shape = node.get_cur_shape(self.dataloader_name)
            local_realloc = node not in self.node_to_shape_map or \
                    local_shape != self.node_to_shape_map[node]
            need_reallocation = need_reallocation or local_realloc
            self.node_to_arr_map[node] = node.get_arr(self.dataloader_name)
            feed_shapes[node] = local_shape

        # reallocation, infer shapes and allocate memory
        if need_reallocation:
            self.infer_shape(feed_shapes)
            self.memory_plan()

        # computing
        for node in self.computing_nodes:
            if node.on_cpu and isinstance(self.node_to_arr_map[node], ndarray.NDArray):
                if DNNL_LIB['cpu_ArraySet'] and not isinstance(node, DataD2HOp):
                    cpu_array_set(self.node_to_arr_map[node], 0.0)
                else:
                    # here we suppose not using DNNL_LIB
                    # self.node_to_arr_map[node][:] = np.zeros(self.node_to_shape_map[node]).astype(np.float32)
                    pass

            input_vals = [self.node_to_arr_map[n] for n in node.inputs]
            node_val = self.node_to_arr_map[node]

            for n in node.inputs:
                if n.event:
                    n.event.sync()

            if isinstance(node, (ParameterServerCommunicateOp, ParameterServerSparsePullOp)):
                # Here we use d2h stream in ps op, since the stream is used for d2h data transfer.
                # Please take care at this part.
                node.compute(input_vals, node_val, self.d2h_stream)

            elif isinstance(node, AllReduceCommunicateOp):
                node.compute(input_vals, node_val, self.nccl_comm, self.nccl_stream)

            elif isinstance(node, DataH2DOp):
                node.compute(input_vals, node_val, self.h2d_stream)

            elif isinstance(node, (DataD2HOp, DataD2HSparseOp)):
                node.compute(input_vals, node_val, self.d2h_stream)

            else:
                node.compute(input_vals, node_val, self.comp_stream)
                if isinstance(node.event, Event):
                    # for d2h op / eval nodes / nodes before allreduce or ps nodes
                    node.event.record(self.comp_stream)
        for n in self.eval_node_list:
            # every node in eval_node_list should have an event (except dataloader/optimizer...)
            if n.event:
                n.event.sync()

        # get results
        results = [self.node_to_arr_map[n] for n in self.eval_node_list]
        if convert_to_numpy_ret_vals:
            for i in range(len(results)):
                if results[i] is not None:
                    results[i] = results[i].asnumpy()

        return results

    def save(self, file_path):
        assert os.path.isdir(file_path), 'Need to specify a work directory to save parameters.'
        if self.comm_mode in (None, 'AllReduce'):
            # when using allreduce, users need to specify the worker whose rank equals 0 to save
            for node in self.topo_order:
                if isinstance(node, PlaceholderOp) and node.trainable:
                    np.save(os.path.join(file_path, node.name + '.npy'), node.tensor_value.asnumpy())
        else:
            self.ps_comm.BarrierWorker()
            if self.config.worker_id == './W0':
                for node in self.topo_order:
                    if isinstance(node, PlaceholderOp) and node.trainable:
                        if node.is_embed or self.comm_mode == 'PS':
                            node.event.sync()
                            nodeid = ctypes.c_int(node.id)
                            self.ps_comm.SaveParam(nodeid, ctypes.c_char_p(bytes(file_path, 'utf-8')))
                            self.ps_comm.Wait(nodeid)
                        else:
                            np.save(os.path.join(file_path, node.name + '.npy'), node.tensor_value.asnumpy())
            self.ps_comm.BarrierWorker()

    def load(self, file_path):
        assert os.path.isdir(file_path), 'Need to specify a work directory to load parameters.'
        if self.comm_mode in (None, 'AllReduce'):
            for node in self.topo_order:
                if isinstance(node, PlaceholderOp) and node.trainable:
                    node.tensor_value[:] = np.load(os.path.join(file_path, node.name + '.npy'))
        else:
            self.ps_comm.BarrierWorker()
            if self.config.worker_id == './W0':
                for node in self.topo_order:
                    if isinstance(node, PlaceholderOp) and node.trainable:
                        if node.is_embed or self.comm_mode == 'PS':
                            node.event.sync()
                            nodeid = ctypes.c_int(node.id)
                            self.ps_comm.LoadParam(nodeid, ctypes.c_char_p(bytes(file_path, 'utf-8')))
                            node.event.update()
            self.ps_comm.BarrierWorker()
            for node in self.topo_order:
                if isinstance(node, PlaceholderOp) and node.trainable and not node.is_embed:
                    if self.comm_mode == 'PS':
                        node.event.sync()
                        nodeid = ctypes.c_int(node.id)
                        self.ps_comm.Pull(nodeid, self.param_psval_map[node].handle)
                        node.event.update()
                    else:
                        node.tensor_value[:] = np.load(os.path.join(file_path, node.name + '.npy'))
                elif isinstance(node, EmbeddingLookUp) and self.config.prefetch:
                    node.event.sync()
                    nodeid = ctypes.c_int(node.inputs[0].id)
                    self.ps_comm.SparsePull(nodeid, node.inputs[1].get_next_arr(self.dataloader_name).handle, self.param_psval_map[node.inputs[0]].handle)
                    node.event.update()
            self.ps_comm.BarrierWorker()

    def recordLoads(self):
        for node in self.param_psval_map:
            node.event.sync()
        self.ps_comm.getLoads()

    def __del__(self):
        if self.comp_stream is not None:
            self.comp_stream.sync()
        if self.h2d_stream is not None:
            self.h2d_stream.sync()
        if self.d2h_stream is not None:
            self.d2h_stream.sync()
        if self.nccl_stream is not None:
            self.nccl_stream.sync()
        for node in self.param_nodes:
            if node.event:
                node.event.sync()
        if self.comm_mode == 'PS' or self.comm_mode == 'Hybrid':
            worker_finish()


def gradients(output_node, node_list):
    """Take gradient of output node with respect to each node in node_list.

    Parameters
    ----------
    output_node: output node that we are taking derivative of.
    node_list: list of nodes that we are taking derivative wrt.

    Returns
    -------
    A list of gradient values, one for each node in node_list respectively.

    """
    node_to_output_grads_list = {}
    node_to_output_grads_list[output_node] = [OnesLike.oneslike_op(output_node)]
    node_to_output_grad = {}
    # Traverse forward graph in reverse topological order
    reverse_topo_order = reversed(find_topo_sort([output_node]))
    for node in reverse_topo_order:
        output_grad = sum_node_list(node_to_output_grads_list[node])
        if output_grad is None:
            for n in node.inputs:
                if n not in node_to_output_grads_list:
                    node_to_output_grads_list[n] = []
            continue
        node_to_output_grad[node] = output_grad
        input_grads_list = node.gradient(output_grad)
        for i in range(len(node.inputs)):
            if node.inputs[i] not in node_to_output_grads_list:
                node_to_output_grads_list[node.inputs[i]] = []
            # Calculate partial adjoint for input nodes.
            node_to_output_grads_list[node.inputs[i]].append(
                input_grads_list[i])

    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list

##################
# Helper Methods #
##################


def topo_sort_with_hook(node_list, config):
    visited = set()
    for node in node_list:
        topo_sort_dfs_with_hook(node, visited, config)


def topo_sort_dfs_with_hook(node, visited, config):
    if node in visited:
        return
    visited.add(node)
    node.backward_hook(config)
    for n in node.inputs:
        topo_sort_dfs_with_hook(n, visited, config)
    node.forward_hook(config)


def find_topo_sort(node_list):
    """Given a list of nodes, return a topo ordering of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a
    topological sort.

    """
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)

def find_topo_sort_inference(node_list):
    topo_order = find_topo_sort(node_list)
    embedding_list = list()
    embedding_outputs = dict()
    embedding_cnt = dict()
    for node in topo_order:
        if isinstance(node, EmbeddingLookUp):
            embedding_outputs[node] = list()
            embedding_cnt[node] = 0
            embedding_list.append(node)
        else:
            for input_node in node.inputs:
                if isinstance(input_node, EmbeddingLookUp):
                    embedding_outputs[input_node].append(node)
                    embedding_cnt[input_node] += 1
        # parameterServerSparsePull_op(embedding, *outputs)
    topo_order_inference = list()
    for node in topo_order:
        topo_order_inference.append(node)
        for embedding in embedding_list:
            if node in embedding_outputs[embedding]:
                embedding_cnt[embedding] -= 1
            if embedding_cnt[embedding] == 0:
                topo_order_inference.append(parameterServerSparsePull_op(embedding, embedding_outputs[embedding]))
                embedding_list.remove(embedding)

    return topo_order_inference


def fetch_sparse_parameter_value(node_list, config):
    for node in node_list:
        if isinstance(node, ParameterServerSparsePullOp):
            node.forward_hook(config)

def fetch_dense_parameter_value(node_list, config):
    assert config.comm_mode in ('PS', 'Hybrid')
    topo_order = find_topo_sort(node_list)
    val_list = []
    # get var list
    for node in topo_order:
        if isinstance(node, PlaceholderOp) and node.trainable:
            val_list.append(node)
    for node in val_list:
        if config.use_sparse_pull and node.is_embed:
            continue
        else:
            pull_val = ndarray.empty(node.shape, ctx=ndarray.cpu(0))
            config.ps_comm.Pull(node.id, pull_val.handle)
            config.ps_map[node] = pull_val
            node.tensor_value = pull_val
        node.event.update()

def sum_node_list(node_list):
    """Custom sum func to avoid creating redundant nodes in Python sum func."""
    node_list = [n for n in node_list if n is not None]
    if node_list == []:
        return None
    return reduce(add, node_list)


def broadcast_rule(shape_a, shape_b):
    """Return output shape of broadcast shape_a, shape_b.
    e.g. broadcast_rule((3,2), (4,3,2))
    returns output_shape = (4,3,2)

    Check out explanations and more examples at
    https://docs.scipy.org/doc/numpy-1.10.0/user/basics.broadcasting.html
    http://eli.thegreenplace.net/2015/broadcasting-arrays-in-numpy/
    """
    assert(isinstance(shape_a, tuple))
    assert(isinstance(shape_b, tuple))
    if len(shape_a) > len(shape_b):
        longer_shape, shorter_shape = shape_a, shape_b
    else:
        longer_shape, shorter_shape = shape_b, shape_a
    len_diff = len(longer_shape) - len(shorter_shape)
    for i in range(len_diff):
        # pad with leading 1s
        shorter_shape = (1,) + shorter_shape
    assert len(shorter_shape) == len(longer_shape)
    output_shape = list(longer_shape)
    for i in range(len(output_shape)):
        assert (shorter_shape[i] == longer_shape[i]) \
            or (shorter_shape[i] == 1) \
            or (longer_shape[i] == 1)
        output_shape[i] = max(shorter_shape[i], longer_shape[i])
    return tuple(output_shape)
