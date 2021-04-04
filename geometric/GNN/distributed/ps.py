import os
import ctypes
import numpy as np
import threading

from athena import ndarray

def ps_get_worker_communicator():
    import athena.gpu_ops as ad
    ad.get_worker_communicate()
    return ad.get_worker_communicate()

def pointer(arr):
    # Convert a numpy array (dtype=long) to raw C pointer
    assert(arr.data.c_contiguous)
    assert(arr.dtype == np.long)
    return ctypes.cast(arr.ctypes.data, ctypes.POINTER(ctypes.c_long))

class PS:
    rank = None
    nrank = None
    offset = 0
    feature_len = None
    communicator = None
    trace_file = None
    trace_len = None

def ps_init(rank, nrank):
    assert(rank >=0 and nrank >= 1)
    assert(rank < nrank)
    PS.rank = rank
    PS.nrank = nrank
    PS.communicator = ps_get_worker_communicator()

def ps_set_trace(trace_file):
    PS.trace_file = trace_file
    PS.trace_len = 0

#-------------------------------------------------------------------------------
# args can be both integer or array
def ps_node_id(node, node_from):
    return PS.nrank * node + node_from

def ps_node_feat_id(node, node_from):
    node_id = ps_node_id(node, node_from)
    return PS.offset + 2 * node_id

def ps_node_edge_id(node, node_from):
    node_id = ps_node_id(node, node_from)
    return PS.offset + 2 * node_id + 1

#-------------------------------------------------------------------------------
def ps_upload(x, y, indptr, indices, nodes_from):
    num_nodes = x.shape[0]
    PS.feature_len = x.shape[1]
    all_pushed_id = []
    #upload feature, label, degree (aggregated into one)
    degree = indptr[1:] - indptr[:-1]
    #feat_id_arr = np.empty(num_nodes, dtype=np.long)
    feat_length_arr = np.repeat(x.shape[1] + 2, num_nodes)
    feat_data_arr = np.concatenate([x, y.reshape(-1,1), degree.reshape(-1,1)], axis=1)
    feat_data_arr = ndarray.array(feat_data_arr, ctx = ndarray.cpu())
    #upload edge info (2 * degree)
    #edge_id_arr = np.empty(num_nodes, dtype=np.long)
    edge_length_arr = degree * 2
    edge_data_arr = np.concatenate([indices, nodes_from]).reshape(2, -1).T
    edge_data_arr = ndarray.array(np.ascontiguousarray(edge_data_arr), ctx = ndarray.cpu())

    feat_id_arr = ps_node_feat_id(np.arange(num_nodes), PS.rank)
    edge_id_arr = ps_node_edge_id(np.arange(num_nodes), PS.rank)
    query1 = PS.communicator.PushData(
        pointer(feat_id_arr),
        num_nodes,
        feat_data_arr.handle,
        pointer(feat_length_arr)
    )
    query2 = PS.communicator.PushData(
        pointer(edge_id_arr),
        num_nodes,
        edge_data_arr.handle,
        pointer(edge_length_arr)
    )
    PS.communicator.WaitData(query1)
    PS.communicator.WaitData(query2)

def ps_download(nodes_id, nodes_from, feature_only=False):
    if PS.trace_file is not None:
        for a,b in zip(nodes_id, nodes_from):
            print(a, b, file=PS.trace_file)
            PS.trace_len += 1
    num_nodes = len(nodes_id)
    feat_id_arr = ps_node_feat_id(nodes_id, nodes_from)
    feat_id_arr = np.array(list(feat_id_arr), dtype=np.long)
    feat_data_arr = ndarray.empty(shape=[num_nodes, PS.feature_len + 2])
    feat_length_arr = np.repeat(PS.feature_len + 2, num_nodes)
    #from time import time
    #start = time()
    query = PS.communicator.PullData(
        pointer(feat_id_arr),
        num_nodes,
        feat_data_arr.handle,
        pointer(feat_length_arr)
    )
    PS.communicator.WaitData(query)
    #print("Pull_Data", time() - start, num_nodes)
    feat_data_arr = feat_data_arr.asnumpy()
    feature = feat_data_arr[:,:-2]
    label = feat_data_arr[:,-2:-1].reshape(-1).astype(np.int32)
    degree = feat_data_arr[:,-1:].reshape(-1).astype(np.long)
    if feature_only:
        return feature, label, degree
    edge_id_arr = ps_node_edge_id(nodes_id, nodes_from)
    edge_id_arr = np.array(list(edge_id_arr), dtype=np.long)
    edge_length_arr = degree * 2
    edge_data_arr = ndarray.empty(shape=[edge_length_arr.sum()])
    query = PS.communicator.PullData(
        pointer(edge_id_arr),
        num_nodes,
        edge_data_arr.handle,
        pointer(edge_length_arr)
    )
    PS.communicator.WaitData(query)
    #print("Pull_Data2", time() - start)
    edge_data_arr = edge_data_arr.asnumpy().reshape(-1, 2).T.astype(np.long)
    edge_data_arr = np.ascontiguousarray(edge_data_arr)
    return feature, label, degree, edge_data_arr
