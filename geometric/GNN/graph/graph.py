import numpy as np

import libc_GNN as _C
from libc_GNN import Graph

def get_lib():
    return _C

def shuffle(graph):
    perm = np.random.permutation(graph.num_nodes)
    rperm = np.argsort(perm) # reversed permutation
    return Graph(
        graph.x[rperm],
        graph.y[rperm],
        (perm[graph.edge_index[0]], perm[graph.edge_index[1]]),
        graph.num_classes
    )

def split_training_set(graph, n):
    n = int(n)
    assert(0 < n and n <= graph.num_nodes)
    x = graph.x[:n]
    y = graph.y[:n]
    included_edges = (graph.edge_index[0] < n) * (graph.edge_index[1] < n)
    included_edges = np.where(included_edges)
    edge_index = graph.edge_index[0][included_edges], graph.edge_index[1][included_edges]
    return Graph(x, y, edge_index, graph.num_classes)

def dense_efficient(graph):
    return graph.num_edges / (graph.num_nodes ** 2)

def mp_matrix(graph, device, system="Athena", use_original_gcn_norm=False):
    norm = graph.gcn_norm(use_original_gcn_norm)
    if system=="Athena":
        from athena import ndarray
        mp_mat = ndarray.sparse_array(
            values=norm,
            indices=(graph.edge_index[1], graph.edge_index[0]),
            shape=(graph.num_nodes, graph.num_nodes),
            ctx=device
        )
        return mp_mat
    elif system=="Pytorch":
        import torch
        indices = np.vstack((graph.edge_index[1], graph.edge_index[0]))
        mp_mat = torch.sparse.FloatTensor(
            indices=torch.LongTensor(indices),
            values=torch.FloatTensor(norm),
            size=(graph.num_nodes, graph.num_nodes)
        )
        return mp_mat.to(device)
    elif system=="tensorflow":
        import tensorflow as tf
        indices = np.vstack((graph.edge_index[1], graph.edge_index[0])).T
        shape = np.array([graph.num_nodes, graph.num_nodes], dtype=np.int64)
        mp_val = tf.compat.v1.SparseTensorValue(indices, norm, shape)
        return mp_val
    else:
        raise NotImplementedError

def pick_edges(edge_index, index):
    return edge_index[0][index], edge_index[1][index]
