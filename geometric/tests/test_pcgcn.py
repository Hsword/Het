import numpy as np
from scipy import sparse
from GNN.dataset import load_dataset
from GNN.layer import PCGCN
from GNN.graph import *

from athena import ndarray
from athena import gpu_ops as ad
from athena import optimizer

import time

def convert_to_one_hot(vals, max_val = 0):
    """Helper method to convert label array to one-hot array."""
    if max_val == 0:
      max_val = vals.max() + 1
    one_hot_vals = np.zeros((vals.size, max_val))
    one_hot_vals[np.arange(vals.size), vals] = 1
    return one_hot_vals

graph = load_dataset("Cora")

train_split = graph.num_nodes // 10

# dual mode message passing
# use dense matmul for subgraph with dense_efficient > dense_threshold
# bisect profiling required to decide this value
dense_threshold = 1

def train_athena(num_epoch):
    ctx = ndarray.gpu(0)
    feed_dict = {}
    nparts = 4
    graph.add_self_loop()
    norm = graph.gcn_norm(True)
    graphs, edge_list, reindexed_edges = graph.part_graph(nparts)
    x_val = np.concatenate(list(map(lambda g: g.x, graphs)))
    y_concat = np.concatenate(list(map(lambda g: g.y, graphs)))
    y_val = convert_to_one_hot(y_concat, max_val=graph.num_classes) # shape=(n, num_classes)
    x_ = ad.Variable(name="x_")
    y_ = ad.Variable(name="y_")
    feed_dict[x_] = ndarray.array(x_val, ctx=ctx)
    feed_dict[y_] = ndarray.array(y_val, ctx=ctx)
    gcn1 = PCGCN(graph.num_features, 16, npart=nparts)
    gcn2 = PCGCN(16,graph.num_classes, npart=nparts)
    mp_val = [[None for j in range(nparts)] for i in range(nparts)]
    use_sparse = [True for g in graphs]
    for i in range(nparts):
        for j in range(nparts):
            if i==j:
                edges = graphs[i].edge_index
            else:
                edges = pick_edges(reindexed_edges, edge_list[i][j])

            if i==j and use_sparse[i] == False:
                mp_val[i][j] = sparse.csr_matrix(
                    (
                        norm[edge_list[i][j]],
                        (edges[1], edges[0])
                    ),
                    shape=(graphs[j].num_nodes, graphs[i].num_nodes)
                ).toarray()
            else:
                mp_val[i][j] = ndarray.sparse_array(
                    values=norm[edge_list[i][j]],
                    indices=(edges[1], edges[0]),
                    shape=(graphs[j].num_nodes, graphs[i].num_nodes),
                    ctx=ctx
                )
            feed_dict[gcn1.mp[i][j]] = mp_val[i][j]
            feed_dict[gcn2.mp[i][j]] = mp_val[i][j]

    subgraph_size = list(map(lambda g: g.num_nodes, graphs))
    x = gcn1(x_, subgraph_size=subgraph_size, use_sparse=use_sparse)
    x = ad.relu_op(x)
    y = gcn2(x, subgraph_size=subgraph_size, use_sparse=use_sparse)
    # y_train = ad.slice_op(y, (0, 0), (train_split, graph.num_classes))

    # loss = ad.softmaxcrossentropy_op(y_train, y_)
    loss = ad.softmaxcrossentropy_op(y, y_)
    opt = optimizer.AdamOptimizer(0.01)
    train_op = opt.minimize(loss)
    executor = ad.Executor([loss, y, train_op], ctx=ctx)
    losses = []
    for i in range(num_epoch):
        loss_val, y_predicted, _ = executor.run(feed_dict = feed_dict)

        y_predicted = y_predicted.asnumpy().argmax(axis=1)
        acc = (y_predicted == y_concat).sum()
        losses.append(loss_val.asnumpy()[0])
        if i==0:
            start_time = time.time()
        print("Train loss :", loss_val.asnumpy().mean())
        print("Val accuracy:", acc/len(y_predicted))
    print("Athena time:", (time.time()-start_time)/199)
    return losses

loss = train_athena(200)
