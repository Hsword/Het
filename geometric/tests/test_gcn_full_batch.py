import numpy as np
from GNN.dataset import load_dataset
from GNN.layer import GCN
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

graph_full = load_dataset("Reddit")
train_split = int(0.8 * graph_full.num_nodes)
graph_full.add_self_loop()
graph = split_training_set(graph_full, train_split)
num_features = graph_full.num_features # =1433
num_classes = graph_full.num_classes # =7

hidden_layer_size = 128

def train_athena(num_epoch):
    ctx = ndarray.gpu(0)

    x_ = ad.Variable(name="x_")
    y_ = ad.Variable(name="y_")

    gcn1 = GCN(num_features, hidden_layer_size)
    gcn2 = GCN(hidden_layer_size, num_classes)

    mp_val = mp_matrix(graph, ctx, use_original_gcn_norm=True)
    feed_dict = {
        gcn1.mp : mp_val,
        gcn2.mp : mp_val,
        x_ : ndarray.array(graph.x, ctx=ctx),
        y_ : ndarray.array(convert_to_one_hot(graph.y, max_val=num_classes), ctx=ctx)
    }

    x = gcn1(x_)
    x = ad.relu_op(x)
    y = gcn2(x)

    loss = ad.softmaxcrossentropy_op(y, y_)

    opt = optimizer.AdamOptimizer(0.01)
    train_op = opt.minimize(loss)
    executor = ad.Executor([loss, y, train_op], ctx=ctx)
    start_time = time.time()
    for i in range(num_epoch):
        loss_val, y_predicted, _ = executor.run(feed_dict = feed_dict)

        y_predicted = y_predicted.asnumpy().argmax(axis=1)
        acc = (y_predicted == graph.y).sum()
        if i==0:
            start_time= time.time()
        print("Train loss :", loss_val.asnumpy().mean())
        print("Train accuracy:", acc/len(y_predicted))
        print("Athena time:",i, time.time()-start_time)
    print("Athena time:", time.time()-start_time)

    mp_val = mp_matrix(graph_full, ctx)

    feed_dict = {
        gcn1.mp : mp_val,
        gcn2.mp : mp_val,
        x_ : ndarray.array(graph_full.x, ctx=ctx),
    }
    executor_eval = ad.Executor([y], ctx=ctx)
    y_predicted, = executor_eval.run(feed_dict=feed_dict)
    y_predicted = y_predicted.asnumpy().argmax(axis=1)
    acc = (y_predicted == graph_full.y)[train_split:].sum()
    print("Test accuracy:", acc/len(y_predicted[train_split:]))

if __name__ == "__main__":
    train_athena(100)