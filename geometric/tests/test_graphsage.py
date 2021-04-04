import numpy as np
from GNN.dataset import load_dataset
from GNN.layer import *
from GNN.graph import *

from athena import initializers
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
#graph_full = shuffle(graph_full)
train_split = int(0.8 * graph_full.num_nodes)
graph_full.add_self_loop()
graph = split_training_set(graph_full, train_split)
def transform(graph):
    mp_val = mp_matrix(graph, ndarray.gpu(0))
    return graph, mp_val
hidden_layer_size = 128

def train_athena(num_epoch):
    ctx = ndarray.gpu(0)

    x_ = ad.Variable(name="x_")
    y_ = ad.Variable(name="y_")
    mask_ = ad.Variable(name="mask_")

    gcn1 = GraphSage(graph.num_features, hidden_layer_size, activation="relu", dropout=0.1)
    gcn2 = GraphSage(2*hidden_layer_size, hidden_layer_size, activation="relu", dropout=0.1)

    x = gcn1(x_)
    x = gcn2(x)
    W = initializers.xavier_uniform(shape=(2*hidden_layer_size, graph.num_classes))
    B = initializers.zeros(shape=(graph.num_classes,))
    x = ad.matmul_op(x, W)
    y = x + ad.broadcastto_op(B, x)

    loss = ad.softmaxcrossentropy_op(y, y_)
    loss = ad.mul_op(loss, mask_)
    opt = optimizer.AdamOptimizer(0.01)
    train_op = opt.minimize(loss)
    executor = ad.Executor([loss, y, train_op], ctx=ctx)

    def eval():
        start = time.time()
        ad.Dropout.DropoutOp.phase = "eval"
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
        ad.Dropout.DropoutOp.phase = "training"
    epoch = 0
    nnodes = 0
    batch_size = 1000
    with GraphSageSampler(graph, batch_size, depth=2, num_sample_thread=4) as sampler:
        start = time.time()
        while True:
            g_sample, mask = sampler.sample()
            mp_val = mp_matrix(g_sample, ctx)
            #print(time.time() - start)
            feed_dict = {
                gcn1.mp : mp_val,
                gcn2.mp : mp_val,
                mask_ : ndarray.array(mask,ctx=ctx),
                x_ : ndarray.array(g_sample.x, ctx=ctx),
                y_ : ndarray.array(convert_to_one_hot(g_sample.y, max_val=graph.num_classes), ctx=ctx)
            }
            loss_val, y_predicted, _ = executor.run(feed_dict = feed_dict)
            y_predicted = y_predicted.asnumpy().argmax(axis=1)
            acc = ((y_predicted == g_sample.y) * mask).sum()
            # print(i, "Train loss :", loss_val.asnumpy().mean())
            # print(i, "Train accuracy:", acc/len(y_predicted))
            nnodes += batch_size
            if nnodes > graph_full.num_nodes:
                nnodes = 0
                epoch += 1
                print("Epoch :", epoch, time.time() - start)
                print("Train accuracy:", acc/mask.sum())
                eval()
                start = time.time()
                if epoch >= num_epoch:
                    break

if __name__ == "__main__":
    train_athena(50)