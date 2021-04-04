from athena import gpu_ops as ad
from athena import optimizer
from athena import initializers as init
from athena import dataloader as dl
from athena import ndarray

import numpy as np
import time


def wdl_criteo(dense, sparse, labels):
    batch_size = 128
    feature_dimension = 33762577
    embedding_size = 128
    learning_rate = 0.01
    if isinstance(dense, tuple):
        dense_input = dl.dataloader_op([[dense[0], batch_size, 'train'], [dense[1], batch_size, 'validate']])
        sparse_input = dl.dataloader_op([[sparse[0], batch_size, 'train'], [sparse[1], batch_size, 'validate']])
        y_ = dl.dataloader_op([[labels[0], batch_size, 'train'], [labels[1], batch_size, 'validate']])
    else:
        dense_input = dl.dataloader_op([[dense, batch_size, 'train']])
        sparse_input = dl.dataloader_op([[sparse, batch_size, 'train']])
        y_ = dl.dataloader_op([[labels, batch_size, 'train']])
    print("Data loaded.")
    Embedding = init.random_normal([feature_dimension, embedding_size], stddev=0.01, name="snd_order_embedding", ctx=ndarray.cpu(0))
    sparse_input = ad.embedding_lookup_op(Embedding, sparse_input, ctx=ndarray.cpu(0))
    sparse_input = ad.array_reshape_op(sparse_input, (-1, 26*embedding_size))

    #DNN
    flatten = dense_input
    W1 = init.random_normal([13, 256], stddev=0.01, name = "W1")
    W2 = init.random_normal([256, 256], stddev=0.01, name = "W2")
    W3 = init.random_normal([256, 256], stddev=0.01, name = "W3")

    W4 = init.random_normal([256 + 26*embedding_size, 1], stddev=0.01, name = "W4")

    fc1 = ad.matmul_op(flatten, W1)
    relu1 = ad.relu_op(fc1)
    fc2 = ad.matmul_op(relu1, W2)
    relu2 = ad.relu_op(fc2)
    y3 = ad.matmul_op(relu2, W3)

    y4 = ad.concat_op(sparse_input, y3, axis = 1)
    y = ad.matmul_op(y4, W4)
    y = ad.sigmoid_op(y)

    loss = ad.binarycrossentropy_op(y, y_)
    loss = ad.reduce_mean_op(loss, [0])
    opt = optimizer.SGDOptimizer(learning_rate=learning_rate)
    train_op = opt.minimize(loss)
    
    return loss, y, y_, train_op