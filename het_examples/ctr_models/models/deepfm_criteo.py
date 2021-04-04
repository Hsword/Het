from athena import gpu_ops as ad
from athena import optimizer
from athena import initializers as init
from athena import dataloader as dl
from athena import ndarray

import numpy as np
import time


# https://zhuanlan.zhihu.com/p/109933924
def dfm_criteo(dense, sparse, labels):
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

    # FM
    Embedding1 = init.random_normal([feature_dimension, 1], stddev=0.01, name="fst_order_embedding", ctx=ndarray.cpu(0))
    FM_W = init.random_normal([13, 1], stddev=0.01, name="dense_parameter")
    sparse_1dim_input = ad.embedding_lookup_op(Embedding1, sparse_input, ctx=ndarray.cpu(0))
    fm_dense_part = ad.matmul_op(dense_input, FM_W)
    fm_sparse_part = ad.reduce_sum_op(sparse_1dim_input, axes = 1)
    """ fst order output"""
    y1 = fm_dense_part + fm_sparse_part

    Embedding2 = init.random_normal([feature_dimension, embedding_size], stddev=0.01, name="snd_order_embedding", ctx=ndarray.cpu(0))
    sparse_2dim_input = ad.embedding_lookup_op(Embedding2, sparse_input, ctx=ndarray.cpu(0))
    sparse_2dim_sum = ad.reduce_sum_op(sparse_2dim_input, axes = 1)
    sparse_2dim_sum_square = ad.mul_op(sparse_2dim_sum, sparse_2dim_sum)

    sparse_2dim_square = ad.mul_op(sparse_2dim_input, sparse_2dim_input)
    sparse_2dim_square_sum = ad.reduce_sum_op(sparse_2dim_square, axes = 1)
    sparse_2dim = sparse_2dim_sum_square +  -1 * sparse_2dim_square_sum
    sparse_2dim_half = sparse_2dim * 0.5
    """snd order output"""
    y2 = ad.reduce_sum_op(sparse_2dim_half, axes = 1, keepdims = True)
    
    #DNN
    flatten = ad.array_reshape_op(sparse_2dim_input,(-1, 26*embedding_size))
    W1 = init.random_normal([26*embedding_size, 256], stddev=0.01, name = "W1")
    W2 = init.random_normal([256, 256], stddev=0.01, name = "W2")
    W3 = init.random_normal([256, 1], stddev=0.01, name = "W3")

    fc1 = ad.matmul_op(flatten, W1)
    relu1 = ad.relu_op(fc1)
    fc2 = ad.matmul_op(relu1, W2)
    relu2 = ad.relu_op(fc2)
    y3 = ad.matmul_op(relu2, W3)

    y4 = y1 + y2
    y = y4 + y3
    y = ad.sigmoid_op(y)

    loss = ad.binarycrossentropy_op(y, y_)
    loss = ad.reduce_mean_op(loss, [0])
    opt = optimizer.SGDOptimizer(learning_rate=learning_rate)
    train_op = opt.minimize(loss)

    return loss, y, y_, train_op
