from athena import gpu_ops as ad
from athena import optimizer
from athena import initializers as init
from athena import dataloader as dl

import numpy as np
import time



def residual_layer(x0, input_dim, hidden_dim):
    
    embedding_len = input_dim
    weight_1 = init.random_normal(shape=(input_dim, hidden_dim), stddev=0.1, name='weight_1')
    bias_1 = init.random_normal(shape=(hidden_dim,), stddev=0.1, name='bias_1')
    weight_2 = init.random_normal(shape=(hidden_dim, input_dim), stddev=0.1, name='weight_2')
    bias_2 = init.random_normal(shape=(input_dim,), stddev=0.1, name='bias_2')

    x0w = ad.matmul_op(x0, weight_1) #(batch, hidden_dim)
    x0w_b = x0w + ad.broadcastto_op(bias_1, x0w)
   
    relu1 = ad.relu_op(x0w_b)
    x1w = ad.matmul_op(relu1, weight_2) #(batch, input_dim)
    x1w_b = x1w + ad.broadcastto_op(bias_2, x1w)
    residual = x1w_b + x0
    y = ad.relu_op(residual)
    return y


def build_residual_layers(x0, input_dim, hidden_dim, num_layers = 3):
    for i in range(num_layers):
        x0 = residual_layer(x0, input_dim, hidden_dim)
    return x0

def dc_criteo(dense, sparse, labels):

    batch_size = 128
    feature_dimension = 33762577
    embedding_size = 8
    learning_rate = 0.001
    if isinstance(dense, tuple):
        dense_input = dl.dataloader_op([[dense[0], batch_size, 'train'], [dense[1], batch_size, 'validate']])
        sparse_input = dl.dataloader_op([[sparse[0], batch_size, 'train'], [sparse[1], batch_size, 'validate']])
        y_ = dl.dataloader_op([[labels[0], batch_size, 'train'], [labels[1], batch_size, 'validate']])
    else:
        dense_input = dl.dataloader_op([[dense, batch_size, 'train']])
        sparse_input = dl.dataloader_op([[sparse, batch_size, 'train']])
        y_ = dl.dataloader_op([[labels, batch_size, 'train']])
    print("Data loaded.")

    Embedding = init.random_normal([feature_dimension, embedding_size], stddev=0.01, name="snd_order_embedding")
    sparse_input = ad.embedding_lookup_op(Embedding, sparse_input)
    sparse_input = ad.array_reshape_op(sparse_input, (-1, 26*embedding_size))

    ## dc_model
    x = ad.concat_op(sparse_input, dense_input, axis=1)
   
    input_dim = 26 * 8 + 13
    hidden_dim = input_dim
    residual_out = build_residual_layers(x, input_dim, hidden_dim, num_layers = 5)

    W4 = init.random_normal([26*embedding_size + 13, 1], stddev=0.1, name = "W4")
    y = ad.matmul_op(residual_out, W4)
    y = ad.sigmoid_op(y)

    loss = ad.binarycrossentropy_op(y, y_)
    loss = ad.reduce_mean_op(loss, [0])
    opt = optimizer.SGDOptimizer(learning_rate=learning_rate)
    train_op = opt.minimize(loss)
   
    return loss, y, y_, train_op
