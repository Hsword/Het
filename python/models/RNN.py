from athena import gpu_ops as ad
from athena import initializers as init
import numpy as np


def rnn(x, y_):
    '''
    RNN model, for MNIST dataset.

    Parameters:
        x: Variable(athena.gpu_ops.Node.Node), shape (N, dims)
        y_: Variable(athena.gpu_ops.Node.Node), shape (N, num_classes)
    Return:
        loss: Variable(athena.gpu_ops.Node.Node), shape (1,)
        y: Variable(athena.gpu_ops.Node.Node), shape (N, num_classes)
    '''

    print("Building RNN model...")
    diminput = 28
    dimhidden = 128
    dimoutput = 10
    nsteps = 28

    weight1 = init.random_normal(shape=(diminput, dimhidden), stddev=0.1, name='rnn_weight1')
    bias1 = init.random_normal(shape=(dimhidden, ), stddev=0.1, name='rnn_bias1')
    weight2 = init.random_normal(shape=(dimhidden+dimhidden, dimhidden), stddev=0.1, name='rnn_weight2')
    bias2 = init.random_normal(shape=(dimhidden, ), stddev=0.1, name='rnn_bias2')
    weight3 = init.random_normal(shape=(dimhidden, dimoutput), stddev=0.1, name='rnn_weight3')
    bias3 = init.random_normal(shape=(dimoutput, ), stddev=0.1, name='rnn_bias3')
    last_state = ad.Variable(value=np.zeros((1,)).astype(np.float32), name='initial_state', trainable=False)

    for i in range(nsteps):
        cur_x = ad.slice_op(x, (0, i*diminput), (-1, diminput))
        h = ad.matmul_op(cur_x, weight1)
        h = h + ad.broadcastto_op(bias1, h)

        if i == 0:
            last_state = ad.broadcastto_op(last_state, h)
        s = ad.concat_op(h, last_state, axis=1)
        s = ad.matmul_op(s, weight2)
        s = s + ad.broadcastto_op(bias2, s)
        last_state = ad.relu_op(s)
    
    final_state = last_state
    x = ad.matmul_op(final_state, weight3)
    y = x + ad.broadcastto_op(bias3, x)
    loss = ad.softmaxcrossentropy_op(y, y_)
    loss = ad.reduce_mean_op(loss, [0])
    return loss, y
