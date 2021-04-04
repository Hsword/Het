from athena import ndarray
from athena import gpu_links as gpu_op
from athena import gpu_ops as ad
from athena import initializers as init
from athena import onnx as ax

import onnxruntime as rt

import numpy as np




import argparse
import six.moves.cPickle as pickle
import gzip
import os
import pdb
import ctypes
import time
batch_size=128



def cnn(executor_ctx=None, num_epochs=10, print_loss_val_each_epoch=False):


    print("Build CNN model...")

    W1 = init.random_normal((32,1,5,5),stddev=0.1, name='W1')
    W2 = init.random_normal((64,32,5,5),stddev=0.1, name='W2')
    W3 = init.random_normal((7*7*64,10),stddev=0.1, name='W3')
    b3 = init.random_normal((10,),stddev=0.1, name='b3')

    X = ad.Variable(name="X")

    z1 = ad.conv2d_op(X, W1, padding=2, stride=1)
    z2 = ad.relu_op(z1)
    z3 = ad.avg_pool2d_op(z2, kernel_H=2, kernel_W=2, padding=0, stride=2)

    z4 = ad.conv2d_op(z3, W2, padding=2, stride=1)
    z5 = ad.relu_op(z4)
    z6 = ad.avg_pool2d_op(z5, kernel_H=2, kernel_W=2, padding=0, stride=2)

    z6_flat = ad.array_reshape_op(z6, (-1, 7 * 7 * 64))
    y = ad.matmul_op(z6_flat, W3)+b3



    executor=ad.Executor(
        [y],
        ctx=executor_ctx)

    rand = np.random.RandomState(seed=123)
    X_val=rand.normal(scale=0.1, size=(batch_size, 1,28,28)).astype(np.float32)

    ath= executor.run(
        feed_dict={
            X: X_val})

    ax.athena2onnx.export(executor,[X],[y],'ath.onnx')
    #
    #
    sess=rt.InferenceSession("ath.onnx")
    input=sess.get_inputs()[0].name

    pre=sess.run(None,{input:X_val.astype(np.float32)})[0]

    np.testing.assert_allclose(ath[0],pre,rtol=1e-2)


if __name__ == '__main__':
    cnn(executor_ctx=None, num_epochs=5, print_loss_val_each_epoch=True)
