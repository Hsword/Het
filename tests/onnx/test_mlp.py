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


def mnist_mlp(executor_ctx=None, num_epochs=10, print_loss_val_each_epoch=False):


    print("Build 3-layer MLP model...")

    W1 = init.random_normal((784,256),stddev=0.1, name='W1')
    W2 = init.random_normal((256,256),stddev=0.1, name='W2')
    W3 = init.random_normal((256,10),stddev=0.1, name='W3')
    b1 = init.random_normal((256,),stddev=0.1, name='b1')
    b2 = init.random_normal((256,),stddev=0.1, name='b2')
    b3 = init.random_normal((10,),stddev=0.1, name='b3')

    X = ad.Variable(name="X")

    # relu(X W1+b1)
    z1 =ad.matmul_op(X, W1)+b1
    z2 = ad.relu_op(z1)

    # relu(z3 W2+b2)
    z3 = ad.matmul_op(z2, W2)+b2
    z4 = ad.relu_op(z3)

    # softmax(z5 W2+b2)
    y = ad.matmul_op(z4, W3)+b3


    executor=ad.Executor(
        [y],
        ctx=executor_ctx)

    rand = np.random.RandomState(seed=123)
    X_val=rand.normal(scale=0.1, size=(batch_size, 784)).astype(np.float32)

    ath= executor.run(
        feed_dict={
            X: X_val})

    ax.athena2onnx.export(executor,[X],[y],'ath.onnx')
    #
    #
    sess=rt.InferenceSession("ath.onnx")
    input=sess.get_inputs()[0].name
    pre=sess.run(None,{input:X_val.astype(np.float32)})[0]

    np.testing.assert_allclose(pre,ath[0],rtol=1e-2)


if __name__ == '__main__':
    mnist_mlp(executor_ctx=None, num_epochs=5, print_loss_val_each_epoch=True)
