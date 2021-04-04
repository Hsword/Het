import numpy as np
import scipy
from athena import ndarray
from athena import gpu_links as gpu_op
from athena import gpu_ops as ad

def softmax_func(y):
    """Numerically stable softmax."""
    b = y - np.max(y, axis=1, keepdims=True)
    expb = np.exp(b)
    softmax = expb / np.sum(expb, axis=1, keepdims=True)
    return softmax

def test_csrmm_op(executor_ctx):
    X = ad.Variable(name="X")
    W = ad.Variable(name="W")
    Y = ad.csrmm_op(X, W)
    Y_ = ad.Variable(name="Y_")
    loss = ad.softmaxcrossentropy_op(Y, Y_)
    loss = ad.reduce_mean_op(loss, [0])
    grads = ad.gradients(loss, [W, Y])
    
    executor = ad.Executor(
        [loss, grads[0], grads[1]], ctx=executor_ctx)
    
    rand = np.random.RandomState(seed=123)

    W_val = rand.normal(scale=0.1, size=[70000, 2]).astype(np.float32)
    if ndarray.is_gpu_ctx(executor_ctx):
        W_val = ndarray.array(W_val, ctx=executor_ctx)
    
    X_val = scipy.sparse.rand(500, 70000, density=1e-5,format='coo',dtype=np.float32)
    Y_val = np.random.uniform(0, 10, size=(500, 2)).astype(np.float32) 
    
    loss_val = executor.run(feed_dict={X: X_val, Y_: Y_val, W: W_val})
    
    if ndarray.is_gpu_ctx(executor_ctx):
        W_val = W_val.asnumpy()
    loss_val = [val.asnumpy() for val in loss_val]
    
    y_groundtruth = X_val.dot(W_val)
    loss_groundtruth = np.mean(
                -np.sum(Y_val * np.log(softmax_func(y_groundtruth)), axis=1), keepdims=True)
    Y_grad_groundtruth = (softmax_func(y_groundtruth) + -1 * Y_val) * np.ones(loss_groundtruth.shape) / 500
    W_grad_groundtruth = X_val.T.dot(Y_grad_groundtruth)

    np.testing.assert_allclose(loss_val[0], loss_groundtruth, rtol=1e-4)
    np.testing.assert_allclose(loss_val[1], W_grad_groundtruth, rtol=1e-4)
    np.testing.assert_allclose(loss_val[2], Y_grad_groundtruth, rtol=1e-4)
    
    
test_csrmm_op(ndarray.cpu(0))
test_csrmm_op(ndarray.gpu(1))


def test_csrmv_op(executor_ctx):
    X = ad.Variable(name="X")
    W = ad.Variable(name="W")
    Y = ad.csrmv_op(X, W)
    Y_ = ad.Variable(name="Y_")
    temp = Y + (-1) * Y_
    loss = temp * temp

    grads = ad.gradients(loss, [W, Y])
    
    executor = ad.Executor(
        [loss, grads[0], grads[1]], ctx=executor_ctx)
    
    rand = np.random.RandomState(seed=123)

    W_val =rand.normal(scale=0.1, size=[70000, ])
    if ndarray.is_gpu_ctx(executor_ctx):
        W_val = ndarray.array(W_val, ctx=executor_ctx)
    
    X_val = scipy.sparse.rand(500, 70000, density=1e-5,format='coo',dtype=np.float32)
    Y_val = np.random.uniform(0, 10, size=(500, )).astype(np.float32) 
    
    loss_val = executor.run(feed_dict={X: X_val, Y_: Y_val, W: W_val})
    
    if ndarray.is_gpu_ctx(executor_ctx):
        W_val = W_val.asnumpy()
    loss_val = [val.asnumpy() for val in loss_val]
    
    y_groundtruth = X_val.dot(W_val)
    loss_groundtruth = (y_groundtruth - Y_val) ** 2
    Y_grad_groundtruth = 2 * (y_groundtruth - Y_val) * np.ones(loss_groundtruth.shape)
    W_grad_groundtruth = X_val.T.dot(Y_grad_groundtruth)
    

    np.testing.assert_allclose(loss_val[0], loss_groundtruth, rtol=1e-4)
    np.testing.assert_allclose(loss_val[1], W_grad_groundtruth, rtol=1e-4)
    np.testing.assert_allclose(loss_val[2], Y_grad_groundtruth, rtol=1e-4)
    

test_csrmv_op(ndarray.cpu(0))
test_csrmv_op(ndarray.gpu(1))