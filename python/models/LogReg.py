from athena import gpu_ops as ad
from athena import initializers as init


def logreg(x, y_):
    '''
    Logistic Regression model, for MNIST dataset.

    Parameters:
        x: Variable(athena.gpu_ops.Node.Node), shape (N, dims)
        y_: Variable(athena.gpu_ops.Node.Node), shape (N, num_classes)
    Return:
        loss: Variable(athena.gpu_ops.Node.Node), shape (1,)
        y: Variable(athena.gpu_ops.Node.Node), shape (N, num_classes)
    '''

    print("Build logistic regression model...")
    weight = init.zeros((784, 10), name='logreg_weight')
    bias = init.zeros((10,), name='logreg_bias')
    x = ad.matmul_op(x, weight)
    y = x + ad.broadcastto_op(bias, x)
    loss = ad.softmaxcrossentropy_op(y, y_)
    loss = ad.reduce_mean_op(loss, [0])
    return loss, y
