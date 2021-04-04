from athena import gpu_ops as ad
from athena import initializers as init


def conv_relu_avg(x, shape):
    weight = init.random_normal(shape=shape, stddev=0.1)
    x = ad.conv2d_op(x, weight, padding=2, stride=1)
    x = ad.relu_op(x)
    x = ad.avg_pool2d_op(x, kernel_H=2, kernel_W=2, padding=0, stride=2)
    return x


def fc(x, shape):
    weight = init.random_normal(shape=shape, stddev=0.1)
    bias = init.random_normal(shape=shape[-1:], stddev=0.1)
    x = ad.array_reshape_op(x, (-1, shape[0]))
    x = ad.matmul_op(x, weight)
    y = x + ad.broadcastto_op(bias, x)
    return y


def cnn_3_layers(x, y_):
    '''
    3-layer-CNN model, for MNIST dataset.

    Parameters:
        x: Variable(athena.gpu_ops.Node.Node), shape (N, dims)
        y_: Variable(athena.gpu_ops.Node.Node), shape (N, num_classes)
    Return:
        loss: Variable(athena.gpu_ops.Node.Node), shape (1,)
        y: Variable(athena.gpu_ops.Node.Node), shape (N, num_classes)
    '''
    
    print('Building 3-layer-CNN model...')
    x = ad.array_reshape_op(x, [-1, 1, 28, 28])
    x = conv_relu_avg(x, (32, 1, 5, 5))
    x = conv_relu_avg(x, (64, 32, 5, 5))
    y = fc(x, (7 * 7 * 64, 10))
    loss = ad.softmaxcrossentropy_op(y, y_)
    loss = ad.reduce_mean_op(loss, [0])
    return loss, y    
