from athena import gpu_ops as ad
from athena import initializers as init


def conv_pool(x, in_channel, out_channel, name):
    weight = init.random_normal(shape=(out_channel, in_channel, 5, 5), stddev=0.1, name=name+'_weight')
    x = ad.conv2d_op(x, weight, padding=2, stride=1)
    x = ad.relu_op(x)
    x = ad.max_pool2d_op(x, kernel_H=2, kernel_W=2, padding=0, stride=2)
    return x


def fc(x, shape, name, with_relu=True):
    weight = init.random_normal(shape=shape, stddev=0.1, name=name+'_weight')
    bias = init.random_normal(shape=shape[-1:], stddev=0.1, name=name+'_bias')
    x = ad.matmul_op(x, weight)
    x = x + ad.broadcastto_op(bias, x)
    if with_relu:
        x = ad.relu_op(x)
    return x


def lenet(x, y_):
    '''
    LeNet model, for MNIST dataset.

    Parameters:
        x: Variable(athena.gpu_ops.Node.Node), shape (N, dims)
        y_: Variable(athena.gpu_ops.Node.Node), shape (N, num_classes)
    Return:
        loss: Variable(athena.gpu_ops.Node.Node), shape (1,)
        y: Variable(athena.gpu_ops.Node.Node), shape (N, num_classes)
    '''

    print('Building LeNet model...')
    x = ad.array_reshape_op(x, (-1, 1, 28, 28))
    x = conv_pool(x, 1,  6, name='lenet_conv1')
    x = conv_pool(x, 6, 16, name='lenet_conv2')
    x = ad.array_reshape_op(x, (-1, 7*7*16))
    x = fc(x, (7*7*16, 120), name='lenet_fc1', with_relu=True)
    x = fc(x, (120, 84), name='lenet_fc2', with_relu=True)
    y = fc(x, (84,  10), name='lenet_fc3', with_relu=False)
    loss = ad.softmaxcrossentropy_op(y, y_)
    loss = ad.reduce_mean_op(loss, [0])
    return loss, y
