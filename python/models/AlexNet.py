from athena import gpu_ops as ad
from athena import initializers as init


def conv_bn_relu_pool(x, in_channel, out_channel, name, with_relu=True, with_pool=False):
    weight = init.random_normal(shape=(out_channel, in_channel, 3, 3), stddev=0.1, name=name+'_weight')
    bn_scale = init.random_normal(shape=(1, out_channel, 1, 1), stddev=0.1, name=name+'_bn_scale')
    bn_bias = init.random_normal(shape=(1, out_channel, 1, 1), stddev=0.1, name=name+'_bn_bias')
    x = ad.conv2d_op(x, weight, stride=1, padding=1)
    x = ad.batch_normalization_op(x, bn_scale, bn_bias)
    if with_relu:
        x = ad.relu_op(x)
    if with_pool:
        x = ad.max_pool2d_op(x, kernel_H=2, kernel_W=2, stride=2, padding=0)
    return x


def fc(x, shape, name, with_relu=True):
    weight = init.random_normal(shape=shape, stddev=0.1, name=name+'_weight')
    bias = init.random_normal(shape=shape[-1:], stddev=0.1, name=name+'_bias')
    x = ad.matmul_op(x, weight)
    x = x + ad.broadcastto_op(bias, x)
    if with_relu:
        x = ad.relu_op(x)
    return x


def alexnet(x, y_):
    '''
    AlexNet model, for MNIST dataset.

    Parameters:
        x: Variable(athena.gpu_ops.Node.Node), shape (N, dims)
        y_: Variable(athena.gpu_ops.Node.Node), shape (N, num_classes)
    Return:
        loss: Variable(athena.gpu_ops.Node.Node), shape (1,)
        y: Variable(athena.gpu_ops.Node.Node), shape (N, num_classes)
    '''

    print('Building AlexNet model...')
    x = ad.array_reshape_op(x, [-1, 1, 28, 28])
    x = conv_bn_relu_pool(x,   1,  32, 'alexnet_conv1', with_relu=True, with_pool=True)
    x = conv_bn_relu_pool(x,  32,  64, 'alexnet_conv2', with_relu=True, with_pool=True)
    x = conv_bn_relu_pool(x,  64, 128, 'alexnet_conv3', with_relu=True, with_pool=False)
    x = conv_bn_relu_pool(x, 128, 256, 'alexnet_conv4', with_relu=True, with_pool=False)
    x = conv_bn_relu_pool(x, 256, 256, 'alexnet_conv5', with_relu=False, with_pool=True)
    x = ad.array_reshape_op(x, (-1, 256*3*3))
    x = fc(x, (256*3*3, 1024), name='alexnet_fc1', with_relu=True)
    x = fc(x, (1024, 512), name='alexnet_fc2', with_relu=True)
    y = fc(x, (512, 10), name='alexnet_fc3', with_relu=False)
    loss = ad.softmaxcrossentropy_op(y, y_)
    loss = ad.reduce_mean_op(loss, [0])
    return loss, y
