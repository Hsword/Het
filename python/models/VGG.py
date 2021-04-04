from athena import gpu_ops as ad
from athena import initializers as init


def conv_bn_relu(x, in_channel, out_channel, name):
    weight = init.random_normal(shape=(out_channel, in_channel, 3, 3), 
        stddev=0.1, name=name+'_weight')
    bn_scale = init.random_normal(shape=(1, out_channel, 1, 1), 
        stddev=0.1, name=name+'_bn_scale')
    bn_bias = init.random_normal(shape=(1, out_channel, 1, 1), 
        stddev=0.1, name=name+'_bn_bias')
    
    conv = ad.conv2d_op(x, weight, padding=1, stride=1)
    bn = ad.batch_normalization_op(conv, bn_scale, bn_bias)
    act = ad.relu_op(bn)
    return act


def vgg_2block(x, in_channel, out_channel, name):
    x = conv_bn_relu(x, in_channel, out_channel, name=name+'_layer1')
    x = conv_bn_relu(x, out_channel, out_channel, name=name+'_layer2')
    x = ad.max_pool2d_op(x, kernel_H=2, kernel_W=2, padding=0, stride=2)
    return x


def vgg_3block(x, in_channel, out_channel, name):
    x = conv_bn_relu(x, in_channel, out_channel, name=name+'_layer1')
    x = conv_bn_relu(x, out_channel, out_channel, name=name+'_layer2')
    x = conv_bn_relu(x, out_channel, out_channel, name=name+'_layer3')
    x = ad.max_pool2d_op(x, kernel_H=2, kernel_W=2, padding=0, stride=2)
    return x


def vgg_4block(x, in_channel, out_channel, name):
    x = conv_bn_relu(x, in_channel, out_channel, name=name+'_layer1')
    x = conv_bn_relu(x, out_channel, out_channel, name=name+'_layer2')
    x = conv_bn_relu(x, out_channel, out_channel, name=name+'_layer3')
    x = conv_bn_relu(x, out_channel, out_channel, name=name+'_layer4')
    x = ad.max_pool2d_op(x, kernel_H=2, kernel_W=2, padding=0, stride=2)
    return x


def vgg_fc(x, in_feat, out_feat, name):
    weight = init.random_normal(shape=(in_feat, out_feat), 
        stddev=0.1, name=name+'_weight')
    bias = init.random_normal(shape=(out_feat,), 
        stddev=0.1, name=name+'_bias')
    x = ad.matmul_op(x, weight)
    x = x + ad.broadcastto_op(bias, x)
    return x


def vgg(x, y_, num_layers):
    '''
    VGG model, for CIFAR10 dataset.

    Parameters:
        x: Variable(athena.gpu_ops.Node.Node), shape (N, C, H, W)
        y_: Variable(athena.gpu_ops.Node.Node), shape (N, num_classes)
        num_layers: 16 or 19
    Return:
        loss: Variable(athena.gpu_ops.Node.Node), shape (1,)
        y: Variable(athena.gpu_ops.Node.Node), shape (N, num_classes)
    '''
    
    if num_layers == 16:
        print('Building VGG-16 model...')
        x = vgg_2block(x,   3,  64, 'vgg_block1')
        x = vgg_2block(x,  64, 128, 'vgg_block2')
        x = vgg_3block(x, 128, 256, 'vgg_block3')
        x = vgg_3block(x, 256, 512, 'vgg_block4')
        x = vgg_3block(x, 512, 512, 'vgg_block5')
        
    elif num_layers == 19:
        print('Building VGG-19 model...')
        x = vgg_2block(x,   3,  64, 'vgg_block1')
        x = vgg_2block(x,  64, 128, 'vgg_block2')
        x = vgg_4block(x, 128, 256, 'vgg_block3')
        x = vgg_4block(x, 256, 512, 'vgg_block4')
        x = vgg_4block(x, 512, 512, 'vgg_block5')
    
    else:
        assert False, 'VGG model should have 16 or 19 layers!'
    
    x = ad.array_reshape_op(x, (-1, 512))
    x = vgg_fc(x,  512, 4096, 'vgg_fc1')
    x = vgg_fc(x, 4096, 4096, 'vgg_fc2')
    y = vgg_fc(x, 4096,   10, 'vgg_fc3')

    loss = ad.softmaxcrossentropy_op(y, y_)
    loss = ad.reduce_mean_op(loss, [0])

    return loss, y


def vgg16(x, y_):
    return vgg(x, y_, 16)


def vgg19(x, y_):
    return vgg(x, y_, 19)
