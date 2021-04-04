import numpy as np
import tensorflow as tf


def cross_layer(x0, x1):
    # x0: input embedding feature (batch_size, 26 * embedding_size + 13)
    # x1: the output of last layer (batch_size, 26 * embedding_size + 13)

    embed_dim = x1.shape[-1]
    rand = np.random.RandomState(seed=123)
    w = tf.Variable(rand.normal(scale = 0.01, size = (embed_dim,)), dtype = tf.float32)
    b = tf.Variable(rand.normal(scale = 0.01, size = (embed_dim,)), dtype = tf.float32)
    x_1w = tf.tensordot(tf.reshape(x1, [-1, 1, embed_dim]), w, axes = 1)
    cross = x0 * x_1w
    return cross + x1 + b

def build_cross_layer(x0, num_layers = 3):
    x1 = x0
    for i in range(num_layers):
        x1 = cross_layer(x0, x1)
    return x1


def dcn_criteo(dense_input, sparse_input, y_, cluster=None, task_id=None, use_hvd=False):
    feature_dimension = 33762577
    embedding_size = 128
    learning_rate = 0.003 / 8 # here to comply with HETU
    use_ps = cluster is not None

    if use_ps:
        device = tf.device("/job:ps/task:0/cpu:0")
    else:
        device = tf.device("/cpu:0")
    with device:
        rand = np.random.RandomState(seed=123)
        Embedding = tf.get_variable(
                name="Embedding",
                dtype=tf.float32,
                trainable=True,
                # pylint: disable=unnecessary-lambda
                shape=(feature_dimension, embedding_size),
                initializer=tf.random_normal_initializer(stddev=0.01)
                )        
        sparse_input_embedding = tf.nn.embedding_lookup(Embedding, sparse_input)
    
    if use_ps:
        device = tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d/gpu:0" % (task_id),
                cluster=cluster))
        # device = tf.device("/job:worker/task:0/gpu:0")
    else:
        device = tf.device("/gpu:0")
        global_step = tf.Variable(0, name="global_step", trainable=False)
    with device:
        if use_ps:
            global_step = tf.Variable(0, name="global_step", trainable=False)
        rand = np.random.RandomState(seed=123)
        flatten = tf.reshape(sparse_input_embedding, (-1, 26*embedding_size))
        x = tf.concat((flatten, dense_input), 1)
        # CrossNet
        cross_output = build_cross_layer(x, num_layers=3)
        # DNN
        flatten = x

        W1 = tf.Variable(rand.normal(scale = 0.01, size = [26*embedding_size + 13, 256]), dtype = tf.float32)
        W2 = tf.Variable(rand.normal(scale = 0.01, size = [256, 256]), dtype = tf.float32)
        W3 = tf.Variable(rand.normal(scale = 0.01, size = [256, 256]), dtype = tf.float32)
        W4 = tf.Variable(rand.normal(scale = 0.01, size = [256 + 26 * embedding_size + 13, 1]), dtype = tf.float32)
        
        fc1 = tf.matmul(flatten, W1)
        relu1 = tf.nn.relu(fc1)
        fc2 = tf.matmul(relu1, W2)
        relu2 = tf.nn.relu(fc2)
        y3 = tf.matmul(relu2, W3)

        y4 = tf.concat((cross_output, y3), 1)
        y = tf.matmul(y4, W4)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        if use_hvd:
            import horovod.tensorflow as hvd
            optimizer = hvd.DistributedOptimizer(optimizer)
        train_op = optimizer.minimize(loss, global_step=global_step)

        if use_ps:
            return loss, y, train_op, global_step
        else:
            return loss, y, train_op
