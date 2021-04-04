import numpy as np
import tensorflow as tf


def dfm_criteo(dense_input, sparse_input, y_, cluster=None, task_id=None, use_hvd=False):
    feature_dimension = 33762577
    embedding_size = 128
    learning_rate = 0.01 / 8 # here to comply with HETU
    use_ps = cluster is not None

    if use_ps:
        device = tf.device("/job:ps/task:0/cpu:0")
    else:
        device = tf.device("/cpu:0")
    with device:
        rand = np.random.RandomState(seed=123)
        Embedding1 = tf.get_variable(
                name="Embedding1",
                dtype=tf.float32,
                trainable=True,
                # pylint: disable=unnecessary-lambda
                shape=(feature_dimension, 1),
                initializer=tf.random_normal_initializer(stddev=0.01)
                )        
        Embedding2 = tf.get_variable(
                name="embeddings",
                dtype=tf.float32,
                trainable=True,
                # pylint: disable=unnecessary-lambda
                shape=(feature_dimension, embedding_size),
                initializer=tf.random_normal_initializer(stddev=0.01)
                )
        sparse_1dim_input = tf.nn.embedding_lookup(Embedding1, sparse_input)
        sparse_2dim_input = tf.nn.embedding_lookup(Embedding2, sparse_input)

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
        # FM
        FM_W = tf.Variable(rand.normal(scale = 0.01, size = [13, 1]), dtype = tf.float32)
        
        fm_dense_part = tf.matmul(dense_input, FM_W)
        fm_sparse_part = tf.reduce_sum(sparse_1dim_input, 1)
        """ fst order output"""
        y1 = fm_dense_part + fm_sparse_part

        
        sparse_2dim_sum = tf.reduce_sum(sparse_2dim_input, 1)
        sparse_2dim_sum_square = tf.multiply(sparse_2dim_sum, sparse_2dim_sum)

        sparse_2dim_square = tf.multiply(sparse_2dim_input, sparse_2dim_input)
        sparse_2dim_square_sum = tf.reduce_sum(sparse_2dim_square, 1)
        sparse_2dim = sparse_2dim_sum_square +  -1 * sparse_2dim_square_sum
        sparse_2dim_half = sparse_2dim * 0.5
        """snd order output"""
        y2 = tf.reduce_sum(sparse_2dim_half, 1, keepdims = True)
        
        #DNN
        flatten = tf.reshape(sparse_2dim_input,(-1, 26*embedding_size))

        W1 = tf.Variable(rand.normal(scale = 0.01, size = [26*embedding_size, 256]), dtype = tf.float32)
        W2 = tf.Variable(rand.normal(scale = 0.01, size = [256, 256]), dtype = tf.float32)
        W3 = tf.Variable(rand.normal(scale = 0.01, size = [256, 1]), dtype = tf.float32)
        
        fc1 = tf.matmul(flatten, W1)
        relu1 = tf.nn.relu(fc1)
        fc2 = tf.matmul(relu1, W2)
        relu2 = tf.nn.relu(fc2)
        y3 = tf.matmul(relu2, W3)

        y4 = y1 + y2
        y = y4 + y3
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_))

        param_list = [W1, W2, W3, FM_W, Embedding1, Embedding2]

        # grad_param_list = tf.gradients(loss, param_list)
        # global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        if use_hvd:
            import horovod.tensorflow as hvd
            optimizer = hvd.DistributedOptimizer(optimizer)
        train_op = optimizer.minimize(loss, global_step=global_step)

        if use_ps:
            return loss, y, train_op, global_step
        else:
            return loss, y, train_op
