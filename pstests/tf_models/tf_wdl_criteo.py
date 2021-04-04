import numpy as np
import tensorflow as tf


def wdl_criteo(dense_input, sparse_input, y_, cluster=None, task_id=None, use_hvd=False):
    feature_dimension = 33762577
    embedding_size = 128
    learning_rate = 0.01 / 8 # here to comply with HETU
    use_ps = cluster is not None
    partitioner = tf.fixed_size_partitioner(4, 0)
    with tf.device(tf.train.replica_device_setter(cluster=cluster)):
        with tf.device('/cpu:0'):
            rand = np.random.RandomState(seed=123)
            Embedding = tf.compat.v1.get_variable(
                    name="Embedding",
                    dtype=tf.float32,
                    trainable=True,
                    # pylint: disable=unnecessary-lambda
                    shape=(feature_dimension, embedding_size),
                    initializer=tf.random_normal_initializer(stddev=0.01),
                    partitioner=partitioner, 
                    )        
            sparse_input_embedding = tf.nn.embedding_lookup(Embedding, sparse_input)
            W1 = tf.get_variable(name='W1', dtype=tf.float32, shape=[13, 256], initializer=tf.random_normal_initializer(stddev=0.01), partitioner=partitioner)
            W2 = tf.get_variable(name='W2', dtype=tf.float32, shape=[256, 256], initializer=tf.random_normal_initializer(stddev=0.01), partitioner=partitioner)
            W3 = tf.get_variable(name='W3', dtype=tf.float32, shape=[256, 256], initializer=tf.random_normal_initializer(stddev=0.01), partitioner=partitioner)
            W4 = tf.get_variable(name='W4', dtype=tf.float32, shape=[256 + 26 * embedding_size, 1], initializer=tf.random_normal_initializer(stddev=0.01), partitioner=partitioner)
        with tf.device('/gpu:0'):
            rand = np.random.RandomState(seed=123)
            sparse_input_embedding = tf.reshape(sparse_input_embedding, (-1, 26*embedding_size))
            flatten = dense_input
            fc1 = tf.matmul(flatten, W1)
            relu1 = tf.nn.relu(fc1)
            fc2 = tf.matmul(relu1, W2)
            relu2 = tf.nn.relu(fc2)
            y3 = tf.matmul(relu2, W3)

            y4 = tf.concat((sparse_input_embedding, y3), 1)
            y = tf.matmul(y4, W4)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_))

            optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
            if use_hvd:
                import horovod.tensorflow as hvd
                optimizer = hvd.DistributedOptimizer(optimizer)
            train_op = optimizer.minimize(loss)

        return loss, y, train_op
