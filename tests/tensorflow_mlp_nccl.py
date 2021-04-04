import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import tensorflow as tf
import gzip
import numpy as np
import os
import six.moves.cPickle as pickle
import time
def load_mnist_data(dataset):
    """ Load the dataset
    Code adapted from http://deeplearning.net/tutorial/code/logistic_sgd.py

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    """
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('Loading data...')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    return train_set, valid_set, test_set

def convert_to_one_hot(vals, max_val = 0):
    """Helper method to convert label array to one-hot array."""
    if max_val == 0:
      max_val = vals.max() + 1
    one_hot_vals = np.zeros((vals.size, max_val))
    one_hot_vals[np.arange(vals.size), vals] = 1
    return one_hot_vals

# if __name__ == "__main__":

#     global_batch_size = 4000
#     mirrored_strategy = tf.distribute.MirroredStrategy()
#     # 在mirrored_strategy空间下
#     # network
#     with mirrored_strategy.scope():
#         rand = np.random.RandomState(seed=123)
#         W1_val = rand.normal(scale=0.1, size=(784, 256))
#         W2_val = rand.normal(scale=0.1, size=(256, 256))
#         W3_val = rand.normal(scale=0.1, size=(256, 10))
#         b1_val = rand.normal(scale=0.1, size=(256))
#         b2_val = rand.normal(scale=0.1, size=(256))
#         b3_val = rand.normal(scale=0.1, size=(10))
#         W1 = tf.Variable(W1_val, dtype = tf.float32)
#         W2 = tf.Variable(W2_val, dtype = tf.float32)
#         W3 = tf.Variable(W3_val, dtype = tf.float32)
#         b1 = tf.Variable(b1_val, dtype = tf.float32)
#         b2 = tf.Variable(b2_val, dtype = tf.float32)
#         b3 = tf.Variable(b3_val, dtype = tf.float32)

#         # relu(X W1 + b1)
#         z1 = tf.matmul(x, W1) + b1
#         z2 = tf.nn.relu(z1)
        
#         # relu(z2 W2 + b2)
#         z3 = tf.matmul(z2,W2) + b2
#         z4 = tf.nn.relu(z3)

#         # relu(z4 W3 + b3)
#         z5 = tf.matmul(z4,W3) + b3
#         y = tf.nn.softmax(z5)
#         # loss
#         # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_real*tf.log(y),reduction_indices=[1]))
#         cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = z5 , labels = y_real))


#         optimizer = tf.train.GradientDescentOptimizer(0.1)
#     # 在mirrored_strategy空间下
#     # dataset
#     with mirrored_strategy.scope():
#         datasets = load_mnist_data("mnist.pkl.gz")
#         train_set_x, train_set_y = datasets[0]
#         valid_set_x, valid_set_y = datasets[1]
#         test_set_x, test_set_y = datasets[2]
#         n_train_batches = train_set_x.shape[0] // global_batch_size
#         n_valid_batches = valid_set_x.shape[0] // global_batch_size       
#         dataset = tf.data.Dataset.from_tensors((train_set_x, convert_to_one_hot(train_set_y, max_val=10))).batch(global_batch_size)
#         # print(dataset)
#     # 这里要分发一下数据
#         dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
#         # print(dist_dataset.__dict__['_cloned_datasets'])
#     def train_step(dist_inputs):
#         def step_fn(inputs):
#             features, labels = inputs
#             logits = model(features)
#             cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
#             logits=logits, labels=labels)
#             loss = tf.reduce_sum(cross_entropy) * (1.0 / global_batch_size)
#             train_op = optimizer.minimize(loss)
#             with tf.control_dependencies([train_op]):
#                 return tf.identity(loss)
#     # 返回所有gpu的loss
#         per_replica_losses = mirrored_strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
#     # reduce loss并返回
#         mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
#         return mean_loss
#     with mirrored_strategy.scope():
#         input_iterator = dist_dataset.make_initializable_iterator()
#         iterator_init = input_iterator.initialize()
#         var_init = tf.global_variables_initializer()
#         loss = train_step(input_iterator.get_next())
#         with tf.Session() as sess:
#             sess.run([var_init, iterator_init])
#             for _ in range(100):
#                 print(sess.run(loss))

if __name__ == "__main__":
    start = time.time()
    global_batch_size = 4000
    mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"])
    # 在mirrored_strategy空间下
    with mirrored_strategy.scope():
        # model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
        model = tf.keras.Sequential([tf.keras.layers.Dense(256, input_shape=(784,), activation = 'relu'),
                                     tf.keras.layers.Dense(256, activation = 'relu'),
                                     tf.keras.layers.Dense(10, activation = 'relu')])
        optimizer = tf.train.GradientDescentOptimizer(0.001)
    # 在mirrored_strategy空间下
    with mirrored_strategy.scope():
        datasets = load_mnist_data("mnist.pkl.gz")
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]
        n_train_batches = train_set_x.shape[0] // global_batch_size
        n_valid_batches = valid_set_x.shape[0] // global_batch_size       
        # dataset = tf.data.Dataset.from_tensor_slices((tf.cast(train_set_x, tf.float32),
        #                                               tf.cast(train_set_y, tf.int32))).batch(global_batch_size)
        dataset = tf.data.Dataset.from_tensor_slices((tf.cast(train_set_x, tf.float32),
                                                      tf.cast(convert_to_one_hot(train_set_y, max_val=10), tf.int32))).repeat(100).batch(global_batch_size)

        # print("========>", dataset)
    # 这里要分发一下数据
        dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
        # print(dist_dataset.__dict__['_cloned_datasets'])
    def train_step(dist_inputs):
        def step_fn(inputs):
            features, labels = inputs
            # print(features)
            # print(labels)
            logits = model(features)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=labels)
            loss = tf.reduce_sum(cross_entropy) * (1.0 / global_batch_size)
            train_op = optimizer.minimize(loss)
            with tf.control_dependencies([train_op]):
                return tf.identity(loss)
    # 返回所有gpu的loss
        per_replica_losses = mirrored_strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
    # reduce loss并返回
        mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        return mean_loss
    with mirrored_strategy.scope():
        input_iterator = dist_dataset.make_initializable_iterator()
        iterator_init = input_iterator.initialize()
        var_init = tf.global_variables_initializer()
        loss = train_step(input_iterator.get_next())
        with tf.Session() as sess:
            # start = time.time()  1
            sess.run([var_init, iterator_init])
            # start = time.time()
            for _ in range(1000):
                # print("here")
                sess.run(loss)
                # print(sess.run(loss))
        end = time.time()
        print("running time is %g s"%(end - start))

'''
if __name__ == "__main__":
    global_batch_size = 16
    mirrored_strategy = tf.distribute.MirroredStrategy()
    # 在mirrored_strategy空间下
    with mirrored_strategy.scope():
        model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
        optimizer = tf.train.GradientDescentOptimizer(0.1)
    # 在mirrored_strategy空间下
    with mirrored_strategy.scope():
        dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(1000).batch(global_batch_size)
        print(dataset)
    # 这里要分发一下数据
        dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
        print(dist_dataset.__dict__['_cloned_datasets'])
    def train_step(dist_inputs):
        def step_fn(inputs):
            features, labels = inputs
            logits = model(features)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=labels)
            loss = tf.reduce_sum(cross_entropy) * (1.0 / global_batch_size)
            train_op = optimizer.minimize(loss)
            with tf.control_dependencies([train_op]):
                return tf.identity(loss)
    # 返回所有gpu的loss
        per_replica_losses = mirrored_strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
    # reduce loss并返回
        mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        return mean_loss
    with mirrored_strategy.scope():
        input_iterator = dist_dataset.make_initializable_iterator()
        iterator_init = input_iterator.initialize()
        var_init = tf.global_variables_initializer()
        loss = train_step(input_iterator.get_next())
        with tf.Session() as sess:
            sess.run([var_init, iterator_init])
            for _ in range(100):
                print(sess.run(loss))
'''