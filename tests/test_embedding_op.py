import numpy as np
import tensorflow as tf
from athena import ndarray
from athena import gpu_links as gpu_op
from athena import gpu_ops as ad
from athena import optimizer

def test_embedding(executor_ctx=ndarray.gpu(0)):
    embedding = ad.Variable('embeddingtable', value=np.random.rand(5, 5))
    index = ad.Variable(name="index")
    ids= [[0, 1], [0, 1]]
    ids = np.array(ids)
    ids = ndarray.array(ids, ctx = executor_ctx)
    y= ad.embedding_lookup_op(embedding, index)
    opt = optimizer.SGDOptimizer(0.1)
    train_op = opt.minimize(y)
    executor = ad.Executor([y, train_op],ctx=executor_ctx)

    print("embedding:", embedding.tensor_value.asnumpy())
    print("ids:", ids.asnumpy())
    out, _ = executor.run(feed_dict={index: ids})
    print(out.asnumpy())
    print(embedding.tensor_value.asnumpy())


def test_embedding_with_tf(opt_name, iters=10000, executor_ctx=ndarray.gpu(0)):
    from time import time

    value = np.random.rand(5, 5)
    ids= [[0, 1], [0, 1]]
    ids = np.array(ids)

    # tf part
    tf_embedding = tf.Variable(value, dtype=tf.float32)
    tf_ids = tf.placeholder(tf.int32)
    tf_y = tf.nn.embedding_lookup(tf_embedding, tf_ids)
    tf_opts = {
        'sgd': tf.train.GradientDescentOptimizer(0.1),
        'momentum': tf.train.MomentumOptimizer(0.1, momentum=0.9),
        'nesterov': tf.train.MomentumOptimizer(0.1, momentum=0.9, use_nesterov=True),
        'adagrad': tf.train.AdagradOptimizer(0.1, initial_accumulator_value=1e-7, use_locking=True),
        'adam': tf.train.AdamOptimizer(0.1, epsilon=1e-7, use_locking=True),
    }
    tf_opt = tf_opts[opt_name]

    tf_trainop = tf_opt.minimize(tf_y)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        start = time()
        for i in range(iters):
            tf_out, _ = sess.run([tf_y, tf_trainop], feed_dict={tf_ids: ids})
        end = time()
        print('tensorflow time using: ', end - start)
        tf_new_embedding = sess.run([tf_embedding])[0]
        print(tf_out)
        print(tf_new_embedding)
    
    print()
    
    # athena part
    embedding = ad.Variable('embeddingtable', value=value)
    index = ad.Variable(name="index")
    
    ids = ndarray.array(ids, ctx = executor_ctx)
    y= ad.embedding_lookup_op(embedding, index)
    athena_opts = {
        'sgd': optimizer.SGDOptimizer(0.1),
        'momentum': optimizer.MomentumOptimizer(0.1),
        'nesterov': optimizer.MomentumOptimizer(0.1, nesterov=True),
        'adagrad': optimizer.AdaGradOptimizer(0.1),
        'adam': optimizer.AdamOptimizer(0.1),
    }
    opt = athena_opts[opt_name]
    
    train_op = opt.minimize(y)
    executor = ad.Executor([y, train_op],ctx=executor_ctx)

    # print("embedding:", embedding.tensor_value.asnumpy())
    # print("ids:", ids.asnumpy())
    start = time()
    for i in range(iters):
        out, _ = executor.run(feed_dict={index: ids})
    end = time()
    print('athena time using: ', end - start)
    out = out.asnumpy()
    new_embedding = embedding.tensor_value.asnumpy()
    print(out)
    print(new_embedding)

    np.testing.assert_allclose(out, tf_out, rtol=1e-5)
    np.testing.assert_allclose(new_embedding, tf_new_embedding, rtol=1e-5)

test_embedding()
test_embedding(ndarray.cpu(0))
test_embedding_with_tf(opt_name='sgd')
test_embedding_with_tf(opt_name='sgd', executor_ctx=ndarray.cpu(0))
test_embedding_with_tf(opt_name='momentum')
test_embedding_with_tf(opt_name='nesterov', iters=1000)
test_embedding_with_tf(opt_name='adagrad')
test_embedding_with_tf(opt_name='adam')

