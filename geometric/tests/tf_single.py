from GNN.graph import *
from GNN import distributed
from GNN.distributed.sampler import DistributedGraphSageSampler

from athena import gpu_ops as ad
from athena.launcher import launch

import numpy as np
import time, os, sys
import argparse
import yaml
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf

def tf_gcn(x, normed_adj, in_features, out_features):
    initializer = tf.keras.initializers.glorot_uniform()
    weight = tf.Variable(initializer(shape=[in_features, out_features]), dtype = tf.float32)
    bias = tf.Variable(tf.keras.initializers.zeros()([out_features]), dtype = tf.float32)
    x = tf.matmul(x, weight)
    x = x + bias
    x = tf.sparse.matmul(normed_adj, x)
    return x

def model(normed_adj, sparse_input, y_, train_mask):
    embedding_size = args.hidden_size
    num_feature = meta["feature"] - 1
    with tf.device("/cpu:0"):
        Embedding = tf.get_variable(
                name="Embedding",
                dtype=tf.float32,
                trainable=True,
                # pylint: disable=unnecessary-lambda
                shape=(meta["idx_max"], embedding_size),
                initializer=tf.random_normal_initializer(stddev=0.1)
                )
        sparse_input_embedding = tf.nn.embedding_lookup(Embedding, sparse_input)
        global_step = tf.Variable(0, name="global_step", trainable=False)
    with tf.device("/gpu:0"):
        x = tf.reshape(sparse_input_embedding, (-1, num_feature * embedding_size))
        x = tf_gcn(x, normed_adj, num_feature * embedding_size, embedding_size)
        x = tf.nn.relu(x)
        y = tf_gcn(x, normed_adj, embedding_size, meta["class"])
        y_ = tf.one_hot(y_, meta["class"])
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
        loss = loss * train_mask
        loss = tf.reduce_mean(loss)
        optimizer = tf.train.GradientDescentOptimizer(args.learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return loss, y, train_op

def padding(graph, target):
    assert graph.num_nodes <= target
    extra = target - graph.num_nodes
    x = np.concatenate([graph.x, np.tile(graph.x[0], [extra, 1])])
    y = np.concatenate([graph.y, np.repeat(graph.y[0], extra)])
    return Graph(x, y, graph.edge_index, graph.num_classes)

def prepare_data(ngraph):
    rank = ad.get_worker_communicate().rank()
    nrank = ad.get_worker_communicate().nrank()
    graphs = []
    graphsage_sample_depth = 2
    graphsage_sample_width = 2
    node_upper_bound = args.batch_size * ((graphsage_sample_width ** (graphsage_sample_depth + 1)) - 1)
    print("Start Sampling {} graphs".format(ngraph))
    def transform(result):
        [graph, sample_mask] = result
        train_mask = np.zeros(node_upper_bound)
        train_mask[0:graph.num_nodes] = sample_mask * graph.x[:, -1]
        test_mask = np.zeros(node_upper_bound)
        test_mask[0:graph.num_nodes] = (sample_mask - graph.x[:, -1]) * sample_mask
        graph = padding(graph, node_upper_bound)
        mp_val = mp_matrix(graph, 0, system="tensorflow")
        return graph, mp_val, train_mask, test_mask
    with DistributedGraphSageSampler(args.path, args.batch_size, graphsage_sample_depth, graphsage_sample_width,
        rank=rank, nrank=nrank ,transformer=transform, cache_size_factor=1, reduce_nonlocal_factor=0, num_sample_thread=4) as sampler:
        for i in tqdm(range(ngraph)):
            g_sample, mp_val, train_mask, test_mask = sampler.sample()
            graphs.append([g_sample, mp_val, train_mask, test_mask])
    return graphs

def train_main(args):
    with open(os.path.join(args.path, "meta.yml"), 'rb') as f:
        meta = yaml.load(f.read(), Loader=yaml.FullLoader)
    hidden_layer_size = args.hidden_size
    num_epoch = args.num_epoch
    rank = ad.get_worker_communicate().rank()
    device_id = rank % args.num_local_worker
    nrank = ad.get_worker_communicate().nrank()
    distributed.ps_init(rank, nrank)
    ngraph = meta["partition"]["nodes"][rank] // args.batch_size
    graphs = prepare_data(ngraph)
    idx, epoch, nnodes = 0, 0, 0
    worker_device = "gpu:0"
    graph_len = graphs[0][0].y.shape[0]
    with tf.device(worker_device):
        norm_adj = tf.compat.v1.sparse.placeholder(tf.float32, name="norm_adj")
        sparse_feature = tf.placeholder(tf.int32, [graph_len, meta["feature"] - 1])
        y_ = tf.placeholder(tf.int32, [graph_len], name="y_")
        train_mask = tf.placeholder(tf.float32, [graph_len], name="train_mask")
    loss, y, train_op = model(norm_adj, sparse_feature, y_, train_mask)
    init=tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(init)
    acc_cnt, total_cnt = 0, 0
    train_acc, train_cnt = 0, 0
    start = time.time()
    while True:
        g_sample, mp_val, mask, mask_eval = graphs[idx]
        idx = (idx + 1) % ngraph
        feed_dict = {
            norm_adj : mp_val,
            sparse_feature : g_sample.x[:, 0:-1],
            y_ : g_sample.y,
            train_mask : mask
        }
        loss_val = sess.run([loss, y, train_op], feed_dict=feed_dict)
        pred_val = loss_val[1]
        acc_val = np.equal(np.argmax(pred_val, 1), g_sample.y).astype(np.float)
        acc_cnt += (acc_val * mask_eval).sum()
        total_cnt +=  mask_eval.sum()
        nnodes += mask.sum() + mask_eval.sum()
        train_acc += (acc_val * mask).sum()
        train_cnt += mask.sum()
        if nnodes > meta["partition"]["nodes"][rank] // 10:
            nnodes = 0
            epoch += 1
            print("Acc : ", acc_cnt / total_cnt, train_acc / train_cnt ,"Time : ", time.time() - start)
            print(pred_val)
            start = time.time()
            acc_cnt, total_cnt = 0, 0
            train_acc, train_cnt = 0, 0
            if epoch >= num_epoch:
                break

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--path", "-p", required=True)
    parser.add_argument("--num_epoch", default=300, type=int)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--learning_rate", default=1, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    args = parser.parse_args()
    with open(os.path.join(args.path, "meta.yml"), 'rb') as f:
        meta = yaml.load(f.read(), Loader=yaml.FullLoader)
    launch(train_main, args)
