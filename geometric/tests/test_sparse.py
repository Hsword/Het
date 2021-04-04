from GNN.layer import *
from GNN.graph import *
from GNN import distributed
from GNN.distributed.sampler import DistributedGraphSageSampler

from athena import ndarray, optimizer
from athena import gpu_ops as ad
from athena import dataloader as dl
from athena.launcher import launch

import numpy as np
import time, os, sys
import yaml
import multiprocessing
import argparse
import signal
from tqdm import tqdm

logfile = open("log.txt", "w")

class SharedTrainingStat():
    def __init__(self):
        self.manager = multiprocessing.Manager()
        self.lock = self.manager.Lock()
        self.total = self.manager.Value("total", 0)
        self.acc = self.manager.Value("acc", 0)
        self.loss = self.manager.Value("loss", 0.0)
        self.count = self.manager.Value("count", 0)
        self.train_total = self.manager.Value("train_total", 0)
        self.train_acc = self.manager.Value("train_acc", 0)
        self.train_loss = self.manager.Value("train_loss", 0.0)
        self.train_count = self.manager.Value("train_count", 0)
        self.time = []

    def update(self, acc, total, loss):
        self.lock.acquire()
        self.total.value += total
        self.acc.value += acc
        self.loss.value += loss
        self.count.value += 1
        self.lock.release()

    def update_train(self, acc, total, loss):
        self.lock.acquire()
        self.train_total.value += total
        self.train_acc.value += acc
        self.train_loss.value += loss
        self.train_count.value += 1
        self.lock.release()

    def print(self, start=""):
        self.lock.acquire()
        if len(self.time) > 3:
            epoch_time = np.array(self.time[1:])-np.array(self.time[:-1])
            print("epoch time: {:.3f}+-{:.3f}".format(np.mean(epoch_time), np.var(epoch_time)))
        self.time.append(time.time())
        print(
            start,
            "test loss: {:.3f} test acc: {:.3f} train loss: {:.3f} train acc: {:.3f}".format(
                self.loss.value / self.count.value,
                self.acc.value / self.total.value,
                self.train_loss.value / self.train_count.value,
                self.train_acc.value / self.train_total.value
            )
        )
        print(
            self.loss.value / self.count.value, self.acc.value / self.total.value,
            self.train_loss.value / self.train_count.value, self.train_acc.value / self.train_total.value,
            file=logfile, flush=True
        )
        self.total.value = 0
        self.acc.value = 0
        self.loss.value = 0
        self.count.value = 0
        self.train_total.value = 0
        self.train_acc.value = 0
        self.train_loss.value = 0
        self.train_count.value = 0
        self.lock.release()


def convert_to_one_hot(vals, max_val = 0):
    """Helper method to convert label array to one-hot array."""
    if max_val == 0:
      max_val = vals.max() + 1
    one_hot_vals = np.zeros((vals.size, max_val))
    one_hot_vals[np.arange(vals.size), vals] = 1
    return one_hot_vals

def padding(graph, target):
    assert graph.num_nodes <= target
    extra = target - graph.num_nodes
    x = np.concatenate([graph.x, np.tile(graph.x[0], [extra, 1])])
    y = np.concatenate([graph.y, np.repeat(graph.y[0], extra)])
    return Graph(x, y, graph.edge_index, graph.num_classes)

def prepare_data(ngraph):
    rank = ad.get_worker_communicate().rank()
    nrank = int(os.environ["DMLC_NUM_WORKER"])
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
        mp_val = mp_matrix(graph, ndarray.gpu(rank))
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
    rank = ad.get_worker_communicate().rank() % args.num_local_worker
    nrank = int(os.environ["DMLC_NUM_WORKER"])
    ctx = ndarray.gpu(rank)
    embedding_width = args.hidden_size
    extract_width = embedding_width*(meta["feature"]-1)

    y_ = dl.GNNDataLoaderOp(lambda g: ndarray.array(convert_to_one_hot(g.y, max_val=g.num_classes), ctx=ndarray.cpu()))
    mask_ = ad.Variable(name="mask_")
    gcn1 = GCN(extract_width, hidden_layer_size, activation="relu")
    gcn2 = GCN(hidden_layer_size, meta["class"])
    index = dl.GNNDataLoaderOp(lambda g: ndarray.array(g.x[:, 0:-1], ctx=ndarray.cpu()), ctx=ndarray.cpu())
    embedding = initializers.random_normal([meta["idx_max"], embedding_width], stddev=0.1)
    embed =  ad.embedding_lookup_op(embedding, index)
    embed = ad.array_reshape_op(embed, (-1, extract_width))
    # embed = ad.reduce_mean_op(embed, axes=1)
    # x = ad.concat_op(x_, embed, axis=1)
    x = gcn1(embed)
    y = gcn2(x)
    loss = ad.softmaxcrossentropy_op(y, y_)
    train_loss = loss * mask_
    train_loss = ad.reduce_mean_op(train_loss, [0])
    opt = optimizer.SGDOptimizer(args.learning_rate)
    train_op = opt.minimize(train_loss)
    ad.worker_init()
    distributed.ps_init(rank, nrank)


    ngraph = meta["partition"]["nodes"][rank] // args.batch_size
    graphs = prepare_data(ngraph)
    idx = 0
    g_sample, mp_val, mask, mask_eval = graphs[idx]
    idx = (idx + 1) % ngraph
    dl.GNNDataLoaderOp.step(g_sample)
    dl.GNNDataLoaderOp.step(g_sample)
    epoch = 0
    nnodes = 0
    executor = ad.Executor([loss, y, train_op], ctx=ctx, comm_mode='PS', use_sparse_pull=False ,cstable_policy=args.cache)
    while True:
        g_sample_nxt, mp_val_nxt, mask_nxt, mask_eval_nxt = graphs[idx]
        idx = (idx + 1) % ngraph
        dl.GNNDataLoaderOp.step(g_sample_nxt)
        feed_dict = {
            gcn1.mp : mp_val,
            gcn2.mp : mp_val,
            mask_ : mask
        }
        loss_val, y_predicted, _ = executor.run(feed_dict = feed_dict)
        y_predicted = y_predicted.asnumpy().argmax(axis=1)

        acc = np.sum((y_predicted == g_sample.y) * mask_eval)
        train_acc = np.sum((y_predicted == g_sample.y) * mask)
        stat.update(acc, mask_eval.sum(), np.sum(loss_val.asnumpy()*mask_eval)/mask_eval.sum())
        stat.update_train(train_acc, mask.sum(), np.sum(loss_val.asnumpy()*mask)/mask.sum())

        distributed.ps_get_worker_communicator().BarrierWorker()
        nnodes += mask.sum() + mask_eval.sum()
        if nnodes > meta["partition"]["nodes"][rank]:
            nnodes = 0
            epoch += 1
            if rank == 0:
                stat.print(epoch)
            if epoch >= num_epoch:
                break
        g_sample, mp_val, mask, mask_eval = g_sample_nxt, mp_val_nxt, mask_nxt, mask_eval_nxt

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--path", "-p", required=True)
    parser.add_argument("--num_epoch", default=300, type=int)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--learning_rate", default=1, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--cache", default="LFUOpt", type=str)
    args = parser.parse_args()
    stat = SharedTrainingStat()
    launch(train_main, args)
