from GNN.layer import *
from GNN.graph import *
from GNN import distributed
from GNN.distributed.sampler import DistributedGraphSageSampler

from athena import ndarray, optimizer
from athena import gpu_ops as ad
from athena import dataloader as dl
from athena.communicator.mpi_nccl_comm import ncclDataType_t, ncclRedOp_t

import numpy as np
import time, os, sys
import yaml
import multiprocessing
import argparse
import signal
from tqdm import tqdm

# ../../build/_deps/openmpi-build/bin/mpirun -np 4 --allow-run-as-root python3 test_sparse_hybrid.py

class TrainStat():
    def __init__(self):
        self.file = open("log.txt", "w")
        self.train_stat = np.zeros(4)
        self.test_stat = np.zeros(4)
        self.count = 0
        self.time = []

    def update_test(self, cnt, total, loss):
        self.test_stat += [1, cnt, total, loss]

    def update_train(self, cnt, total, loss):
        self.train_stat += [1, cnt, total, loss]

    def sync_and_clear(self):
        self.count += 1
        train_stat = ndarray.array(self.train_stat, ndarray.cpu())
        test_stat = ndarray.array(self.test_stat, ndarray.cpu())
        comm.dlarrayNcclAllReduce(train_stat, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclSum, comm.stream)
        comm.dlarrayNcclAllReduce(test_stat, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclSum, comm.stream)
        comm.stream.sync()
        train_stat, test_stat = train_stat.asnumpy(), test_stat.asnumpy()
        printstr = "epoch {}: test loss: {:.3f} test acc: {:.3f} train loss: {:.3f} train acc: {:.3f}".format(
            self.count,
            test_stat[3] / test_stat[0],
            test_stat[1] / test_stat[2],
            train_stat[3] / train_stat[0],
            train_stat[1] / train_stat[2],
        )
        logstr = "{} {} {} {}".format(
            test_stat[3] / test_stat[0],
            test_stat[1] / test_stat[2],
            train_stat[3] / train_stat[0],
            train_stat[1] / train_stat[2],
        )
        self.time.append(time.time())
        if comm.device_id.value == 0:
            print(printstr, flush=True)
            print(logstr, file=self.file, flush=True)
            if len(self.time) > 3:
                epoch_time = np.array(self.time[1:])-np.array(self.time[:-1])
                print("epoch time: {:.3f}+-{:.3f}".format(np.mean(epoch_time), np.var(epoch_time)))

        self.train_stat[:] = 0
        self.test_stat[:] = 0


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
        mp_val = mp_matrix(graph, ndarray.gpu(device_id))
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
    ad.worker_init()
    rank = ad.get_worker_communicate().rank()
    nrank = int(os.environ["DMLC_NUM_WORKER"])
    ctx = ndarray.gpu(device_id)
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
    distributed.ps_init(rank, nrank)

    ngraph = meta["partition"]["nodes"][rank] // args.batch_size
    nbatch = meta["node"] // args.batch_size // nrank
    graphs = prepare_data(ngraph)
    idx = 0
    g_sample, mp_val, mask, mask_eval = graphs[idx]
    idx = (idx + 1) % ngraph
    dl.GNNDataLoaderOp.step(g_sample)
    dl.GNNDataLoaderOp.step(g_sample)
    epoch, batch_idx = 0, 0
    executor = ad.Executor([loss, y, train_op], ctx=ctx, comm_mode='Hybrid', use_sparse_pull=False ,cstable_policy=args.cache)
    train_state = TrainStat()
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
        train_state.update_train(train_acc, mask.sum(), np.sum(loss_val.asnumpy()*mask)/mask.sum())
        train_state.update_test(acc, mask_eval.sum(), np.sum(loss_val.asnumpy()*mask_eval)/mask_eval.sum())
        batch_idx += 1
        if batch_idx == nbatch:
            batch_idx = 0
            epoch += 1
            train_state.sync_and_clear()
            if epoch >= num_epoch:
                break
        g_sample, mp_val, mask, mask_eval = g_sample_nxt, mp_val_nxt, mask_nxt, mask_eval_nxt

def signal_handler(signal, frame):
    print("SIGINT signal caught, stop Training")
    exit(0)

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
    comm, device_id = ad.mpi_nccl_init()
    file_path = args.config
    settings = yaml.load(open(file_path).read(), Loader=yaml.FullLoader)
    for k, v in settings['shared'].items():
        os.environ[k] = str(v)
    os.environ["DMLC_ROLE"] = "worker"
    signal.signal(signal.SIGINT, signal_handler)
    train_main(args)
    ad.mpi_nccl_finish(comm)
