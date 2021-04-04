from GNN.layer import *
from GNN.graph import *
from GNN import distributed
from GNN.distributed.sampler import DistributedSubgraphSampler

from athena import ndarray, optimizer
from athena import gpu_ops as ad

import numpy as np
import time, os, sys
import yaml
import multiprocessing
import argparse
import signal

def convert_to_one_hot(vals, max_val = 0):
    """Helper method to convert label array to one-hot array."""
    if max_val == 0:
      max_val = vals.max() + 1
    one_hot_vals = np.zeros((vals.size, max_val))
    one_hot_vals[np.arange(vals.size), vals] = 1
    return one_hot_vals

def train_athena(args):
    with open(os.path.join(args.path, "meta.yml"), 'rb') as f:
        meta = yaml.load(f.read(), Loader=yaml.FullLoader)
    hidden_layer_size = args.hidden_size
    num_epoch = args.num_epoch
    rank = int(os.environ["WORKER_ID"])
    nrank = int(os.environ["DMLC_NUM_WORKER"])
    ctx = ndarray.gpu(rank)

    x_ = ad.Variable(name="x_")
    y_ = ad.Variable(name="y_")
    gcn1 = GCN(meta["feature"], hidden_layer_size, activation="relu")
    gcn2 = GCN(hidden_layer_size, meta["class"])
    x = gcn1(x_)
    y = gcn2(x)
    loss = ad.softmaxcrossentropy_op(y, y_)
    loss = ad.reduce_mean_op(loss, [0])
    opt = optimizer.SGDOptimizer(0.1)
    train_op = opt.minimize(loss)
    executor = ad.Executor([loss, y, train_op], ctx=ctx, comm_mode='PS')
    distributed.ps_init(rank, nrank)

    def transform(graph):
        mp_val = mp_matrix(graph, ndarray.gpu(rank))
        return graph, mp_val
    with DistributedSubgraphSampler(args.path, 4000, 2, rank=rank, nrank=nrank ,transformer=transform, cache_size_factor=0, reduce_nonlocal_factor=0.5) as sampler:
        epoch = 0
        nnodes = 0
        start = time.time()
        while True:
            g_sample, mp_val = sampler.sample()
            feed_dict = {
                gcn1.mp : mp_val,
                gcn2.mp : mp_val,
                x_ : ndarray.array(g_sample.x, ctx=ctx),
                y_ : ndarray.array(convert_to_one_hot(g_sample.y, max_val=g_sample.num_classes), ctx=ctx)
            }
            loss_val, y_predicted, _ = executor.run(feed_dict = feed_dict)
            y_predicted = y_predicted.asnumpy().argmax(axis=1)
            acc = (y_predicted == g_sample.y).sum()
            distributed.ps_get_worker_communicator().BarrierWorker()
            nnodes += g_sample.num_nodes
            if nnodes > meta["partition"]["nodes"][rank]:
                nnodes = 0
                epoch += 1
                print("Epoch :", epoch, time.time() - start)
                print("Train accuracy:", acc/len(y_predicted))
                start = time.time()
                if epoch >= num_epoch:
                    break

def start_process(settings, args):
    for key, value in settings.items():
        os.environ[key] = str(value)
    if os.environ['DMLC_ROLE'] == "server":
        ad.server_init()
        ad.server_finish()
    elif os.environ['DMLC_ROLE'] == "worker":
        train_athena(args)
    elif os.environ['DMLC_ROLE'] == "scheduler":
        ad.scheduler_init()
        ad.scheduler_finish()
    else:
        raise ValueError("Unknown role", os.environ['DMLC_ROLE'])

def signal_handler(signal, frame):
    print("SIGINT signal caught, stop Training")
    for proc in process_list:
        proc.kill()
    exit(0)

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--path", "-p", required=True)
    parser.add_argument("--num_epoch", default=100, type=int)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--learning_rate", default=0.1, type=float)
    args = parser.parse_args()
    file_path = args.config
    settings = yaml.load(open(file_path).read(), Loader=yaml.FullLoader)
    process_list = []
    for key, value in settings.items():
        if key != 'shared':
            proc = multiprocessing.Process(target=start_process, args=[value, args])
            process_list.append(proc)
            proc.start()
    signal.signal(signal.SIGINT, signal_handler)
    for proc in process_list:
        proc.join()
