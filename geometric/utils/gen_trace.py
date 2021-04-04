from athena import ndarray
from athena import gpu_ops as ad
import sys, os
import yaml
import multiprocessing
import argparse

from GNN.dataset import load_dataset
from GNN.graph import ps, DistributedDemo

args = None

def worker():
    rank = int(os.environ["WORKER_ID"])
    nrank = int(os.environ["DMLC_NUM_WORKER"])
    ps.ps_init(rank, nrank)
    if rank == 0:
        f = open(args.output, "w")
        ps.ps_set_trace(f)

    graph = load_dataset(args.dataset)
    walk_length = int(args.walk)
    num_head = int(graph.num_nodes / nrank / walk_length / 10)
    with DistributedDemo(graph, num_head, walk_length, rank=rank, nrank=nrank) as sampler:
        while rank == 0 and ps.PS.trace_len < int(args.length):
            g = sampler.sample()
            print("{}/{}".format(ps.PS.trace_len, args.length))
        ps.ps_get_worker_communicator().Barrier_Worker()
    if rank == 0:
        f.close()


def main(setting):
    for key, value in setting.items():
        os.environ[key] = str(value)
    if os.environ['DMLC_ROLE'] == "server":
        ad.server_init()
        ad.server_finish()
    elif os.environ['DMLC_ROLE'] == "worker":
        ad.worker_init()
        worker()
        ad.worker_finish()
    elif os.environ['DMLC_ROLE'] == "scheduler":
        ad.scheduler_init()
        ad.scheduler_finish()
    else:
        raise ValueError("Unknown role", os.environ['DMLC_ROLE'])

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--output", "-o", default="trace.txt")
    parser.add_argument("--dataset", default="Reddit")
    parser.add_argument("--length", "-l", default=100000)
    parser.add_argument("--walk", default=2)
    args = parser.parse_args()
    file_path = args.config
    settings = yaml.load(open(file_path).read(), Loader=yaml.FullLoader)
    process_list = []
    for key, value in settings.items():
        if key != 'shared':
            proc = multiprocessing.Process(target=main, args=[value])
            process_list.append(proc)
            proc.start()
    for proc in process_list:
        proc.join()