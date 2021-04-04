from athena import ndarray, optimizer
from athena import gpu_ops as ad

import time, os, sys
import yaml
import multiprocessing
import argparse
import signal
import numpy as np
import ctypes

from concurrent.futures import ThreadPoolExecutor
import threading

def pointer(arr):
    assert(arr.data.c_contiguous)
    assert(arr.dtype == np.long)
    return ctypes.cast(arr.ctypes.data, ctypes.POINTER(ctypes.c_long))

nitem = 2000
item_len = 1000
max_thread = 10

def test():
    ctx = ndarray.cpu(0)
    rank = int(os.environ["WORKER_ID"])
    nrank = int(os.environ["DMLC_NUM_WORKER"])
    if rank > 0:
        return
    arr = ndarray.array(np.random.rand(nitem, item_len),ctx = ctx) # generate a long buffer

    push_indices = np.arange(nitem)
    print(push_indices)
    push_length = np.repeat(item_len, repeats=nitem)
    worker_communicate = ad.get_worker_communicate()
    query = worker_communicate.PushData(pointer(push_indices), nitem, arr.handle, pointer(push_length))
    worker_communicate.WaitData(query)
    print("data_pushed")
    t = ThreadPoolExecutor(max_workers=max_thread)
    byte_count = 0
    arr2 = ndarray.array(np.random.rand(nitem, item_len),ctx = ctx)
    def pull_data():
        query = worker_communicate.PullData(pointer(push_indices), nitem, arr2.handle, pointer(push_length))
        worker_communicate.WaitData(query)
        # print( np.all(arr.asnumpy() == arr2.asnumpy()) )
        nonlocal byte_count
        byte_count += nitem * item_len * 4
    def watch():
        nonlocal byte_count
        start = time.time()
        while True:
            time.sleep(1)
            speed = byte_count / (time.time() - start)
            print("speed : {} MB/s".format(speed / 2**20))
    task_list = [None for i in range(max_thread)]
    threading.Thread(target=watch).start()
    while True:
        for i in range(max_thread):
            if task_list[i] is None or task_list[i].done():
                task_list[i] = t.submit(pull_data)


def start_process(settings, args):
    for key, value in settings.items():
        os.environ[key] = str(value)
    if os.environ['DMLC_ROLE'] == "server":
        ad.server_init()
        ad.server_finish()
    elif os.environ['DMLC_ROLE'] == "worker":
        ad.worker_init()
        test()
        ad.worker_finish()
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
