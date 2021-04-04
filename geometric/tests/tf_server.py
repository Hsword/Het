import os
import tensorflow as tf
import multiprocessing
import signal
import json
import sys
import argparse

os.environ["CUDA_VISIBLE_DEVICES"]=""
def pop_env():
    for k in ['https_proxy', 'http_proxy']:
        if k in os.environ:
            os.environ.pop(k)
pop_env()

def start_server(cluster, task):
    server = tf.train.Server(cluster, job_name='ps', task_index=task)
    server.join()


def main():
    raw_config = '../config/tf_config.json'
    config = json.load(open(raw_config))
    cluster=tf.train.ClusterSpec(config)
    global proc
    proc = multiprocessing.Process(target=start_server, args=[cluster, int(args.i)])
    proc.start()
    signal.signal(signal.SIGINT, signal_handler)
    proc.join()


def signal_handler(signal, frame):
    print("SIGINT signal caught, stop Training")
    global proc
    proc.kill()
    exit(0)


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("i")
    args = parser.parse_args()
    main()
