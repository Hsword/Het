from athena import ndarray
from athena import gpu_ops as ad

import os
import numpy as np
import time
import yaml
import multiprocessing
import signal
import numpy as np
import argparse
from sklearn import metrics

# ../build/deps/bin/mpirun --allow-run-as-root -np 8 python run_hybrid_local.py --model
# ../build/_deps/openmpi-build/bin/mpirun -np 8 --allow-run-as-root python run_hybrid_local.py --model

def worker(model, rank, args):
    def train(iterations, auc_enabled=True):
        train_loss, train_acc, train_auc = [], [], []
        for it in range(iterations):
            loss_val, predict_y, y_val, _ = executor.run(convert_to_numpy_ret_vals=True)
            if y_val.shape[1] == 1: # for criteo case
                acc_val = np.equal(
                    y_val,
                    predict_y > 0.5).astype(np.float)
            else:
                acc_val = np.equal(
                    np.argmax(y_val, 1),
                    np.argmax(predict_y, 1)).astype(np.float)
            train_loss.append(loss_val[0])
            train_acc.append(acc_val)
            if auc_enabled:
                train_auc.append(metrics.roc_auc_score(y_val, predict_y))
        if auc_enabled:
            return np.mean(train_loss), np.mean(train_acc), np.mean(train_auc)
        else:
            return np.mean(train_loss), np.mean(train_acc)
    def validate(iterations):
        test_loss, test_acc, test_auc = [], [], []
        for it in range(iterations):
            loss_val, test_y_predicted, y_test_val = val_executor.run(convert_to_numpy_ret_vals=True)
            if y_test_val.shape[1] == 1: # for criteo case
                correct_prediction = np.equal(
                    y_test_val,
                    test_y_predicted > 0.5).astype(np.float)
            else:
                correct_prediction = np.equal(
                    np.argmax(y_test_val, 1),
                    np.argmax(test_y_predicted, 1)).astype(np.float)
            test_loss.append(loss_val[0])
            test_acc.append(correct_prediction)
            test_auc.append(metrics.roc_auc_score(y_test_val, test_y_predicted))
        return np.mean(test_loss), np.mean(test_acc), np.mean(test_auc)
    
    if args.all:
        from models.load_data import process_all_criteo_data
        dense, sparse, labels = process_all_criteo_data(return_val=args.val)
    elif args.val:
        from models.load_data import process_head_criteo_data
        dense, sparse, labels = process_head_criteo_data(return_val=True)
    else:
        from models.load_data import process_sampled_criteo_data
        dense, sparse, labels = process_sampled_criteo_data()
    loss, prediction, y_, train_op = model(dense, sparse, labels)

    executor = ad.Executor([loss, prediction, y_, train_op], ctx=ndarray.gpu(rank),\
        dataloader_name='train', stream_mode='AllStreams', comm_mode='Hybrid', use_sparse_pull=True, cstable_policy=args.cache, bsp=args.bsp, seed=123, cache_bound=args.bound)
    if args.val:
        val_executor = ad.Executor([loss, prediction, y_], ctx=ndarray.gpu(rank),\
            dataloader_name='validate', stream_mode='AllStreams', comm_mode='Hybrid', use_sparse_pull=True, inference=True)
    
    if args.all:
        raw_log_file = './logs/localhybrid_%s' % (args.model)
        if args.bsp:
            raw_log_file += '_bsp'
        else:
            raw_log_file += '_asp'
        if args.cache:
            raw_log_file += '_%s' % (args.cache)
        raw_log_file += '_%d.log' % (rank)
        print('Processing all data, log to', raw_log_file)
        log_file = open(raw_log_file, 'w')
        total_loop = 20 * (executor.batch_num // 1000)
        for lp in range(total_loop):
            print("iters: %d" % (lp * 1000))
            train_loss, train_acc, train_auc = train(1000)
            if args.val:
                val_loss, val_acc, val_auc = validate(100)
                printstr = "train_loss: %.4f, train_acc: %.4f, train_auc: %.4f, test_loss: %.4f, test_acc: %.4f, test_auc: %.4f"\
                        % (train_loss, train_acc, train_auc, val_loss, val_acc, val_auc)
            else:
                printstr = "train_loss: %.4f, train_acc: %.4f, train_auc: %.4f"\
                        % (train_loss, train_acc, train_auc)
            print(printstr)
            log_file.write(printstr + '\n')
            if lp % 5 == 0:
                log_file.flush()
    else:
        total_epoch = 20
        iterations = executor.batch_num

        for ep in range(total_epoch):
            if ep == 5:
                start = time.time()
            print("epoch %d" % ep)
            ep_st = time.time()
            train_loss, train_acc = train(executor.batch_num, auc_enabled=False)
            ep_en = time.time()
            if args.val:
                executor.ps_comm.BarrierWorker()
                val_loss, val_acc, val_auc = validate(val_executor.batch_num)
                executor.ps_comm.BarrierWorker()
                print("train_loss: %.4f, train_acc: %.4f, train_time: %.4f, test_loss: %.4f, test_acc: %.4f, test_auc: %.4f"
                        % (train_loss, train_acc, ep_en - ep_st, val_loss, val_acc, val_auc))
            else:
                print("train_loss: %.4f, train_acc: %.4f, train_time: %.4f"
                        % (train_loss, train_acc, ep_en - ep_st))
        print('all time:', time.time() - start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model to be tested")
    parser.add_argument("--config", type=str, default="./settings/local_w8.yml", help="configuration for ps")
    parser.add_argument("--val", action="store_true", help="whether to use validation")
    parser.add_argument("--cache", default=None, help="cache policy")
    parser.add_argument("--bsp", action="store_true", help="whether to use bsp instead of asp")
    parser.add_argument("--all", action="store_true", help="whether to use all data")
    parser.add_argument("--bound", default=100, help="cache bound")
    args = parser.parse_args()
    config = args.config
    import models
    model = eval('models.' + args.model)
    settings = yaml.load(open(config).read(), Loader=yaml.FullLoader)
    comm, device_id = ad.mpi_nccl_init()
    print('Model:', args.model, '; rank:', device_id)
    # value = settings['w' + str(device_id)]
    value = settings['shared']
    os.environ['DMLC_ROLE'] = 'worker'
    for k, v in value.items():
        os.environ[k] = str(v)
    worker(model, device_id, args)
    ad.mpi_nccl_finish(comm)


if __name__ =='__main__':
    main()
