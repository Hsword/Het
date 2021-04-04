from athena import ndarray
from athena import gpu_ops as ad
from athena.launcher import launch

import numpy as np
import time
import numpy as np
import argparse
from sklearn import metrics
from tqdm import tqdm

def worker(args):
    model = args.model
    rank = ad.get_worker_communicate().rank()
    def train(iterations):
        train_loss, train_acc, train_auc = [], [], []
        for it in tqdm(range(iterations)):
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
            train_auc.append(metrics.roc_auc_score(y_val, predict_y))
        return np.mean(train_loss), np.mean(train_acc), np.mean(train_auc)
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

    from models.load_data import process_all_criteo_data
    dense, sparse, labels = process_all_criteo_data(return_val=args.val)
    loss, prediction, y_, train_op = model(dense, sparse, labels)

    executor = ad.Executor([loss, prediction, y_, train_op], ctx=ndarray.gpu(rank),\
        dataloader_name='train', stream_mode='AllStreams', comm_mode='PS', use_sparse_pull=True, cstable_policy=args.cache, bsp=args.bsp, cache_bound=args.bound)
    if args.val:
        print('Validation enabled...')
        val_executor = ad.Executor([loss, prediction, y_], ctx=ndarray.gpu(rank),\
            dataloader_name='validate', stream_mode='AllStreams', comm_mode='PS', use_sparse_pull=True, inference=True)

    raw_log_file = './logs/localps_%s' % (args.model)
    if args.bsp:
        raw_log_file += '_bsp'
    else:
        raw_log_file += '_asp'
    if args.cache:
        raw_log_file += '_%s' % (args.cache)
    raw_log_file += '_%d.log' % (rank)
    print('Processing all data, log to', raw_log_file)
    log_file = open(raw_log_file, 'w')
    # total_loop = 20 * (executor.batch_num // 1000)
    total_epoch = 400
    for ep in range(total_epoch):
        # print("iters: %d" % (lp * 1000))
        print("epoch %d" % ep)
        st_time = time.time()
        train_loss, train_acc, train_auc = train(executor.batch_num // 10 + (ep % 10 == 9) * (executor.batch_num % 10))
        en_time = time.time()
        train_time = en_time - st_time
        if args.val:
            executor.ps_comm.BarrierWorker()
            val_loss, val_acc, val_auc = validate(val_executor.batch_num)
            executor.ps_comm.BarrierWorker()
            printstr = "train_loss: %.4f, train_acc: %.4f, train_auc: %.4f, test_loss: %.4f, test_acc: %.4f, test_auc: %.4f, train_time: %.4f"\
                    % (train_loss, train_acc, train_auc, val_loss, val_acc, val_auc, train_time)
        else:
            printstr = "train_loss: %.4f, train_acc: %.4f, train_auc: %.4f, train_time: %.4f"\
                    % (train_loss, train_acc, train_auc, train_time)
        executor.recordLoads()
        print(printstr)
        log_file.write(printstr + '\n')
        log_file.flush()

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model to be tested")
    parser.add_argument("--config", type=str, default="../config/local_w4.yml", help="configuration for ps")
    parser.add_argument("--val", action="store_true", help="whether to use validation")
    parser.add_argument("--cache", default=None, help="cache policy")
    parser.add_argument("--bsp", action="store_true", help="whether to use bsp instead of asp")
    parser.add_argument("--bound", default=100, help="cache bound")
    args = parser.parse_args()
    import models
    model = eval('models.' + args.model)
    print('Model:', args.model)
    args.model = model
    launch(worker, args)
