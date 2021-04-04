from athena import ndarray
from athena import gpu_ops as ad

import numpy as np
import time
import argparse
from tqdm import tqdm
from sklearn import metrics


def main():
    def train(iterations, auc_enabled=True, tqdm_enabled=False):
        localiter = tqdm(range(iterations)) if tqdm_enabled else range(iterations)
        train_loss = []
        train_acc = []
        if auc_enabled:
            train_auc = []
        for it in localiter:
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
    def validate(iterations, tqdm_enabled=False):
        localiter = tqdm(range(iterations)) if tqdm_enabled else range(iterations)
        test_loss = []
        test_acc = []
        test_auc = []
        for it in localiter:
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model to be tested")
    parser.add_argument("--val", action="store_true", help="whether to use validation")
    parser.add_argument("--all", action="store_true", help="whether to use all data")
    args = parser.parse_args()
    import models
    model = eval('models.' + args.model)
    print('Model:', args.model)

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

    executor = ad.Executor([loss, prediction, y_, train_op], ctx=ndarray.gpu(0),\
        dataloader_name='train', stream_mode='AllStreams')
    if args.val:
        print('Validation enabled...')
        val_executor = ad.Executor([loss, prediction, y_], ctx=ndarray.gpu(0),\
            dataloader_name='validate', stream_mode='AllStreams', inference=True)

    if args.all:
        print('Processing all data...')
        log_file = open('./logs/local_' + args.model + '.log', 'w')
        total_epoch = 20
        for ep in range(total_epoch):
            print("ep: %d" % ep)
            ep_st = time.time()
            train_loss, train_acc, train_auc = train(executor.batch_num // 10 + (ep % 10 == 9) * (executor.batch_num % 10), tqdm_enabled=True)
            ep_en = time.time()
            if args.val:
                val_loss, val_acc, val_auc = validate(val_executor.batch_num)
                printstr = "train_loss: %.4f, train_acc: %.4f, train_auc: %.4f, test_loss: %.4f, test_acc: %.4f, test_auc: %.4f, train_time: %.4f"\
                        % (train_loss, train_acc, train_auc, val_loss, val_acc, val_auc, ep_en - ep_st)
            else:
                printstr = "train_loss: %.4f, train_acc: %.4f, train_auc: %.4f, train_time: %.4f"\
                        % (train_loss, train_acc, train_auc, ep_en - ep_st)
            print(printstr)
            log_file.write(printstr + '\n')
    else:
        total_epoch = 50
        for ep in range(total_epoch):
            if ep == 5:
                start = time.time()
            print("epoch %d" % ep)
            ep_st = time.time()
            train_loss, train_acc = train(executor.batch_num, auc_enabled=False)
            ep_en = time.time()
            if args.val:
                val_loss, val_acc, val_auc = validate(val_executor.batch_num)
                print("train_loss: %.4f, train_acc: %.4f, train_time: %.4f, test_loss: %.4f, test_acc: %.4f, test_auc: %.4f"
                        % (train_loss, train_acc, ep_en - ep_st, val_loss, val_acc, val_auc))
            else:
                print("train_loss: %.4f, train_acc: %.4f, train_time: %.4f"
                        % (train_loss, train_acc, ep_en - ep_st))
        print('all time:', time.time() - start)


if __name__ == '__main__':
    main()
