import numpy as np
import tensorflow as tf
import time
import argparse
from tqdm import tqdm
from sklearn import metrics
import horovod.tensorflow as hvd


def train_criteo(model, args):
    hvd.init()

    def get_current_shard(data):
        part_size = data.shape[0] // hvd.size()
        start = part_size * hvd.rank()
        end = start + part_size if hvd.rank() != hvd.size() - 1 else data.shape[0]
        return data[start:end]

    if args.all:
        from models.load_data import process_all_criteo_data
        dense, sparse, all_labels = process_all_criteo_data()
        dense_feature = get_current_shard(dense[0])
        sparse_feature = get_current_shard(sparse[0])
        labels = get_current_shard(all_labels[0])
        val_dense = get_current_shard(dense[1])
        val_sparse = get_current_shard(sparse[1])
        val_labels = get_current_shard(all_labels[1])
    else:
        from models.load_data import process_sampled_criteo_data
        dense_feature, sparse_feature, labels = process_sampled_criteo_data()
        dense_feature = get_current_shard(dense_feature)
        sparse_feature = get_current_shard(sparse_feature)
        labels = get_current_shard(labels)

    batch_size = 128
    dense_input = tf.placeholder(tf.float32, [batch_size, 13])
    sparse_input = tf.placeholder(tf.int32, [batch_size, 26])
    y_ = y_ = tf.placeholder(tf.float32, [batch_size, 1])

    loss, y, train_op = model(dense_input, sparse_input, y_, use_hvd=True)

    init=tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(visible_device_list=str(hvd.local_rank()))
    # hooks = [hvd.BroadcastGlobalVariablesHook(0)]
    sess = tf.train.MonitoredTrainingSession(config=tf.ConfigProto(gpu_options=gpu_options))
    # sess = tf.train.MonitoredTrainingSession(hooks=hooks, config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(init)
    
    my_feed_dict = {
        dense_input: np.empty(shape = (batch_size, 13)),
        sparse_input: np.empty(shape = (batch_size, 26)),
        y_: np.empty(shape = (batch_size, 1)),
    }

    if args.all:
        raw_log_file = './logs/tf_hvd_%s_%d.log' % (args.model, hvd.local_rank())
        print('Processing all data, log to', raw_log_file)
        log_file = open(raw_log_file, 'w')
        iterations = dense_feature.shape[0] // batch_size
        total_epoch = 400
        start_index = 0
        for ep in range(total_epoch):
            # print("iters: %d" % (lp * 1000))
            print("epoch %d" % ep)
            st_time = time.time()
            train_loss, train_acc, train_auc = [], [], []
            for it in tqdm(range(iterations // 10 + (ep % 10 == 9) * (iterations % 10))):
                my_feed_dict[dense_input][:] = dense_feature[start_index: start_index + batch_size]
                my_feed_dict[sparse_input][:] = sparse_feature[start_index : start_index + batch_size]
                my_feed_dict[y_][:] = labels[start_index : start_index+batch_size]
                start_index += batch_size
                if start_index + batch_size > dense_feature.shape[0]:
                    start_index = 0
                loss_val = sess.run([loss, y, y_, train_op], feed_dict=my_feed_dict)
                pred_val = loss_val[1]
                true_val = loss_val[2]
                acc_val = np.equal(
                    true_val,
                    pred_val > 0.5)
                train_loss.append(loss_val[0])
                train_acc.append(acc_val)
                train_auc.append(metrics.roc_auc_score(true_val, pred_val))
            tra_accuracy = np.mean(train_acc)
            tra_loss = np.mean(train_loss)
            tra_auc = np.mean(train_auc)
            en_time = time.time()
            train_time = en_time - st_time
            printstr = "train_loss: %.4f, train_acc: %.4f, train_auc: %.4f, train_time: %.4f"\
                        % (tra_loss, tra_accuracy, tra_auc, train_time)
            print(printstr)
            log_file.write(printstr + '\n')
            log_file.flush()

    else:
        iteration = dense_feature.shape[0] // batch_size

        epoch = 50
        for ep in range(epoch):
            print('epoch', ep)
            if ep == 5:
                start = time.time()
            ep_st = time.time()
            train_loss = []
            train_acc = []
            for idx in range(iteration):
                start_index = idx * batch_size
                my_feed_dict[dense_input][:] = dense_feature[start_index: start_index + batch_size]
                my_feed_dict[sparse_input][:] = sparse_feature[start_index : start_index + batch_size]
                my_feed_dict[y_][:] = labels[start_index : start_index+batch_size]  

                loss_val = sess.run([loss, y, y_, train_op], feed_dict=my_feed_dict)
                pred_val = loss_val[1]
                true_val = loss_val[2]
                if pred_val.shape[1] == 1: # for criteo case
                    acc_val = np.equal(
                        true_val,
                        pred_val > 0.5)
                else:
                    acc_val = np.equal(
                        np.argmax(pred_val, 1),
                        np.argmax(true_val, 1)).astype(np.float)
                train_loss.append(loss_val[0])
                train_acc.append(acc_val)
            tra_accuracy = np.mean(train_acc)
            tra_loss = np.mean(train_loss)
            ep_en = time.time()
            print("train_loss: %.4f, train_acc: %.4f, train_time: %.4f"
                % (tra_loss, tra_accuracy, ep_en - ep_st))
        print('all time:', (time.time() - start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model to be tested")
    parser.add_argument("--all", action="store_true", help="whether to use all data")
    args = parser.parse_args()
    raw_model = args.model
    import tf_models
    model = eval('tf_models.' + raw_model)
    dataset = raw_model.split('_')[-1]
    print('Model:', raw_model)
    train_criteo(model, args)

if __name__ == '__main__':
    main()
