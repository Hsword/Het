from athena import ndarray
from athena import gpu_ops as ad
from athena import optimizer
import models
from models.load_data import load_mnist_data, normalize_cifar10, convert_to_one_hot

import os
import json
import numpy as np
import argparse
from time import time


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='model to be tested')
    parser.add_argument('--settings', type=str, required=True, help='settings file to be used')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--opt', type=str, default='sgd', help='optimizer to be used, default sgd; sgd / momentum / adagrad / adam')
    parser.add_argument('--num-epochs', type=int, default=10, help='epoch number')
    parser.add_argument('--gpu', type=int, default=0, help='gpu to be used, -1 means cpu')
    parser.add_argument('--validate', action='store_true', help='whether to use validation')
    parser.add_argument('--timing', action='store_true', help='whether to time the training phase')
    args = parser.parse_args()

    settings_file = open(args.settings)
    settings = json.load(settings_file)

    for key in settings:
        if type(settings[key]) == str:
            os.environ[key] = settings[key]
        else:
            os.environ[key] = str(settings[key]) ## type is str
    
    if os.environ['DMLC_ROLE'] == "server":
        ad.server_init()
        ad.server_finish()
        exit()
    elif os.environ['DMLC_ROLE'] == "scheduler":
        ad.scheduler_init()
        ad.scheduler_finish()
        exit()

    if args.gpu == -1:
        executor_ctx = None
        print('Use CPU.')
    else:
        executor_ctx = ndarray.gpu(args.gpu)
        print('Use GPU %d.' % args.gpu)
    
    assert args.model in ['alexnet', 'cnn_3_layers', 'lenet', 'logreg', 'lstm', 'mlp', 'resnet18', 'resnet34', 'rnn', 'vgg16', 'vgg19'], \
        'Model not supported!'
    model = eval('models.' + args.model)
    if args.model in ['resnet18', 'resnet34', 'vgg16', 'vgg19']:
        dataset = 'CIFAR10'
    else:
        dataset = 'MNIST'
    
    assert args.opt in ['sgd', 'momentum', 'nesterov', 'adagrad', 'adam'], 'Optimizer not supported!'
    if args.opt == 'sgd':
        print('Use SGD Optimizer.')
        opt = optimizer.SGDOptimizer(learning_rate=args.learning_rate)
    else:
        assert False, 'Other optimizers not implemented yet!'
    # elif args.opt == 'momentum':
    #     print('Use Momentum Optimizer.')
    #     opt = optimizer.MomentumOptimizer(learning_rate=args.learning_rate)
    # elif args.opt == 'nesterov':
    #     print('Use Nesterov Momentum Optimizer.')
    #     opt = optimizer.MomentumOptimizer(learning_rate=args.learning_rate, nesterov=True)
    # elif args.opt == 'adagrad':
    #     print('Use AdaGrad Optimizer.')
    #     opt = optimizer.AdaGradOptimizer(learning_rate=args.learning_rate, initial_accumulator_value=0.1)
    # else:
    #     print('Use Adam Optimizer.')
    #     opt = optimizer.AdamOptimizer(learning_rate=args.learning_rate)

    # model definition
    print('Building model...')
    x = ad.Variable(name="x")
    y_ = ad.Variable(name="y_")
    loss, y = model(x, y_)
    train_op = opt.minimize(loss)
    executor = ad.Executor([loss, y, train_op], ctx=executor_ctx, comm_mode='PS')
    if args.validate:
        val_executor = ad.Executor([loss, y], ctx=executor_ctx, inference=True)

    # data loading
    print('Loading %s data...' % dataset)
    if dataset == 'MNIST':
        datasets = load_mnist_data("mnist.pkl.gz")
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]
        n_train_batches = train_set_x.shape[0] // args.batch_size
        n_valid_batches = valid_set_x.shape[0] // args.batch_size
        # train_set_x: (50000, 784), train_set_y: (50000,)
        # valid_set_x: (10000, 784), valid_set_y: (10000,)
    else:
        train_set_x, train_set_y, valid_set_x, valid_set_y = normalize_cifar10()
        n_train_batches = train_set_x.shape[0] // args.batch_size
        n_valid_batches = valid_set_x.shape[0] // args.batch_size
        # train_set_x: (50000, 3, 32, 32), train_set_y: (50000,)
        # valid_set_x: (10000, 3, 32, 32), valid_set_y: (10000,)

    # training
    print("Start training loop...")
    for i in range(args.num_epochs):
        print("Epoch %d" % i)
        loss_all = 0
        if args.timing:
            start = time()
        for minibatch_index in range(n_train_batches):
            minibatch_start = minibatch_index * args.batch_size
            minibatch_end = (minibatch_index + 1) * args.batch_size
            x_val = train_set_x[minibatch_start:minibatch_end]
            y_val = convert_to_one_hot(
                train_set_y[minibatch_start:minibatch_end], max_val= 10)
            loss_val, predict_y, _ = executor.run(feed_dict={x: x_val, y_: y_val})
            if args.gpu >= 0:
                loss_val = loss_val.asnumpy()
            loss_all += loss_val * len(x_val)
        loss_all /= len(train_set_x)
        print("Loss = %f" % loss_all)
        if args.timing:
            end = time()
            print("Time = %f" % (end - start))
            
        if args.validate:
            correct_predictions = []
            for minibatch_index in range(n_valid_batches):
                minibatch_start = minibatch_index * args.batch_size
                minibatch_end = (minibatch_index + 1) * args.batch_size
                valid_x_val = valid_set_x[minibatch_start:minibatch_end]
                valid_y_val = convert_to_one_hot(
                    valid_set_y[minibatch_start:minibatch_end], max_val = 10)
                loss_val, valid_y_predicted = val_executor.run(
                        feed_dict={x: valid_x_val, y_: valid_y_val}, convert_to_numpy_ret_vals=True)
                correct_prediction = np.equal(
                    np.argmax(valid_y_val, 1),
                    np.argmax(valid_y_predicted, 1)).astype(np.float)
                correct_predictions.extend(correct_prediction)
            accuracy = np.mean(correct_predictions)
            print("Validation accuracy = %f" % accuracy)
