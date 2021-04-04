from athena import gpu_ops as ad
from athena import optimizer
from athena import dataloader as dl
from athena import initializers as init


def wdl_adult(whatever):
    batch_size = 128
    lr=5
    dim_wide = 809

    lr_ = lr / batch_size
    dim_deep = 68

    from .load_data import load_adult_data
    x_train_deep, x_train_wide, y_train, x_test_deep, x_test_wide, y_test = load_adult_data()

    W = init.random_normal([dim_wide+20, 2], stddev=0.1, name="W")
    W1 = init.random_normal([dim_deep, 50], stddev=0.1, name="W1")
    b1 = init.random_normal([50], stddev=0.1, name="b1")
    W2 = init.random_normal([50, 20], stddev=0.1, name="W2")
    b2 = init.random_normal([20], stddev=0.1, name="b2")

    X_wide = dl.dataloader_op([
        [x_train_wide, batch_size, 'train'],
        [x_test_wide, batch_size, 'validate'],
    ])
    y_ = dl.dataloader_op([
        [y_train, batch_size, 'train'],
        [y_test, batch_size, 'validate'],
    ])

    #deep
    Embedding = []
    X_deep = []
    X_deep_input = None

    for i in range(8):
        X_deep_name = "x_deep_" + str(i)
        Embedding_name = "Embedding_deep_" + str(i)
        X_deep.append(dl.dataloader_op([
            [x_train_deep[:,i], batch_size, 'train'],
            [x_test_deep[:,i], batch_size, 'validate'],
        ]))
        Embedding.append(init.random_normal([50, 8], stddev=0.1, name=Embedding_name))
        now = ad.embedding_lookup_op(Embedding[i], X_deep[i])
        now = ad.array_reshape_op(now, (-1, 8))
        if X_deep_input is None:
            X_deep_input = now
        else:
            X_deep_input = ad.concat_op(X_deep_input, now, 1)

    for i in range(4):
        X_deep_name = "x_deep_" + str(8+i)
        X_deep.append(dl.dataloader_op([
            [x_train_deep[:,8+i], batch_size, 'train'],
            [x_test_deep[:,8+i], batch_size, 'validate'],
        ]))
        now = ad.array_reshape_op(X_deep[i + 8], (batch_size, 1))
        X_deep_input = ad.concat_op(X_deep_input, now, 1)

    mat1 = ad.matmul_op(X_deep_input, W1)
    add1 = mat1 + ad.broadcastto_op(b1, mat1)
    relu1= ad.relu_op(add1)
    dropout1 = relu1 #ad.dropout_op(relu1, 0.5)
    mat2 = ad.matmul_op(dropout1, W2)
    add2 = mat2 + ad.broadcastto_op(b2, mat2)
    relu2= ad.relu_op(add2)
    dropout2 = relu2 #ad.dropout_op(relu2, 0.5)
    dmodel=dropout2

    # wide
    wmodel = ad.concat_op(X_wide, dmodel, 1)
    wmodel = ad.matmul_op(wmodel, W)

    prediction = wmodel
    loss = ad.softmaxcrossentropy_op(prediction, y_)
    loss = ad.reduce_mean_op(loss, [0])

    opt = optimizer.SGDOptimizer(learning_rate=lr_)
    train_op = opt.minimize(loss)

    return loss, prediction, y_, train_op
