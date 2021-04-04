from tqdm import tqdm
import os
import math
import logging
from hparams import Hparams
from ath_transformer import Transformer
from data_load import DataLoader
from athena import gpu_ops as ad
from athena import optimizer
from athena import ndarray
import numpy as np
# import time

logging.basicConfig(level=logging.INFO)


logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
print(hp)

logging.info("# Prepare train/eval batches")
dataloader = DataLoader(hp.train1, hp.train2, hp.maxlen1, hp.maxlen2, hp.vocab)

ctx = ndarray.gpu(1)
xs = ad.Variable(name='xs')
ys1 = ad.Variable(name='ys1')
ys2 = ad.Variable(name='ys2')
nonpadding = ad.Variable(name='nonpadding')

logging.info("# Load model")
m = Transformer(hp)
loss = m.train(xs, (ys1, ys2))
loss = ad.div_op(ad.reduce_sum_op(loss * nonpadding, axes=[0, 1]), ad.reduce_sum_op(nonpadding, axes=[0, 1]) + 1e-7)
opt = optimizer.SGDOptimizer(hp.lr)
train_op = opt.minimize(loss)
executor = ad.Executor([loss, train_op], ctx=ctx)

logging.info("# Session")


for ep in range(hp.num_epochs):
    dataloader.make_epoch_data(hp.batch_size)
    for i in tqdm(range(dataloader.batch_num)):
        xs_val, ys_val = dataloader.get_batch()
        # st = time.time()
        xs_val = xs_val[0]
        ys1_val = ys_val[0][:,:-1]
        ys2_val = ys_val[0][:,1:]
        nonpadding_val = np.not_equal(ys2_val, dataloader.get_pad()).astype(np.float32)
        _loss, _ = executor.run(feed_dict={xs: xs_val, ys1: ys1_val, ys2: ys2_val, nonpadding: nonpadding_val})
        # en = time.time()
        # if i == 100:
        #     exit()
        
        log_str = 'Iteration %d, loss %f' % (i, _loss.asnumpy())
        print(log_str)
        # print('time: ', (en - st))

logging.info("Done")
