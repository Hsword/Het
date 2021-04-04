import numpy as np
from GNN.dataset import load_dataset
from GNN.layer import GCN
from GNN.graph import *

from athena import ndarray
from athena import gpu_ops as ad
from athena import optimizer
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import time

# testing correctness with PyG using same initialization for weight and bias
use_same_init = True
if use_same_init:
    init_w1, init_w2, init_b1, init_b2 = 0, 0, 0, 0

def convert_to_one_hot(vals, max_val = 0):
    """Helper method to convert label array to one-hot array."""
    if max_val == 0:
      max_val = vals.max() + 1
    one_hot_vals = np.zeros((vals.size, max_val))
    one_hot_vals[np.arange(vals.size), vals] = 1
    return one_hot_vals

graph_full = load_dataset("Cora")
train_split = int(0.8 * graph_full.num_nodes)
graph_full.add_self_loop()
graph = split_training_set(graph_full, train_split)
num_features = graph_full.num_features # =1433
num_classes = graph_full.num_classes # =7

hidden_layer_size = 16

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def train_torch(num_epoch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    if use_same_init:
        global init_w1, init_w2, init_b1, init_b2
        init_w1 = model.conv1.weight.detach().cpu().numpy()
        init_w2 = model.conv2.weight.detach().cpu().numpy()
        init_b1 = model.conv1.bias.detach().cpu().numpy()
        init_b2 = model.conv2.bias.detach().cpu().numpy()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    x = torch.Tensor(graph.x).to(device)
    y = torch.Tensor(graph.y).to(device, torch.long)
    edge_index = torch.Tensor(graph.edge_index).to(device, torch.long)
    start_time = time.time()
    losses = []
    for epoch in range(num_epoch):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out, y)
        #print("Train loss :",float(loss))
        losses.append(float(loss))
        loss.backward()
        optimizer.step()
        if epoch==0:
            start_time= time.time()
        print(epoch, "torch time:",time.time()-start_time)
    print("torch time:",time.time()-start_time)
    return losses

def train_athena(num_epoch):
    ctx = ndarray.gpu(0)

    x_ = ad.Variable(name="x_")
    y_ = ad.Variable(name="y_")

    if use_same_init:
        gcn1 = GCN(num_features, hidden_layer_size, custom_init=(init_w1, init_b1))
        gcn2 = GCN(hidden_layer_size, num_classes, custom_init=(init_w2, init_b2))
    else:
        gcn1 = GCN(num_features, hidden_layer_size)
        gcn2 = GCN(hidden_layer_size, num_classes)

    mp_val = mp_matrix(graph, ctx, use_original_gcn_norm=True)
    feed_dict = {
        gcn1.mp : mp_val,
        gcn2.mp : mp_val,
        x_ : ndarray.array(graph.x, ctx=ctx),
        y_ : ndarray.array(convert_to_one_hot(graph.y, max_val=num_classes), ctx=ctx)
    }

    x = gcn1(x_)
    x = ad.relu_op(x)
    y = gcn2(x)

    loss = ad.softmaxcrossentropy_op(y, y_)

    opt = optimizer.AdamOptimizer(0.01)
    train_op = opt.minimize(loss)
    executor = ad.Executor([loss, y, train_op], ctx=ctx)
    start_time = time.time()
    losses = []
    for i in range(num_epoch):
        loss_val, y_predicted, _ = executor.run(feed_dict = feed_dict)

        y_predicted = y_predicted.asnumpy().argmax(axis=1)
        acc = (y_predicted == graph.y).sum()
        losses.append(loss_val.asnumpy().mean())
        if i==0:
            start_time= time.time()
        print("Train loss :", loss_val.asnumpy().mean())
        print("Train accuracy:", acc/len(y_predicted))
        print("Athena time:",i, time.time()-start_time)
    print("Athena time:", time.time()-start_time)

    mp_val = mp_matrix(graph_full, ctx)

    feed_dict = {
        gcn1.mp : mp_val,
        gcn2.mp : mp_val,
        x_ : ndarray.array(graph_full.x, ctx=ctx),
    }
    executor_eval = ad.Executor([y], ctx=ctx)
    y_predicted, = executor_eval.run(feed_dict=feed_dict)
    y_predicted = y_predicted.asnumpy().argmax(axis=1)
    acc = (y_predicted == graph_full.y)[train_split:].sum()
    print("Test accuracy:", acc/len(y_predicted[train_split:]))
    return losses

if __name__ == "__main__":
    loss1 = train_torch(100)

    loss2 = train_athena(100)

    for i,j in zip(loss1, loss2):
        print("PyG Loss: ", i , "Athena Loss:", j, "Diff:", abs(i-j))