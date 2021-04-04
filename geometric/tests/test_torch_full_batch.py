import numpy as np
from GNN.dataset import load_dataset
from GNN.layer import *
from GNN.graph import *
from GNN.layer.pytorch import torch_GCN

from sklearn import metrics

import torch
import torch.nn.functional as F

import time

graph_full = load_dataset("Reddit")
#graph_full = shuffle(graph_full)
train_split = int(0.8 * graph_full.num_nodes)
graph_full.add_self_loop()
graph = split_training_set(graph_full, train_split)
hidden_layer_size = 128

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch_GCN(graph.num_features, hidden_layer_size, activation="relu")
        self.conv2 = torch_GCN(hidden_layer_size, graph.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x

def train(num_epoch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()

    x = torch.Tensor(graph.x).to(device)
    y = torch.Tensor(graph.y).to(device, torch.long)
    edge_norm = mp_matrix(graph, 0, system="Pytorch")
    for i in range(num_epoch):
        start = time.time()
        out = model(x, edge_norm)
        loss = F.cross_entropy(out, y)
        acc = int((out.argmax(axis=1) == y).sum()) / y.shape[0]
        loss.backward()
        #if sampler.cur_num == 0:
        optimizer.step()
        optimizer.zero_grad()
        print(i, time.time() - start)

if __name__ == "__main__":
    train(100)