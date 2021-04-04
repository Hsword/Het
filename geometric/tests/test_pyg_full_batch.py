import numpy as np
from GNN.dataset import load_dataset
from GNN.graph import *

import torch
import torch_geometric as geo
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import time


graph_full = load_dataset("Cora")
train_split = int(0.8 * graph_full.num_nodes)
graph_full.add_self_loop()
graph = split_training_set(graph_full, train_split)
num_features = graph_full.num_features # =1433
num_classes = graph_full.num_classes # =7

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

def train_pyg(num_epoch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    x = torch.Tensor(graph.x).to(device)
    y = torch.Tensor(graph.y).to(device, torch.long)
    edge_index = torch.Tensor(graph.edge_index).to(device, torch.long)
    start_time = time.time()
    for epoch in range(num_epoch):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out, y)
        #print("Train loss :",float(loss))
        loss.backward()
        optimizer.step()
        if epoch==0:
            start_time= time.time()
        print(epoch, "PyG time:",time.time()-start_time)

if __name__ == "__main__":
    loss1 = train_pyg(200)