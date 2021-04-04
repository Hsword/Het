import numpy as np
from GNN.dataset import load_dataset
from GNN.layer import GCN
from GNN.graph import *

import torch
import dgl
from dgl.nn import GraphConv
import torch.nn.functional as F

import time

graph = load_dataset("Reddit")
graph.add_self_loop()
num_features = graph.num_features # =1433
num_classes = graph.num_classes # =7

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GraphConv(num_features, 128, norm='both', weight=True, bias=True)
        self.conv2 = GraphConv(128, num_classes, norm='both', weight=True, bias=True)

    def forward(self, x, g):
        x = self.conv1(g, x)
        x = F.relu(x)
        x = self.conv2(g, x)

        return x

def train_dgl(num_epoch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = []
    with RandomWalkSampler(graph, 4000, 2) as sampler:
        for epoch in range(num_epoch):
            subgraph = sampler.sample()
            x = torch.Tensor(subgraph.x).to(device)
            y = torch.Tensor(subgraph.y).to(device, torch.long)
            g = dgl.graph((subgraph.edge_index[0], subgraph.edge_index[1])).to(0)
            optimizer.zero_grad()
            out = model(x, g)
            loss = F.cross_entropy(out, y)
            losses.append(float(loss))
            loss.backward()
            optimizer.step()
            if epoch==0:
                start_time= time.time()
            print("DGL time:",epoch, time.time()-start_time)
    print("DGL time:",time.time()-start_time)
    return losses

if __name__ == "__main__":
    loss1 = train_dgl(100)