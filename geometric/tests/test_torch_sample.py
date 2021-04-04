import numpy as np
from GNN.dataset import load_dataset
from GNN.layer import *
from GNN.graph import *
from GNN.layer.pytorch import torch_Sage

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
        self.conv1 = torch_Sage(graph.num_features, hidden_layer_size, activation="relu", dropout=0.1)
        self.conv2 = torch_Sage(2*hidden_layer_size, hidden_layer_size, activation="relu", dropout=0.1)
        self.classifier = torch.nn.Linear(2*hidden_layer_size, graph.num_classes)
        torch.nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = F.normalize(x, p=2, dim=1)
        x = self.classifier(x)
        return x

def train(num_epoch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()

    def eval():
        model.eval()
        with torch.no_grad():
            x = torch.Tensor(graph_full.x).to(device)
            y = torch.Tensor(graph_full.y).to(device, torch.long)
            edge_norm = mp_matrix(graph_full, 0, system="Pytorch")
            out = model(x, edge_norm)
            loss = F.cross_entropy(out, y)
            y_pred = out.argmax(axis=1)[train_split:].cpu()
            y_true = y[train_split:].cpu()
            micro_f1 = metrics.f1_score(y_true, y_pred, labels=range(graph.num_classes), average="micro")
            macro_f1 = metrics.f1_score(y_true, y_pred, labels=range(graph.num_classes), average="macro")
            print(i, "loss={} mic={} mac={}".format(loss, micro_f1, macro_f1))
        model.train()

    with RandomWalkSampler(graph, 4000, 2) as sampler:
        for i in range(num_epoch):
            start = time.time()
            g_sample = sampler.sample()
            print("Sample", time.time() - start)
            x = torch.Tensor(g_sample.x).to(device)
            y = torch.Tensor(g_sample.y).to(device, torch.long)
            print("Copy", time.time() - start)
            edge_norm = mp_matrix(g_sample, 0, system="Pytorch")
            print("norm", time.time() - start)
            out = model(x, edge_norm)
            loss = F.cross_entropy(out, y)
            acc = int((out.argmax(axis=1) == y).sum()) / y.shape[0]
            print("epoch={} loss={:.5f} acc={:.5f}".format(i, float(loss), acc))
            loss.backward()
            #if sampler.cur_num == 0:
            optimizer.step()
            optimizer.zero_grad()
            if (i+1) % 20 == 0:
                eval()
            print(time.time() - start)

if __name__ == "__main__":
    train(1200)