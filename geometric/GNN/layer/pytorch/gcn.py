import torch
import torch.nn.functional as F

class torch_GCN(torch.nn.Module):
    def __init__(self, in_features, out_features, activation=None, dropout=0):
        super().__init__()
        self.l = torch.nn.Linear(in_features, out_features, bias=True)
        torch.nn.init.xavier_uniform_(self.l.weight)
        self.activation = activation
        self.dropout = dropout

    def forward(self, x, edge_norm):
        if self.dropout > 0:
            x = F.dropout(x, self.dropout)
        x = self.l(x)
        x = torch.sparse.mm(edge_norm, x)
        if self.activation == "relu":
            x = F.relu(x)
        elif self.activation is not None:
            raise NotImplementedError
        return x

class torch_Sage(torch.nn.Module):
    def __init__(self, in_features, out_features, activation=None, dropout=0):
        super().__init__()
        self.l = torch.nn.Linear(in_features, out_features, bias=True)
        torch.nn.init.xavier_uniform_(self.l.weight)
        self.l2 = torch.nn.Linear(in_features, out_features, bias=True)
        torch.nn.init.xavier_uniform_(self.l2.weight)
        self.activation = activation
        self.dropout = dropout

    def forward(self, x, edge_norm):
        feat = x
        if self.dropout > 0:
            x = F.dropout(x, self.dropout)
        x = torch.sparse.mm(edge_norm, x)
        x = self.l(x)
        x = torch.cat([x, self.l2(feat)], dim=1),
        if self.activation == "relu":
            x = F.relu(x[0])
        elif self.activation is not None:
            raise NotImplementedError
        return x