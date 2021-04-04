import os
from ..graph import Graph
import torch.optim
import numpy as np
dataset_list = (
    "Cora",
    "ogbn-products",
    "ogbn-papers100M",
    "Amazon",
    "Reddit",
    "PubMed",
    "AmazonSparse"
)
dataset_root = os.path.dirname(__file__)+"/.dataset/"

#building the dataset using numpy
def load_dataset(name):
    root = dataset_root + name
    if name=="Cora":
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(root=root, name='Cora')
        data = dataset[0]
        g = Graph(
            x=data.x.numpy(),
            y=data.y.numpy(),
            edge_index=data.edge_index.numpy(),
            num_classes=dataset.num_classes
        )
    elif name=="PubMed":
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(root=root, name='PubMed')
        data = dataset[0]
        g = Graph(
            x=data.x.numpy(),
            y=data.y.numpy(),
            edge_index=data.edge_index.numpy(),
            num_classes=dataset.num_classes
        )
    elif name=="Amazon":
        from torch_geometric.datasets import Amazon
        dataset = Amazon(root=root, name="Computers")
        data = dataset[0]
        g = Graph(
            x=data.x.numpy(),
            y=data.y.numpy(),
            edge_index=data.edge_index.numpy(),
            num_classes=dataset.num_classes
        )
    elif name=="Reddit":
        from torch_geometric.datasets import Reddit
        dataset = Reddit(root=root)
        data = dataset[0]
        g = Graph(
            x=data.x.numpy(),
            y=data.y.numpy(),
            edge_index=data.edge_index.numpy(),
            num_classes=dataset.num_classes
        )
    elif name=="ogbn-products" or name=="ogbn-papers100M":
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name=name, root=root)
        data = dataset[0]
        g = Graph(
            x=data.x.numpy(),
            y=data.y.numpy().squeeze(),
            edge_index=data.edge_index.numpy(),
            num_classes=dataset.num_classes
        )
    else:
        raise NotImplementedError

    return g

def load_sparse_dataset(name):
    root = dataset_root + name
    if name=="Reddit":
        from torch_geometric.datasets import Reddit
        dataset = Reddit(root=root)
        data = dataset[0]
        node_id = np.arange(data.num_nodes).reshape(-1, 1)
        g = Graph(
            x=np.concatenate([node_id, data.train_mask.reshape(-1, 1)], axis=1),
            y=data.y.numpy(),
            edge_index=data.edge_index.numpy(),
            num_classes=dataset.num_classes
        )
        idx_max = data.num_nodes
    elif name=="AmazonSparseNode":
        data = np.load(os.path.join(dataset_root + "AmazonSparse", "graph.npz"))
        feat = np.load(os.path.join(dataset_root + "AmazonSparse", "sparsefeature.npy"))
        edge = data['edge'].T
        directed = np.concatenate([edge, edge[[1,0]]], axis=1)
        idx_max = np.max(feat) + 1
        node_id = np.arange(feat.shape[0]).reshape(-1, 1) + idx_max
        idx_max += feat.shape[0]
        g = Graph(
            x = np.concatenate([feat, node_id ,data["train_map"].reshape(-1, 1)], axis=-1),
            y = data['y'],
            edge_index = directed,
            num_classes = np.max(data['y']) + 1
        )
    elif name=="AmazonSparse":
        data = np.load(os.path.join(root, "graph.npz"))
        feat = np.load(os.path.join(root, "sparsefeature.npy"))
        edge = data['edge'].T
        directed = np.concatenate([edge, edge[[1,0]]], axis=1)
        idx_max = np.max(feat) + 1
        print(idx_max)
        g = Graph(
            x = np.concatenate([feat ,data["train_map"].reshape(-1, 1)], axis=-1),
            y = data['y'],
            edge_index = directed,
            num_classes = np.max(data['y']) + 1
        )
    elif name=="ogbn-mag":
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name=name, root=root)
        g, idx_max = process_ogbn_mag_dataset(dataset)
    else:
        raise NotImplementedError

    return g, int(idx_max)

def add_nodeid(graph):
    idx = np.arange(graph.num_nodes).reshape(-1, 1)
    x = np.concatenate([graph.x, idx], axis=-1)
    return Graph(x=x, y=graph.y, edge_index=graph.edge_index, num_classes=graph.num_classes)

def add_train_eval_map(graph, ratio):
    role = np.zeros(graph.num_nodes)
    train_node = int(ratio * graph.num_nodes)
    role[0:train_node] = 1.0
    np.random.shuffle(role)
    role = role.reshape(-1, 1)
    x = np.concatenate([graph.x, role], axis=-1)
    return Graph(x=x, y=graph.y, edge_index=graph.edge_index, num_classes=graph.num_classes)

def process_ogbn_mag_dataset(dataset):
    data = dataset[0]
    year = data.node_year['paper'].numpy()
    train_mask = year < 2018
    edge = data.edge_index_dict['paper', 'cites', 'paper'].numpy()
    directed = np.concatenate([edge, edge[[1,0]]], axis=1)
    num_nodes = data.num_nodes_dict['paper']
    def process_sparse_idx(rel, length, base):
        sp_idx = [[] for i in range(num_nodes)]
        for i, j in rel.T:
            sp_idx[i].append(j)
        for i in range(num_nodes):
            if len(sp_idx[i]) > length:
                sp_idx[i] = sp_idx[i][0:length]
            while len(sp_idx[i]) < length:
                sp_idx[i].append(-1)
        sp_idx = np.array(sp_idx)
        sp_idx += (base + 1)
        return sp_idx

    node_id = np.arange(num_nodes).reshape(-1, 1)
    field = data.edge_index_dict[('paper', 'has_topic', 'field_of_study')].numpy()
    paper_field = process_sparse_idx(field, 10, num_nodes)
    idx_max = num_nodes + data.num_nodes_dict['field_of_study'] + 1
    author = data.edge_index_dict[('author', 'writes', 'paper')].numpy()
    paper_author = process_sparse_idx(author[[1, 0]], 10, idx_max)
    idx_max += data.num_nodes_dict['author'] + 1
    g = Graph(
        x=np.concatenate([paper_field, paper_author ,node_id, train_mask], axis=1),
        y=data.y_dict["paper"].numpy().squeeze(),
        edge_index=directed,
        num_classes=dataset.num_classes
    )
    return g, idx_max
