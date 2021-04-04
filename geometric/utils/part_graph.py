import argparse
import sys, os
import numpy as np
import time
import yaml

from GNN.dataset import load_dataset, load_sparse_dataset

def part_graph(dataset, nparts, output_path, sparse):
    if os.path.exists(output_path):
        os.rmdir(output_path)
    os.mkdir(output_path)
    start = time.time()
    if sparse:
        graph, idx_max = load_sparse_dataset(dataset)
    else:
        graph = load_dataset(dataset)
    print("step1: load_dataset complete, time cost {:.3f}s".format(time.time()-start))
    start = time.time()
    subgraphs, edge_index, edges = graph.part_graph(nparts)
    print("step2: partition graph complete, time cost {:.3f}s".format(time.time()-start))
    start = time.time()
    for i in range(nparts):
        part_dir = os.path.join(output_path, "part{}".format(i))
        os.mkdir(part_dir)
        edge_path = os.path.join(part_dir, "edge.npz")
        data_path = os.path.join(part_dir, "data.npz")
        all_edges = {}
        for j in range(nparts):
            index = edge_index[i][j]
            all_edges["edge_"+str(j)] = (edges[0][index], edges[1][index])

        with open(edge_path, 'wb') as f:
            np.savez(file=f, **all_edges)
        with open(data_path, 'wb') as f:
            np.savez(file=f, x=subgraphs[i].x, y=subgraphs[i].y)
    print("step3: save partitioned graph, time cost {:.3f}s".format(time.time()-start))
    parititon = {
        "nodes" : [g.num_nodes for g in subgraphs],
        "edges" : [g.num_edges for g in subgraphs],
    }
    meta = {
        "name": dataset,
        "node": graph.num_nodes,
        "edge": graph.num_edges,
        "feature": graph.num_features,
        "class": graph.num_classes,
        "num_part": nparts,
        "partition": parititon,
    }
    if sparse:
        meta["idx_max"] = idx_max
    edge_path = os.path.join(output_path, "meta.yml")
    with open(edge_path, 'w') as f:
        yaml.dump(meta, f, sort_keys=False)

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", required=True)
    parser.add_argument("--nparts", "-n", required=True)
    parser.add_argument("--path", "-p", required=True)
    parser.add_argument("--sparse", action="store_true")
    args = parser.parse_args()
    output_path = str(args.path)
    nparts = int(args.nparts)
    dataset = str(args.dataset)
    output_path = os.path.join(output_path, dataset)
    part_graph(dataset, nparts, output_path, args.sparse)
