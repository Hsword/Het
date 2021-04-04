from .graph import Graph
import libc_GNN as _C

import numpy as np
import queue
import threading

class Sampler(object):
    def __init__(self, num_sample_thread=1, transformer=None):
        self.threads = []
        self.transformer = transformer
        self._stop_sampling = False
        self.sample_queue = queue.Queue(maxsize=1)
        for i in range(num_sample_thread):
            th = threading.Thread(target=self._sample_proc)
            self.threads.append(th)

    def _sample(self):
        raise NotImplementedError

    def _sample_proc(self):
        while not self._stop_sampling:
            g = self._sample()
            if self.transformer is not None:
                g = self.transformer(g)
            while not self._stop_sampling:
                try:
                    self.sample_queue.put(g, timeout=1)
                    break
                except queue.Full as x:
                    continue

    def sample(self):
        return self.sample_queue.get()

    def __enter__(self):
        for th in self.threads:
            th.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_trace):
        self._stop_sampling = True
        for th in self.threads:
            if th is not threading.current_thread():
                th.join()

class FullBatchSampler(Sampler):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph

    def _sample(self):
        return self.graph

class RandomWalkSampler(Sampler):
    # Using multiThread to accelerate sampling
    def __init__(self, graph, num, length, num_sample_thread=1, transformer=None):
        super().__init__(num_sample_thread, transformer)
        self.graph = graph
        self._internal = _C.DistributedSampler([graph.edge_index], 0, graph.num_nodes)
        self.random_walk_num = num
        self.random_walk_length = length

    def _sample_nodes(self, num, length):
        heads = np.random.randint(low=0, high=self.graph.num_nodes, size=num)
        nodes = [heads]
        for i in range(length):
            new_heads = self._internal.sample_neighbors(heads, 0)[0]
            nodes.append(new_heads)
            heads = new_heads
        nodes = np.concatenate(nodes)
        return np.unique(nodes)

    def _sample(self):
        nodes = self._sample_nodes(self.random_walk_num, self.random_walk_length)
        edge_index = self._internal.generate_local_subgraph(nodes)
        return Graph(self.graph.x[nodes], self.graph.y[nodes], edge_index, self.graph.num_classes)

    def generate_local_subgraph(self, nodes):
        return self._internal.generate_local_subgraph(nodes)

class GraphSageSampler(Sampler):
    # Using multiThread to accelerate sampling
    def __init__(self, graph, num, depth, width=2, num_sample_thread=1, transformer=None):
        super().__init__(num_sample_thread, transformer)
        self.graph = graph
        self._internal = _C.DistributedSampler([graph.edge_index], 0, graph.num_nodes)
        self.depth = depth
        self.width = width
        self.num = num

    def _sample_nodes(self, num, width, depth):
        heads = np.random.randint(low=0, high=self.graph.num_nodes, size=num)
        all_nodes = [heads]
        last_layer_nodes = heads
        for i in range(depth):
            cur_layer_nodes = []
            for j in range(width):
                sampled = self._internal.sample_neighbors(heads, 0)[0]
                cur_layer_nodes.append(sampled)
            all_nodes.append(np.concatenate(cur_layer_nodes))
        all_nodes = np.concatenate(all_nodes)
        nodes, idx = np.unique(all_nodes, return_index=True)
        core_idx = np.where(idx < num)[0]
        return nodes, core_idx

    def _sample(self):
        nodes, core_idx = self._sample_nodes(self.num, self.width, self.depth)
        edge_index = self._internal.generate_local_subgraph(nodes)
        mask = np.zeros(len(nodes))
        mask[core_idx] = 1
        graph = Graph(self.graph.x[nodes], self.graph.y[nodes], edge_index, self.graph.num_classes)
        return graph, mask

    def generate_local_subgraph(self, nodes):
        return self._internal.generate_local_subgraph(nodes)
