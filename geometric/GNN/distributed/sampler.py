from ..graph import Graph, Sampler
from ..cache import Cache
from .ps import ps_get_worker_communicator
import libc_GNN as _C

import numpy as np
import time

def _load_graph_shard(path, rank, nrank):
    import os, yaml
    with open(os.path.join(path, "meta.yml"), 'rb') as f:
        meta = yaml.load(f.read(), Loader=yaml.FullLoader)
    path = os.path.join(path, "part{}".format(rank))
    with open(os.path.join(path, "edge.npz"), 'rb') as f:
        data = np.load(f)
        edges = []
        for i in range(nrank):
            edges.append(data.get("edge_"+str(i)))
    with open(os.path.join(path, "data.npz"), 'rb') as f:
        data = np.load(f)
        x = data.get("x")
        y = data.get("y")
    return x, y, edges, meta

class DistributedSubgraphSampler(Sampler):
    def __init__(self, path, num, length, rank, nrank, num_sample_thread=1,
        cache_size_factor = 0.2, reduce_nonlocal_factor = 0.5,transformer=None, backend="ps"):
        super().__init__(num_sample_thread, transformer)
        x, y, edges, meta = _load_graph_shard(path, rank, nrank)
        self.graph = Graph(x, y, edges[rank], meta['class'])
        self._internal = _C.DistributedSampler(edges, rank, self.graph.num_nodes)
        self.random_walk_num = num
        self.random_walk_length = length
        self.rank = rank
        self.nrank = nrank
        self.reduce_nonlocal_factor = reduce_nonlocal_factor
        self.cache = Cache(int(self.graph.num_nodes * cache_size_factor), "Priority")

        assert (backend in ["ps", "grpc"])
        self.backend = backend
        if backend == "grpc":
            from .grpc import grpc_download, grpc_upload, grpc_stubs_init
            self.data_download = grpc_download
            self.data_upload = grpc_upload
        elif backend == "ps":
            from .ps import ps_download, ps_upload
            self.data_download = ps_download
            self.data_upload = ps_upload
        # when init, upload its local shard of graph into the ps
        self.data_upload(
            self.graph.x, self.graph.y,
            self._internal.indptr, self._internal.indices, self._internal.nodes_from
        )
        ps_get_worker_communicator().BarrierWorker()
        if backend == "grpc":
            # setup rpc channels
            grpc_stubs_init()

    def _sample(self):
        #initiate random_walk heads within local subgraph
        heads_id = np.random.randint(low=0, high=self.graph.num_nodes, size=self.random_walk_num)
        heads_from = np.repeat(self.rank, self.random_walk_num)
        sample_data = []
        for i in range(self.random_walk_length + 1):
            #for local nodes, find a random neighbor using DistributedSampler.sample_neighbors
            local_heads_id = heads_id[np.where(heads_from == self.rank)]
            new_nodes_id , new_nodes_from = self._internal.sample_neighbors(local_heads_id, self.reduce_nonlocal_factor)

            #find the nodes that are not in local subgraph
            foreign_heads_idx = np.where(heads_from != self.rank)
            foreign_heads_id, foreign_heads_from = heads_id[foreign_heads_idx], heads_from[foreign_heads_idx]

            #query cache to resolve some of the foreign nodes
            idx_hit, idx_miss, x_cache, y_cache, deg_cache, eid_cache, efrom_cache = \
                self.cache.lookup_nodes(foreign_heads_id, foreign_heads_from)
            x_cache = x_cache.reshape([len(idx_hit), self.graph.num_features])
            hit_id, hit_from = foreign_heads_id[idx_hit], foreign_heads_from[idx_hit]
            miss_id, miss_from = foreign_heads_id[idx_miss], foreign_heads_from[idx_miss]
            #use ps to download foreign nodes information(is there is any)
            x_ps, y_ps, deg_ps, edges_ps = self.data_download(miss_id, miss_from)
            self.cache.online_cache_insert(miss_id, miss_from, x_ps, y_ps, deg_ps, edges_ps[0], edges_ps[1])

            #concat data from cache and ps
            x = np.concatenate([x_cache, x_ps])
            y = np.concatenate([y_cache, y_ps])
            deg = np.concatenate([deg_cache, deg_ps])
            eid = np.concatenate([eid_cache, edges_ps[0]])
            efrom = np.concatenate([efrom_cache, edges_ps[1]])
            # edges = np.ascontiguousarray(edges)
            #select the next neighbor for these foreign nodes
            next_idx = _C.utils.random_index(deg)
            new_nodes_id = np.concatenate([new_nodes_id, eid[next_idx]])
            new_nodes_from = np.concatenate([new_nodes_from, efrom[next_idx]])

            #append current round data into sample_data
            reordered_id = np.concatenate([local_heads_id, hit_id, miss_id])
            reordered_from = np.concatenate([np.repeat(self.rank, len(local_heads_id)), hit_from, miss_from])
            sample_data.append((x, y, deg, np.vstack([eid, efrom]), reordered_id, reordered_from))
            #Prepare for the next round
            heads_id = new_nodes_id
            heads_from = new_nodes_from
        #after random_walk_length+1 round, use all these data to form a new sample
        sampled_graph = _C.utils.construct_graph(
            sample_data, self.graph,
            self._internal.indptr,
            self._internal.indices,
            self._internal.nodes_from,
            self.rank,
            self.nrank
        )
        # print("Cache", self.rank, self.cache.hit_rate(), self.cache.limit, self.cache.size())
        self.cache.restore_hit_rate_counter()
        return sampled_graph

class DistributedGraphSageSampler(Sampler):
    def __init__(self, path, batch_size, depth, width, rank, nrank, num_sample_thread=1,
        cache_size_factor = 0.2, reduce_nonlocal_factor = 0.5,transformer=None, backend="ps"):
        super().__init__(num_sample_thread, transformer)
        x, y, edges, meta = _load_graph_shard(path, rank, nrank)
        self.graph = Graph(x, y, edges[rank], meta['class'])
        self._internal = _C.DistributedSampler(edges, rank, self.graph.num_nodes)
        self.depth = depth
        self.width = width
        self.batch_size = batch_size
        self.rank = rank
        self.nrank = nrank
        self.reduce_nonlocal_factor = reduce_nonlocal_factor
        self.cache = Cache(int(self.graph.num_nodes * cache_size_factor), "Priority")

        assert (backend in ["ps", "grpc"])
        self.backend = backend
        if backend == "grpc":
            from .grpc import grpc_download, grpc_upload, grpc_stubs_init
            self.data_download = grpc_download
            self.data_upload = grpc_upload
        elif backend == "ps":
            from .ps import ps_download, ps_upload
            self.data_download = ps_download
            self.data_upload = ps_upload
        # when init, upload its local shard of graph into the ps
        self.data_upload(
            self.graph.x, self.graph.y,
            self._internal.indptr, self._internal.indices, self._internal.nodes_from
        )
        ps_get_worker_communicator().BarrierWorker()
        if backend == "grpc":
            # setup rpc channels
            grpc_stubs_init()

    def _sample_multiple_local(self, local_heads_id):
        nids, nfroms = [], []
        for i in range(self.width):
            nid, nfrom = self._internal.sample_neighbors(local_heads_id, self.reduce_nonlocal_factor)
            nids.append(nid)
            nfroms.append(nfrom)
        return np.concatenate(nids), np.concatenate(nfroms)

    def _sample_multiple_nonlocal(self, degree):
        idxs = []
        for i in range(self.width):
            idx = _C.utils.random_index(degree)
            idxs.append(idx)
        return np.concatenate(idxs)

    def _sample(self):
        #initiate random_walk heads within local subgraph
        heads_id = np.random.randint(low=0, high=self.graph.num_nodes, size=self.batch_size)
        heads_from = np.repeat(self.rank, self.batch_size)
        real_batch_size = len(np.unique(heads_id))
        sample_data = []
        for i in range(self.depth + 1):
            #for local nodes, find a random neighbor using DistributedSampler.sample_neighbors
            local_heads_id = heads_id[np.where(heads_from == self.rank)]
            new_nodes_id , new_nodes_from = self._sample_multiple_local(local_heads_id)

            #find the nodes that are not in local subgraph
            foreign_heads_idx = np.where(heads_from != self.rank)
            foreign_heads_id, foreign_heads_from = heads_id[foreign_heads_idx], heads_from[foreign_heads_idx]

            #query cache to resolve some of the foreign nodes
            idx_hit, idx_miss, x_cache, y_cache, deg_cache, eid_cache, efrom_cache = \
                self.cache.lookup_nodes(foreign_heads_id, foreign_heads_from)
            x_cache = x_cache.reshape([len(idx_hit), self.graph.num_features])
            hit_id, hit_from = foreign_heads_id[idx_hit], foreign_heads_from[idx_hit]
            miss_id, miss_from = foreign_heads_id[idx_miss], foreign_heads_from[idx_miss]
            #use ps to download foreign nodes information(is there is any)
            x_ps, y_ps, deg_ps, edges_ps = self.data_download(miss_id, miss_from)
            self.cache.online_cache_insert(miss_id, miss_from, x_ps, y_ps, deg_ps, edges_ps[0], edges_ps[1])

            #concat data from cache and ps
            x = np.concatenate([x_cache, x_ps])
            y = np.concatenate([y_cache, y_ps])
            deg = np.concatenate([deg_cache, deg_ps])
            eid = np.concatenate([eid_cache, edges_ps[0]])
            efrom = np.concatenate([efrom_cache, edges_ps[1]])
            # edges = np.ascontiguousarray(edges)
            #select the next neighbor for these foreign nodes
            next_idx = self._sample_multiple_nonlocal(deg)
            new_nodes_id = np.concatenate([new_nodes_id, eid[next_idx]])
            new_nodes_from = np.concatenate([new_nodes_from, efrom[next_idx]])

            #append current round data into sample_data
            reordered_id = np.concatenate([local_heads_id, hit_id, miss_id])
            reordered_from = np.concatenate([np.repeat(self.rank, len(local_heads_id)), hit_from, miss_from])
            sample_data.append((x, y, deg, np.vstack([eid, efrom]), reordered_id, reordered_from))
            #Prepare for the next round
            heads_id = new_nodes_id
            heads_from = new_nodes_from
        #after random_walk_length+1 round, use all these data to form a new sample
        sampled_graph = _C.utils.construct_graph(
            sample_data, self.graph,
            self._internal.indptr,
            self._internal.indices,
            self._internal.nodes_from,
            self.rank,
            self.nrank
        )
        mask = np.zeros(sampled_graph.num_nodes)
        mask[0:real_batch_size] = 1
        # print("Cache", self.rank, self.cache.hit_rate(), self.cache.limit, self.cache.size())
        self.cache.restore_hit_rate_counter()
        return sampled_graph, mask
