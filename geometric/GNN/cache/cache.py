import libc_GNN as _C

import threading
import queue

class Cache(_C.Cache):
    def __init__(self, limit, policy):
        # We don't allow cache size=0, that will incur an error
        if limit == 0:
            limit = 1
        assert(limit > 0)
        super().__init__(limit, policy)
        self.policy = policy
        self.restore_hit_rate_counter()

    def hit_rate(self):
        if self.hit_count == 0:
            return 0
        return self.hit_count / (self.miss_count + self.hit_count)

    def restore_hit_rate_counter(self):
        self.hit_count = 0
        self.miss_count = 0

    def __repr__(self):
        return "<Cache {}/{} {}>".format(
            self.size(),
            self.limit,
            self.policy
        )

    def lookup_nodes(self, nodes_id, nodes_from):
        idx_hit, idx_miss, x, y, deg, edge_id, edge_from \
            = self.lookup_packed(nodes_id, nodes_from)
        self.hit_count += len(idx_hit)
        self.miss_count += len(idx_miss)
        return idx_hit, idx_miss, x, y, deg, edge_id, edge_from

    def online_cache_insert(self, nodes_id, nodes_from, x, y, deg, eid, efrom):
        num_nodes = len(nodes_id)
        offset = 0
        for i in range(num_nodes):
            self.insert(nodes_id[i], nodes_from[i], x[i], y[i],
                eid[offset:offset+deg[i]], efrom[offset:offset+deg[i]], deg[i])
            offset = offset + deg[i]
