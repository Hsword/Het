import libc_GNN as _C

class DistributedSampler(_C.DistributedSampler):
    def __repr__(self):
        return "<DistributedSampler nodes={} rank={}>".format(len(self.indptr)-1, self.rank)