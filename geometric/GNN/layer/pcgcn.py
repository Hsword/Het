from athena import ndarray
from athena import gpu_ops as ad
from athena import initializers

class PCGCN(object):
    def __init__(self, in_features: int, out_features: int, npart: int,
                    name="GCN", custom_init=None):
        if custom_init is not None:
            self.weight = ad.Variable(value=custom_init[0], name=name+"_Weight")
            self.bias = ad.Variable(value=custom_init[1], name=name+"_Bias")
        else:
            self.weight = initializers.xavier_uniform(shape=(in_features,out_features), name=name+"_Weight")
            self.bias = initializers.zeros(shape=(out_features,), name=name+"_Bias")
        self.in_features = in_features
        self.out_features = out_features
        self.npart = npart
        #npart * npart message_passing matrix, either dense or sparse
        self.mp = [[ad.Variable("message_passing", trainable=False) for j in range(npart)] for i in range(npart)]

    def __call__(self, input, subgraph_size: list, use_sparse: list):
        """
            Build the computation graph, return the output node
            split , in-graph message-passing, inter-graph message-passing , concat
        """
        x = ad.matmul_op(input, self.weight)
        msg = x + ad.broadcastto_op(self.bias, x)
        output_nodes = []
        msgs = []
        split_at = 0
        # message passing for each subgraph
        for i in range(self.npart):
            sliced_msg = ad.slice_op(
                node=msg,
                begin=(split_at, 0),
                size=(subgraph_size[i], self.out_features)
            )
            split_at += subgraph_size[i]
            msgs.append(sliced_msg)
            if use_sparse[i]:
                output = ad.csrmm_op(self.mp[i][i], sliced_msg)
            else:
                output = ad.matmul_op(self.mp[i][i], sliced_msg)
            output_nodes.append(output)
        # message passing between subgraphs
        for i in range(self.npart):
            for j in range(self.npart):
                if i==j:
                    continue
                output_nodes[j] = output_nodes[j] + ad.csrmm_op(self.mp[i][j], msgs[i])
        # concat all the remaining nodes
        result = output_nodes[0]
        for i in range(1, self.npart):
            result = ad.concat_op(result, output_nodes[i])
        return result