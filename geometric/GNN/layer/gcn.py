from athena import ndarray
from athena import gpu_ops as ad
from athena import initializers

class GCN(object):
    def __init__(self, in_features, out_features, activation=None, dropout=0,
                 name="GCN", custom_init=None, mp_val=None):
        if custom_init is not None:
            self.weight = ad.Variable(value=custom_init[0], name=name+"_Weight")
            self.bias = ad.Variable(value=custom_init[1], name=name+"_Bias")
        else:
            self.weight = initializers.xavier_uniform(shape=(in_features,out_features), name=name+"_Weight")
            self.bias = initializers.zeros(shape=(out_features,), name=name+"_Bias")
        #self.mp is a sparse matrix and should appear in feed_dict later
        self.mp = ad.Variable("message_passing", trainable=False, value=mp_val)
        self.activation = activation
        self.dropout = dropout

    def __call__(self, x):
        """
            Build the computation graph, return the output node
        """
        if self.dropout > 0:
            x = ad.dropout_op(x, 1 - self.dropout)
        x = ad.matmul_op(x, self.weight)
        msg = x + ad.broadcastto_op(self.bias, x)
        x = ad.CuSparse.csrmm_op(self.mp, msg)
        if self.activation == "relu":
            x = ad.relu_op(x)
        elif self.activation is not None:
            raise NotImplementedError
        return x

class GraphSage(object):
    def __init__(self, in_features, out_features, activation=None, dropout=0,
                 name="GCN", custom_init=None, mp_val=None):

        self.weight = initializers.xavier_uniform(shape=(in_features,out_features), name=name+"_Weight")
        self.bias = initializers.zeros(shape=(out_features,), name=name+"_Bias")
        self.weight2 = initializers.xavier_uniform(shape=(in_features,out_features), name=name+"_Weight")
        #self.mp is a sparse matrix and should appear in feed_dict later
        self.mp = ad.Variable("message_passing", trainable=False, value=mp_val)
        self.activation = activation
        self.dropout = dropout

    def __call__(self, x):
        """
            Build the computation graph, return the output node
        """
        feat = x
        if self.dropout > 0:
            x = ad.dropout_op(x, 1 - self.dropout)

        x = ad.CuSparse.csrmm_op(self.mp, x)
        x = ad.matmul_op(x, self.weight)
        x = x + ad.broadcastto_op(self.bias, x)
        if self.activation == "relu":
            x = ad.relu_op(x)
        elif self.activation is not None:
            raise NotImplementedError
        return ad.concat_op(x, ad.matmul_op(feat, self.weight2), axis=1)
