import numpy as np
from scipy import sparse

def convert_to_csr(edge_index :np.ndarray, num_nodes :int, directed=False):
    """convert the edge indexes into csr format so that it can be used by metis"""
    num_edges = edge_index[0].shape[0]
    if directed:
        mat = sparse.csr_matrix(
            (
                np.ones(2*num_edges),
                (np.concatenate([edge_index[0],edge_index[1]]), np.concatenate([edge_index[1],edge_index[0]])) # i, j
            ),
            shape=(num_nodes,num_nodes)
        )
        print(mat.nnz)
    else:
        mat = sparse.csr_matrix(
            (
                np.ones(num_edges),
                (edge_index[0], edge_index[1]) # i, j
            ),
            shape=(num_nodes,num_nodes)
        )
    return mat.indptr, mat.indices
