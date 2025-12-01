import numpy as np
from scipy.sparse import coo_matrix

def build_laplacian(edges, weights, n_points):
    """Return unnormalised Laplacian (D-W) as CSR."""
    i, j = edges[:, 0], edges[:, 1]
    W = coo_matrix((weights, (i, j)), shape=(n_points, n_points))
    W = W + W.T
    d = np.asarray(W.sum(axis=1)).ravel()
    D = coo_matrix((d, (np.arange(n_points), np.arange(n_points))),
                   shape=W.shape)
    return (D - W).tocsr()
