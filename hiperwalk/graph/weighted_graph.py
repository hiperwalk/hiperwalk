import numpy as np
from scipy.sparse import issparse, csr_array, diags

class WeigthedGraph(Graph):
    r"""
    Constructs an arbitrary weighted graph.

    Parameters
    ----------
    adj_matrix : :class:`scipy.sparse.csr_array`, :class:`numpy.ndarray` or :class:`networkx.Graph`
        Any real Hermitian matrix or
        an instance of :class:`networkx.Graph`.
        If :class:`network.Graph`,
        the adjacency matrix of the graph is used.

    Raises
    ------
    TypeError
        If ``adj_matrix`` is not a square matrix.

    Notes
    -----
    .. todo::
        Check if it is more efficient to store the adjacency matrix as
        sparse or dense.
    
    The graph :math:`G(V,E)` on which the quantum walk 
    takes place is specified by
    any real Hermitian matrix :math:`C`.
    """

    def __init__(self, adj_matrix):
        try:
            adj_matrix.adj #throws AttributeError if not networkx graph
            import networkx as nx
            adj_matrix = nx.adjacency_matrix(adj_matrix)
        except AttributeError:
            pass

        if not issparse(adj_matrix):
            adj_matrix = csr_array(adj_matrix)

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise TypeError("Adjacency matrix is not square.")

        # the following is commented because the current way to
        # implement Laplacian in ContinuousTime quantum walks is by
        # passing the Laplacian as adjacency matrix
        # if adj_matrix.data.min() != 1 or adj_matrix.data.max() != 1:
        #     raise ValueError("Adjacency matrix must only have 0's "
        #                      + "and 1's as entries.")

        self._adj_matrix = adj_matrix

    # TODO: add functions to manage weights
