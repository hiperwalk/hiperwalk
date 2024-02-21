import numpy as np
from scipy.sparse import issparse, csr_array, diags

class Multigraph(Graph):
    r"""
    Constructs an arbitrary multigraph.

    Parameters
    ----------
    adj_matrix : :class:`scipy.sparse.csr_array`, :class:`numpy.ndarray` or :class:`networkx.Graph`
        Adjacency matrix, Laplacian matrix, or any real Hermitian matrix.

        If :class:`network.Graph`, the adjacency matrix of the graph is used.

    Raises
    ------
    TypeError
        If ``adj_matrix`` is not a square matrix.

    Notes
    -----
    .. todo::
        Check if it is more efficient to store the adjacency matrix as
        sparse or dense.
    """

    def __init__(self, adj_matrix):
        try:
            adj_matrix.adj #throws AttributeError if not networkx graph
            import networkx as nx
            adj_matrix = nx.adjacency_matrix(adj_matrix, dtype=np.int8)
        except AttributeError:
            pass

        if not issparse(adj_matrix):
            adj_matrix = csr_array(adj_matrix, dtype=np.int8)

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise TypeError("Adjacency matrix is not square.")

        # the following is commented because the current way to
        # implement Laplacian in ContinuousTime quantum walks is by
        # passing the Laplacian as adjacency matrix
        # if adj_matrix.data.min() != 1 or adj_matrix.data.max() != 1:
        #     raise ValueError("Adjacency matrix must only have 0's "
        #                      + "and 1's as entries.")

        self._adj_matrix = adj_matrix
        self._coloring = None

        # TODO: is it useful to store the underlying simple graph?

    def neighbors(self, vertex):
        r"""
        Return all neighbors of the given vertex.
        """
        raise NotImplementedError()

    def number_of_edges(self):
        r"""
        Determine the cardinality of the edge set.
        """
        return self._adj_matrix.sum() >> 1

    def degree(self, vertex):
        r"""
        Return the degree of the given vertex.

        The degree of a vertex :math:`u` in a graph 
        is the number of edges 
        incident to :math:`u`. Loops at :math:`u` are counted once, 
        reflecting the treatment of a loop at vertex :math:`u` as 
        a single arc :math:`(u, u)`.
        """
        raise NotImplementedError()
    
    # TODO: add functions to manage multiedges?
