import numpy as np
from scipy.sparse import issparse, csr_array, diags

class Multigraph(Graph):
    r"""
    Constructs an arbitrary multigraph.

    Parameters
    ----------
    adj_matrix :
        The adjacency matrix of the graph
        (any integer Hermitian matrix).
        Two input types are accepted:

        * Any matrix -- for instance,
            * :class:`scipy.sparse.csr_array`,
            * :class:`numpy.ndarray`,
            * list of lists.
        * :class:`network.Graph`.
            * The adjacency matrix is extracted from the graph.

    copy : bool, default=False
        If ``True``, a hard copy of ``adj_matrix`` is stored.

    Raises
    ------
    TypeError
        If ``adj_matrix`` is not a square matrix.

    Notes
    -----
    The ``adj_matrix.data`` attribute is changed for
    more efficient manipulation.
    If the original matrix is needed,
    invoke the constructor with ``copy=True`` or
    call :meth:`adjacency_matrix()` after creating the multigraph.
    """

    def __init__(self, adj_matrix, copy=False):
        try:
            adj_matrix.adj #throws AttributeError if not networkx graph
            import networkx as nx
            adj_matrix = nx.adjacency_matrix(adj_matrix, dtype=np.int8)
        except AttributeError:
            pass

        if not issparse(adj_matrix):
            adj_matrix = csr_array(adj_matrix)

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise TypeError("Adjacency matrix is not square.")

        if copy:
            adj_matrix = adj_matrix.copy()

        # manipulate data
        data = adj_matrix.data
        for i in range(1, len(data)):
            data[i] += data[i - 1]

        self._adj_matrix = adj_matrix

        # TODO: is it useful to store the underlying simple graph?

    def number_of_edges(self):
        return self._adj_matrix.data[-1] >> 1

    def degree(self, vertex):
        vertex = self.vertex_number(vertex)

        if vertex == 0:
            return adj_matrix.data[1]

        start = self._adj_matrix.indptr[vertex]
        end = self._adj_matrix.indptr[vertex + 1]
        return adj_matrix.data[end] - adj_matrix.data[start]
    
    # TODO: add functions to manage multiedges

    def is_simple(self):
        return False
