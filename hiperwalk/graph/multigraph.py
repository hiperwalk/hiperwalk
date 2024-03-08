import numpy as np
from .graph import Graph
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
            # TODO: verify which representation occupies less space
            adj_matrix = csr_array(adj_matrix)

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise TypeError("Adjacency matrix is not square.")

        if copy:
            adj_matrix = adj_matrix.copy(dtype=np.int32)

        loops = [adj_matrix[v, v]
                 for v in range(adj_matrix.shape[0])]
        self._num_loops = np.sum(loops)
        del loops

        # manipulate data
        if not np.issubdtype(adj_matrix.dtype, np.integer):
            adj_matrix = adj_matrix.astype(np.int32)
        data = adj_matrix.data.astype(np.int32, copy=False)
        for i in range(1, len(data)):
            data[i] += data[i - 1]

        self._adj_matrix = adj_matrix

        # TODO: is it useful to store the underlying simple graph?

    def _entry(self, lin, col):
        return self._adj_matrix[lin, col]

    def _find_entry(self, entry):
        adj_matrix = self._adj_matrix
        index = _interval_binary_search(adj_matrix.data, entry) + 1

        col = adj_matrix.indices[index]
        lin = _interval_binary_search(adj_matrix.indptr, index)

        return (lin, col)

    def number_of_edges(self):
        non_loops = self._adj_matrix.data[-1] - self._num_loops
        num_edges = non_loops >> 1
        return  num_edges + self._num_loops

    def degree(self, vertex):
        vertex = self.vertex_number(vertex)
        adj_matrix = self._adj_matrix

        start = adj_matrix.indptr[vertex] - 1
        end = adj_matrix.indptr[vertex + 1] - 1

        if start < 0:
            return adj_matrix.data[end]
        return adj_matrix.data[end] - adj_matrix.data[start]
    
    # TODO: add functions to manage multiedges

    def adjacency_matrix(self):
        data = np.copy(self._adj_matrix.data)
        for i in range(1, len(data)):
            data[i] -= data[i - 1]

        indices = self._adj_matrix.indices
        indptr = self._adj_matrix.indptr
        adj_matrix = csr_array((data, indices, indptr))
        return adj_matrix

    def laplacian_matrix(self):
        raise NotImplementedError()

    def is_simple(self):
        return False

    def number_of_multiedges(self, u, v):
        r"""
        Number of multiedges that connecet vertices u and v.
        """
        indptr = self._adj_matrix.indptr
        data = self._adj_matrix.data

        index = indptr[u]
        index += self._neighbor_index(u, v)

        out_degree = (data[index] - data[index - 1]
                      if index > 0
                      else data[index])
        return out_degree
