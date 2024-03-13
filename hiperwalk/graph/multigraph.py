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

    def __default_dtype(self):
        return np.int32

    def __count_loops(self, adj_matrix):
        loops = [adj_matrix[v, v]
                 for v in range(adj_matrix.shape[0])]
        self._num_loops = np.sum(loops)

    def _set_adj_matrix(self, adj_matrix):
        if not np.issubdtype(adj_matrix.dtype, self.__default_dtype()):
            adj_matrix = adj_matrix.astype(self.__default_dtype())

        data = adj_matrix.data
        for i in range(1, len(data)):
            data[i] += data[i - 1]

        self._adj_matrix = adj_matrix

    def __init__(self, adj_matrix, copy=False):
        super().__init__(adj_matrix, copy)
        # TODO: is it useful to store the underlying simple graph?

    def _entry(self, row, col):
        return self._adj_matrix[row, col]

    def _find_entry(self, entry):
        adj_matrix = self._adj_matrix
        index = _interval_binary_search(adj_matrix.data, entry) + 1

        col = adj_matrix.indices[index]
        row = _interval_binary_search(adj_matrix.indptr, index)

        return (row, col)

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
