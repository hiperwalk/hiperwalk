import numpy as np
from .graph import Graph
from scipy.sparse import issparse, csr_array, diags

class Multigraph(Graph):
    r"""
    Construct an arbitrary multigraph.

    This class facilitates the creation of a multigraph, 
    in which multiple edges between the same pair of 
    vertices are allowed. The graph's structure is 
    determined by a Hermitian adjacency matrix, 
    the entries of which are non-negative integers 
    that represent the number of multiple edges between 
    vertices. The multigraph also supports loops, 
    which are considered arcs.

    Parameters
    ----------
    adj_matrix : various types accepted
        The adjacency matrix of the graph, which must be a 
        Hermitian matrix with non-negative integer entries. 
        Acceptable input types include:

            * Direct matrix types such as:
                * :class:`scipy.sparse.csr_array`,
                * :class:`numpy.ndarray`,
                * List of lists.
            * :class:`networkx.Graph`:
                The adjacency matrix is automatically extracted
                from the specified networkx graph.

    copy : bool, default=False
        Specifies whether to store a hard copy of ``adj_matrix``:

            * If ``True``, a deep copy of the adjacency matrix is
              created and stored.
            * If ``False``, a reference to the original
              ``adj_matrix`` is stored.

    Raises
    ------
    TypeError
        If ``adj_matrix`` is not a square matrix.

    Notes
    -----
    The ``adj_matrix.data`` attribute may be modified for 
    more efficient manipulation. If the original matrix 
    data is required, it is recommended to initialize the 
    constructor with ``copy=True``. Alternatively, 
    the original matrix can be retrieved anytime by 
    calling :meth:`adjacency_matrix()` after the 
    multigraph has been created.

    When defining an instance of the Coined class on a multigraph, 
    the number of multiple edges impacts the dimension of the coin.
    
    When defining an instance of the ContinuousTime class on a multigraph, 
    the number of multiple edges is treated as an edge weight.
    """
    # def _default_dtype(self):
    #     return np.int32

    def _count_loops(self, adj_matrix):
        loops = [adj_matrix[v, v]
                 for v in range(adj_matrix.shape[0])]
        self._num_loops = np.sum(loops)

    def _set_adj_matrix(self, adj_matrix):
        # if not np.issubdtype(adj_matrix.dtype, self._default_dtype()):
        #     adj_matrix.data = adj_matrix.data.astype(
        #                                 self._default_dtype(),
        #                                 copy=False)

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

    def number_of_edges(self, u=None, v=None):
        r"""
        Return number of edges.

        Return number of edges in the multigraph if
        ``u is None`` and ``v is None``.
        Otherwise,
        return the number of edges incident to both ``u`` and ``v``.

        Parameters
        ----------
        u, v : default=None
            Vertices of the Graph.

        Returns
        -------
        Number of edges in the graph
        """
        if u is None and v is None:
            non_loops = self._adj_matrix.data[-1] - self._num_loops
            num_edges = non_loops >> 1
            return  num_edges + self._num_loops

        # number of edges incident to both u and v
        indptr = self._adj_matrix.indptr
        data = self._adj_matrix.data

        index = indptr[u]
        try:
            index += self._neighbor_index(u, v)
        except ValueError:
            return 0

        out_degree = (data[index] - data[index - 1]
                      if index > 0
                      else data[index])
        return out_degree

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
        for i in range(len(data) - 1, 0, -1):
            data[i] -= data[i - 1]

        indices = self._adj_matrix.indices
        indptr = self._adj_matrix.indptr
        adj_matrix = csr_array((data, indices, indptr), copy=True)
        return adj_matrix

    def laplacian_matrix(self):
        raise NotImplementedError()

    def is_simple(self):
        r"""
        Return False.
        """
        return False
