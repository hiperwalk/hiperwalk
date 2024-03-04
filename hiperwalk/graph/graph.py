import numpy as np
from scipy.sparse import issparse, csr_array, diags
from .._constants import __DEBUG__

def _binary_search(v, elem, start=0, end=None):
    r"""
    This function expects a sorted array and performs a binary search on the subarray 
    v[start:end], looking for the element 'elem'. 
    If the element is found, the function returns the index of the element. 
    If the element is not found, the function returns -1.    
    This is an implementation of binary search following Cormen's algorithm. 
    It is used to improve the time complexity of the search process.
    """

    if end == None:
        end = len(v)
    
    while start < end:
        mid = int((start + end)/2)
        if elem <= v[mid]:
            end = mid
        else:
            start = mid + 1

    return end if v[end] == elem else -1

class Graph():
    r"""
    Constructs an arbitrary graph.

    This class defines the graph structure used for implementing a 
    quantum walk. It encapsulates all necessary properties 
    and functionalities of the graph 
    required for the quantum walk dynamics.

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
    The class methods facilitate the construction of a valid quantum walk 
    and can be provided as parameters to plotting functions.
    For visualizations, the default graph representation will be used.
    Specific classes are available for well-known graph types,
    such as hypercubes and lattices.

    The adjacency matrix is always stored as a
    :class:`scipy.sparse.csr_array`.
    If ``adj_matrix`` is sparse and ``copy=False``,
    the argument will be changed for more efficient manipulation.

    .. warning::

        To reduce memory usage, ``adj_matrix.data`` is set to ``None``.
        This is possible because ``adj_matrix.data`` should be an
        array of ones.

        If the user wishes to keep the original ``adj_matrix``,
        the argument ``copy`` must be set to ``True``.

    .. todo::
        check if it is more efficient to store adj_matrix as numpy array.
        Add numpy array manipulation

    The treatment of the graph depends on the quantum walk model. 
    .. todo::
        Reference new part of documentation.
    """

    def __init__(self, adj_matrix, copy=False):
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

        if copy:
            adj_matrix = adj_matrix.copy()

        del adj_matrix.data
        adj_matrix.data = None
        self._adj_matrix = adj_matrix

    def adjacent(self, u, v):
        r"""
        Return True if vertex ``u`` is adjacent to ``v``.
        """
        # TODO: check implementation of adj_matrix[u, v]
        # if adj_matrix.has_sorted_index, probably scipy is more efficient.
        # if indices are not sorted, scipy probably does a linear search,
        # and a graph-dependent implementation may be more efficient.
        u = self.vertex_number(u)
        v = self.vertex_number(v)
        A = self._adj_matrix
        return v in A.indices[A[u]:A[u+1]]

    def _neighbor_index(self, vertex, neigh):
        r"""
        Return the index of `neigh` in the neihborhood list of `vertex`.

        The returned index satisfies
        ``adj_matrix.indices[adj_matrix.indptr[vertex] + index] == neigh``.

        This is useful for graphs where the adjacency is not listed in
        ascending order.
        It is recommended to override this method for specific graphs.
        """
        # TODO test this function
        # TODO write unitary tests
        if __DEBUG__:
            assert self.adjacent(vertex, neigh)

        adj_matrix = self._adj_matrix
        start = adj_matrix.indptr[vertex]
        end = adj_matrix.indptr[vertex + 1]

        # if indices is in ascending order
        if adj_matrix.has_sorted_indices:
            index = _binary_search(adj_matrix.indices,
                                   neigh,
                                   start=start,
                                   end=end)
            return index - start

        # indices is not in ascending order
        for index in range(start, end):
            if adj_matrix.indices[index] == neigh:
                return index - start

        raise ValueError("Vertices " + str(vertex) + " and "
                         + str(neigh) + " are not adjacent.")

    def neighbors(self, vertex):
        r"""
        Return all neighbors of the given vertex.
        """
        vertex = self.vertex_number(vertex)
        start = self._adj_matrix.indptr[vertex]
        end = self._adj_matrix.indptr[vertex + 1]
        return self._adj_matrix.indices[start:end]

    def number_of_vertices(self):
        r"""
        Determine the cardinality of the vertex set.
        """
        return self._adj_matrix.shape[0]

    def number_of_edges(self):
        r"""
        Determine the cardinality of the edge set.
        """
        return len(self._adj_matrix.indices) >> 1

    def degree(self, vertex):
        r"""
        Return the degree of the given vertex.

        The degree of a vertex :math:`u` in a graph 
        is the number of edges  incident to :math:`u`.
        Loops at :math:`u` are counted once.

        Parameters
        ----------
        vertex :
            Any valid vertex representation.

        Notes
        -----
        .. todo::
            Will we accept loops in simple graphs?
        """
        vertex = self.vertex_number(vertex)
        indptr = self._adj_matrix.indptr
        return indptr[vertex + 1] - indptr[vertex]

    def vertex_number(self, vertex):
        r"""
        Return the vertex number given any vertex representation.

        This method returns the numerical label of the vertex 
        regardless of its representation.
        There are some graphs in which a vertex may have multiple
        representations.
        For example, coordinates in a grid.
        For arbitrary graphs,
        this function returns the argument itself if valid.

        Parameters
        ----------
        vertex: int
            The vertex in any of its representation.
            For general graphs,
            only its label is accepted.

        Returns
        -------
        int
            Vertex number.

        Raises
        ------
        ValueError
            If ``vertex`` is not valid.

        Notes
        -----
        It is useful to have this function implemented for general graphs
        to simplify the implementation of some quantum walk methods.
        """
        vertex = int(vertex)
        num_vert = self.number_of_vertices()
        if vertex < 0 or vertex >= num_vert:
            raise ValueError("Vertex label out of range. " +
                             "Expected integer value from 0 to " +
                             str(num_vert - 1))
        return vertex

    def adjacency_matrix(self):
        r"""
        Return the graph's adjacency matrix.

        Returns
        -------
        :class:`scipy.sparse.csr_array`.

        Notes
        -----
    
        In a weightless graph :math:`G(V, E)` with :math:`n` vertices
        :math:`v_0, \ldots, v_{n-1}`, the adjacency matrix 
        of :math:`G(V, E)` is an 
        :math:`n`-dimensional matrix :math:`A`, defined as follows:
        
        .. math::
            A_{i,j} = \begin{cases}
                1, & \text{if } v_i \text{ is adjacent to } v_j,\\
                0, & \text{otherwise.}
            \end{cases}
    
        In weighted graphs, the entries of :math:`A` represent 
        the weights of the edges. The weight is a non-zero
        real number.

        .. todo::
            Add other return types depending on the stored matrix type.
        """
        data = np.ones(len(self._adj_matrix.indices), dtype=np.int8)
        indices = self._adj_matrix.indices
        indptr = self._adj_matrix.indptr
        # TODO: copy or not?
        return csr_array((data, indices, indptr))

    def laplacian_matrix(self):
        r"""
        Return the graph's Laplacian matrix.

        See Also
        --------
        adjacency_matrix

        Notes
        -----
        The Laplacian matrix is given by

        .. math::
            L = D - A,

        where :math:`A` is the graph's adjacency matrix
        and :math:`D` is the degree matrix

        .. math::
            D_{i, j} = \begin{cases}
                \deg(v_i), & \text{if } i = j\\
                0, & \text{otherwise}.
            \end{cases}

        The degree is calculated by the :meth:`hiperwalk.Graph.degree`
        method.
            
        """
        A = self.adjacency_matrix()
        D = A.sum(axis=1)
        if len(D.shape) == 1:
            D = np.array(D.ravel())
        else:
            D = np.array(D.ravel())[0]
        D = diags(D)
        return D - A

    def is_simple(self):
        r"""
        Return True if instance of simple graph.

        Notes
        -----
        .. todo::
            Decide if simple graph implementation accepts loops.
        """
        return True
