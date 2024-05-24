import numpy as np
from scipy.sparse import issparse, csr_array, diags

def _binary_search(v, elem, start=0, end=None):
    r"""
    This function expects a sorted array and
    performs a binary search on the subarray
    v[start:end], looking for the element 'elem'.
    If the element is found, the function returns the index of the element.
    If the element is not found, the function returns -1.
    This binary search implementation follows Cormen's algorithm.
    It is used to improve the time complexity of the search process.
    """
    if end == None:
        end = len(v)
    
    index = end
    while start < index:
        mid = int((start + index)/2)
        if elem <= v[mid]:
            index = mid
        else:
            start = mid + 1

    return index if index < end and v[index] == elem else -1

def _interval_binary_search(v, elem, start = 0, end = None) -> int:
    r"""
    This function expects a sorted array and performs a
    binary search on the subarray v[start:end],
    looking for the interval with minimum length that
    contains the element.
    If the element is greater or equal than the element of index i
    and less than the element of index i+1,
    the function returns i.
    If the element is less than the minimum value in the array,
    the function `start - 1`.
    If the element is greater or equal than the maximum
    value in the array,
    the function returns the index of the maximum value.
    It is used to improve the time complexity of the search process.
    """
    if elem < v[start]:
        return start - 1
    
    if end is None:
        end = len(v)
    if elem >= v[end - 1]:
        return end - 1
    elif(elem >= v[start] and elem < v[start + 1]):
        return start
    index = None
    while start < end:
        mid = (start + end) // 2
        if(v[mid]<=elem and v[mid+1]>elem):
            index = mid
            break
        elif v[mid] <= elem:
            start = mid + 1
        else:
            end = mid
    return index

class Graph():
    r"""
    Constructs an arbitrary graph.

    This class defines the graph structure used for implementing a 
    quantum walk. It encapsulates all necessary properties
    and functionalities of the graph 
    required for the quantum walk dynamics.
    This class also supports the creation of graphs with loops.

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
        * :class:`networkx.Graph`.
            * The adjacency matrix is extracted from the graph.

    copy : bool, default=False
        If ``True``, a hard copy of ``adj_matrix`` is stored.
        If ``False``,
        the pointer to ``adj_matrix`` is stored and
        the ``adj_matrix``'s data is changed.

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

    Each vertex has at most one loop.

    .. warning::

        To reduce memory usage, ``adj_matrix.data`` is set to ``None``.
        This is possible because ``adj_matrix.data`` should be an
        array of ones.

        If the user wishes to keep the original ``adj_matrix``,
        the argument ``copy`` must be set to ``True``.

    The treatment of the graph depends on the quantum walk model. 

    The default **order of neighbors** is the ascending order.
    A different order of neighbors can be specified using a
    sparse matrix where
    :obj:`scipy.sparse.csr_array.has_sorted_indices` is ``False``.
    In this case,
    the neighbors of ``u`` are given by
    ``adj_matrix.indices[adj_matrix.indptr[u]:adj_matrix.indptr[u+1]]``.

    Examples
    --------

    .. testsetup::

        import hiperwalk as hpw
        import scipy.sparse

    Creating a complete graph with loops and
    the default order of neighbors (ascending order).

    .. doctest::

        >>> num_vert = 4
        >>> A = scipy.sparse.csr_array([[1]*num_vert]*num_vert)
        >>> g = hpw.Graph(A)
        >>> g.neighbors(0)
        array([0, 1, 2, 3], dtype=int32)
        >>> g.neighbors(1)
        array([0, 1, 2, 3], dtype=int32)

    Creating a complete graph with loops and
    the descending order of neighbors.

    .. doctest::

        >>> data = [1]*(num_vert**2)
        >>> indices = list(range(num_vert - 1, -1, -1))*num_vert
        >>> indptr = list(range(0, num_vert*(num_vert + 1), num_vert))
        >>> A = scipy.sparse.csr_array((data, indices, indptr))
        >>> g = hpw.Graph(A)
        >>> g.neighbors(0)
        array([3, 2, 1, 0])
        >>> g.neighbors(1)
        array([3, 2, 1, 0])
    """

    # def _default_dtype(self):
    #     return np.int8

    def _count_loops(self, adj_matrix):
        loops = [adj_matrix[v, v] != 0
                 for v in range(adj_matrix.shape[0])]
        self._num_loops = np.sum(loops)

    def _set_adj_matrix(self, adj_matrix):
        del adj_matrix.data
        adj_matrix.data = None
        self._adj_matrix = adj_matrix

    def __init__(self, adj_matrix, copy=False):
        # TODO: check if it is more efficient to store adj_matrix
        # as numpy array. Add numpy array manipulation

        try:
            adj_matrix.adj #throws AttributeError if not networkx graph
            import networkx as nx
            # adj_matrix = nx.adjacency_matrix(adj_matrix,
            #                                  dtype=self._default_dtype())
            adj_matrix = nx.adjacency_matrix(adj_matrix)
        except AttributeError:
            pass

        # TODO: store numpy matrix
        if not issparse(adj_matrix):
            # adj_matrix = csr_array(adj_matrix,
            #                        dtype=self._default_dtype())
            adj_matrix = csr_array(adj_matrix)

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise TypeError("Adjacency matrix is not square.")

        if copy:
            adj_matrix = adj_matrix.copy()

        self._count_loops(adj_matrix)
        self._set_adj_matrix(adj_matrix)

    def adjacent(self, u, v):
        r"""
        Return True if vertex ``u`` is adjacent to ``v``.
        """
        # TODO: check implementation of adj_matrix[u, v]
        # if adj_matrix.has_sorted_indices,
        # probably scipy is more efficient.
        # if indices are not sorted, scipy probably does a linear search,
        # and a graph-dependent implementation may be more efficient.
        # If adj_matrix.has_sorted_index and
        # scipy does not do binary search, implement it.

        # this function is reimplemented because data=None
        # to use less memory
        u = self.vertex_number(u)
        v = self.vertex_number(v)
        A = self._adj_matrix

        if A.has_sorted_indices:
            i = _binary_search(A.indices, v, start=A.indptr[u],
                               end=A.indptr[u + 1])
            return i != -1

        return v in A.indices[A.indptr[u]:A.indptr[u+1]]

    def _entry(self, row, col):
        entry = self._adj_matrix.indptr[row] + 1
        offset = self._neighbor_index(row, col)

        return entry + offset

    def _find_entry(self, entry):
        r"""
        Find the corresponding row and column of the given entry.

        Notes
        -----
        The adjacency matrix is not stored as a list of 1's.
        Instead, each entry represents the number of ones up
        to that point.
        """
        adj_matrix = self._adj_matrix

        head = adj_matrix.indices[entry]
        tail = _interval_binary_search(adj_matrix.indptr, entry)

        return (tail, head)

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

        adj_matrix = self._adj_matrix
        start = adj_matrix.indptr[vertex]
        end = adj_matrix.indptr[vertex + 1]

        # if indices is in ascending order
        if adj_matrix.has_sorted_indices:
            index = _binary_search(adj_matrix.indices,
                                   neigh,
                                   start=start,
                                   end=end)
            if index >= 0:
                return index - start
            index = end # will raise ValueError down below

        # indices is not in ascending order
        for index in range(start, end):
            if adj_matrix.indices[index] == neigh:
                return index - start

        raise ValueError("Vertices " + str(vertex) + " and "
                         + str(neigh) + " are not adjacent.")

    def neighbors(self, vertex):
        r"""
        Return all neighbors of the given vertex.

        This method retrieves all vertices adjacent to the 
        specified vertex. If the graph allows loops, 
        the vertex may also be considered its 
        own neighbor if such an edge exists.

        Parameters
        ----------
        vertex : int
            The vertex for which neighbors are to be found.

        Returns
        -------
        list of int:
            The neighbors of ``vertex``. 
            The order of the neighbors may vary based on 
            the graph's internal representation.

        Notes
        -----
        The specific order of the neighbors in the returned 
        list is not guaranteed and may change if the 
        underlying graph structure is modified.
        """
        vertex = self.vertex_number(vertex)
        start = self._adj_matrix.indptr[vertex]
        end = self._adj_matrix.indptr[vertex + 1]
        return self._adj_matrix.indices[start:end]

    def number_of_vertices(self):
        r"""
        Return the total number of vertices in the graph.

        This method calculates and returns the cardinality 
        of the vertex set, effectively providing the 
        count of all vertices present in the graph.

        Returns
        -------
        int
            The total number of vertices in the graph.
        """
        return self._adj_matrix.shape[0]

    def number_of_edges(self):
        r"""
        Return the total number of edges in the graph.

        This method calculates and returns the cardinality 
        of the edge set, indicating the total count of 
        all edges present in the graph, including loops.

        Returns
        -------
        int
            The total number of edges in the graph.
        """
        non_loops = len(self._adj_matrix.indices) - self._num_loops
        num_edges = non_loops >> 1
        return  num_edges + self._num_loops

    def number_of_loops(self):
        r"""
        Return the number of loops in the graph.

        This method counts and returns the number of loops 
        in the graph. A loop is defined as an edge that 
        connects a vertex to itself.

        Returns
        -------
        int
            The number of loops in the graph.
        """
        return self._num_loops

    def degree(self, vertex):
        r"""
        Return the degree of the given vertex.

        The degree of a vertex :math:`u` in a graph 
        is the number of edges  incident to :math:`u`.
        Each Loop is counted once.

        Parameters
        ----------
        vertex :
            Any valid vertex representation.

        Notes
        -----
        The Graph class supports graphs with loops.
        """
        vertex = self.vertex_number(vertex)
        indptr = self._adj_matrix.indptr
        return indptr[vertex + 1] - indptr[vertex]

    def vertex_number(self, vertex):
        r"""
        Return the numerical label of a vertex based on its representation.

        This method is designed to handle graphs where vertices might have 
        multiple representations, such as coordinates in a grid. 
        In cases of arbitrary graphs, this function will return 
        the input itself if it is a valid representation.

        Parameters
        ----------
        vertex : int
            The vertex in any valid representation. For arbitrary graphs, only the vertex's label is accepted.

        Returns
        -------
        int
            The numerical label of the vertex.

        Raises
        ------
        ValueError
            If the ``vertex`` is not a valid representation.

        Notes
        -----
        Implementing this function is beneficial for arbitrary graphs 
        as it simplifies the implementation of some quantum walk methods.
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
        Return the adjacency matrix representation of the graph.

        Return a **hard copy** of the adjacency matrix.

        Returns
        -------
        :class:`scipy.sparse.csr_array`.

        Notes
        -----
        In a simple graph :math:`G(V, E)` with :math:`n` vertices
        :math:`v_0, \ldots, v_{n-1}`, the adjacency matrix 
        of :math:`G(V, E)` is an 
        :math:`n`-dimensional matrix :math:`A`, defined as follows:        

        .. math::
            A_{i,j} = \begin{cases}
                1, & \text{if } v_i \text{ is adjacent to } v_j,\\
                0, & \text{otherwise.}
            \end{cases}
    
        If :math:`G(V, E)` has a loop on vertex :math:`v_i`, 
        then :math:`A_{i,i}=1`.
        
        In multigraphs, where multiple edges can exist between
        the same pair of vertices, the adjacency matrix reflects this
        by counting the number of such edges.
        """
        data = np.ones(len(self._adj_matrix.indices), dtype=np.int64)
        indices = self._adj_matrix.indices
        indptr = self._adj_matrix.indptr

        return csr_array((data, indices, indptr), copy=True)

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
        Return True if the graph has no multiedges or weights.

        A simple graph, in this context, does not include multiple 
        edges between the same pair of vertices or 
        any weighted edges, although loops are permitted.

        Notes
        -----
        The Graph class allows for loops in the graph.
        """
        return True

    def _rearrange_matrix_indices(self, matrix):
        r"""
        Rearrange `matrix.indices` accoring to the
        Graph's neighbor order.
        """
        # TODO: Check if ´has_sorted_indices' changes

        indices = matrix.indices
        adj_indices = self._adj_matrix.indices

        for row in range(len(self._adj_matrix.indptr) - 1):
            start = self._adj_matrix.indptr[row]
            end = self._adj_matrix.indptr[row + 1]

            for i in range(start, end):
                if indices[i] != adj_indices[i]:
                    # find right position
                    j = np.where(indices[start:end] == adj_indices[i])
                    j = start + j[0][0]

                    # swap indices
                    temp = indices[i]
                    indices[i] = indices[j]
                    indices[j] = temp

                    # swap data
                    temp = matrix.data[i]
                    matrix.data[i] = matrix.data[j]
                    matrix.data[j] = temp

    def _dict_to_adj_matrix(self, dictionary):
        r"""
        Receives a dictionary and returns the
        adjacency matrix with values overwritten by the dictionary.

        Parameters
        ----------
        dictionary: dict
            A dictionary with entries {`(u, v): value`}

        Raises
        ------
        ValueError
            If trying to remove an edge from the graph --
            assign a value of 0 to an existing edge --, or
            if trying to insert an edge into the graph --
            assign a value to a non-existing edge.
        """
        # expects a dictionary with non-zero values
        # the key corresponds to an edge
        #   edge must be a tuple
        #   raises an error if the edge does not exist
        # the value that corresponds to the given edge

        d = dictionary
        data = np.ones(self._adj_matrix.indices.shape[0],
                       dtype=np.array(list(d.values())).dtype)
        indices = self._adj_matrix.indices
        indptr = self._adj_matrix.indptr

        adj_matrix = csr_array((data, indices, indptr))

        for key in d:
            u = self.vertex_number(key[0])
            v = self.vertex_number(key[1])

            if adj_matrix[u, v] == 0 and d[key] != 0:
                raise ValueError('Inexistent edge ' + str(key))

            if adj_matrix[u, v] != 0 and d[key] == 0:
                raise ValueError('Edge ' + str(key)
                                 + 'cannot be assigned a value of 0.')

            adj_matrix[u, v] = d[key]
            adj_matrix[v, u] = d[key]

        return adj_matrix
