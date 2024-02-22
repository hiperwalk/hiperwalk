import numpy as np
from scipy.sparse import issparse, csr_array, diags

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
    
    The class methods facilitate the construction of a valid quantum walk 
    and can be provided as parameters to plotting functions. For visualizations, 
    the default graph representation will be used. Specific classes are available 
    for well-known graph types, such as hypercubes and lattices.

    The preferred parameter type for the adjacency matrix is
    :class:`scipy.sparse.csr_matrix` with ``dtype=np.int8``.

    The treatment of the graph depends on the quantum walk model. 
    .. todo::
        Reference new part of documentation.
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

    def arc_number(self, arc):
        r"""
        Return the numerical label of the arc.

        Parameters
        ----------
        arc:
            int:
                The arc's numerical label itself is passed
                as argument.
            (tail, head):
                Arc in arc notation.

        Returns
        -------
        label: int
            Numerical label of the arc.

        Examples
        --------
        If arc ``(0, 1)`` exists, the following commands return
        the same result.

        .. testsetup::

            import networkx as nx
            from sys import path
            path.append('../..')
            import hiperwalk as hpw
            nxg = nx.cycle_graph(10)
            adj_matrix = nx.adjacency_matrix(nxg)
            graph = hpw.Graph(adj_matrix)

        >>> graph.arc_number(0) #arc number 0
        0
        >>> graph.arc_number((0, 1)) #arc as tuple
        0
        """
        if not hasattr(arc, '__iter__'):
            num_arcs = self.number_of_arcs()
            if arc < 0 and arc >= num_arcs:
                raise ValueError("Arc value out of range. "
                                 + "Expected arc value from 0 to "
                                 + str(num_arcs - 1))
            return int(arc)

        tail = self._graph.vertex_number(arc[0])
        head = self._graph.vertex_number(arc[1])
        # TODO: the behavior may change after updating neighbors()
        # TODO: the behavior will change for multigraphs
        arc_number = self._adj_matrix.indptr[tail]

        offset = np.where(self.neighbors(head) == tail)
        if len(offset) != 1:
            raise ValueError("Inexistent arc " + str(arc) + ".")
        offset = offset[0]

        arc_number += offset
        return arc_number


    def arc(self, number):
        r"""
        Convert a numerical label to arc notation.
    
        Given an integer that represents the numerical label of an arc,
        this method returns the corresponding arc in ``(tail, head)`` 
        representation.
    
        Parameters
        ----------
        number : int
            The numerical label of the arc.
    
        Returns
        -------
        (int, int)
            The arc represented in ``(tail, head)`` notation.
        """

        adj_matrix = self._adj_matrix
        head = adj_matrix.indices[number]
        #TODO: binary search
        for tail in range(len(adj_matrix.indptr)):
            if adj_matrix.indptr[tail + 1] > number:
                break
        return (tail, head)

    def adjacent(self, u, v):
        r"""
        Return True if vertex ``u`` is adjacent to ``v``.
        """
        # TODO: check implementation of adj_matrix[u, v]
        # if adj_matrix.has_sorted_index, probably scipy is more efficient.
        # if indices are not sorted, scipy probably does a linear search,
        # and a graph-dependent implementation may be more efficient.
        try:
            u = self.vertex_number(u)
            v = self.vertex_number(v)
        except ValueError:
            return False # u or v is not a valid vertex
        return self._adj_matrix[u, v] != 0

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
        for i in range(start, end):
            if adj_matrix.indices[i] == neigh:
                return i - start

    def neighbors(self, vertex):
        r"""
        Return all neighbors of the given vertex.
        """
        start = self._adj_matrix.indptr[vertex]
        end = self._adj_matrix.indptr[vertex + 1]
        return self._adj_matrix.indices[start:end]

    def arcs_with_tail(self, tail):
        r"""
        Return all arcs that have the given tail.
        """
        arcs_lim = self._adj_matrix.indptr
        return np.arange(arcs_lim[tail], arcs_lim[tail + 1])

    def number_of_vertices(self):
        r"""
        Determine the cardinality of the vertex set.
        """
        return self._adj_matrix.shape[0]


    def number_of_arcs(self):
        r"""
        Determine the cardinality of the arc set.

        In simple graphs, the cardinality of the arc set is 
        equal to twice the number of edges. 
        However, for graphs containing loops, the 
        cardinality is incremented by one for each loop.
        """

        return self._adj_matrix.sum()

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
        # TODO: return hard copy depending on argument
        return self._adj_matrix

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
