import numpy as np
import networkx as nx
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
    Represents an arbitrary graph.

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
    
    The graph :math:`G(V,E)` on which the quantum walk 
    takes place is specified by the
    adjacency matrix, Laplacian matrix, or 
    any real Hermitian matrix :math:`C`.
    Let :math:`V` be the vertex set :math:`\{v_0,...,v_{n-1}\}`, 
    where :math:`n=|V|`.
    Two distinct vertices :math:`v_i` and :math:`v_j` in :math:`V`
    are adjacent if and only if :math:`C_{ij}\neq 0`.

    The class methods facilitate the construction of a valid quantum walk 
    and can be provided as parameters to plotting functions. For visualizations, 
    the default graph representation will be used. Specific classes are available 
    for well-known graph types, such as hypercubes and lattices.

    The preferred parameter type for the adjacency matrix is
    :class:`scipy.sparse.csr_matrix` with ``dtype=np.int8``.

    Each edge in the graph :math:`G(V, E)` that connects two distinct
    vertices corresponds to a pair of arcs in the associated directed 
    graph :math:`\vec{G}(V, A)`, where

    .. math::
        \begin{align*}
            A = \bigcup_{v_k v_\ell \in E} \{(v_k, v_\ell), (v_\ell, v_k)\}.
        \end{align*}

    Arcs can be represented using the (tail,head) notation or with numerical labels. 
    In the :obj:`Graph` class, the arc labels are ordered such that for two arcs, 
    :math:`(v_i, v_j)` and :math:`(v_k, v_\ell)`, with labels :math:`a_1` and 
    :math:`a_2` respectively, :math:`a_1 < a_2` if and only if :math:`i < k` or 
    (:math:`i = k` and :math:`j < \ell`). 
    Note that a loop is represented as a single arc.
    
    If ``adj_matrix`` is specified as a real Hermitian matrix :math:`C`, 
    then :math:`C_{ij}` represents the weight of the arc :math:`(v_i, v_j)`. 
    This weight is considered a generalized weight when :math:`C_{ij}` is 
    negative. 

    .. note::
        The arc ordering may change for graphs defined using specific classes.

    For example, the graph :math:`G(V, E)` shown in
    Figure 1 has an adjacency matrix ``adj_matrix``.

    .. testsetup::

        import numpy as np

    >>> adj_matrix = np.array([
    ...     [0, 1, 0, 0],
    ...     [1, 0, 1, 1],
    ...     [0, 1, 0, 1],
    ...     [0, 1, 1, 0]])
    >>> adj_matrix
    array([[0, 1, 0, 0],
           [1, 0, 1, 1],
           [0, 1, 0, 1],
           [0, 1, 1, 0]])

    .. graphviz:: ../../graphviz/graph-example.dot
        :align: center
        :layout: neato
        :caption: Figure 1

    The arcs of the associated digraph in the arc notation are

    >>> arcs = [(i, j) for i in range(4)
    ...                for j in range(4) if adj_matrix[i,j] == 1]
    >>> arcs
    [(0, 1), (1, 0), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]

    Note that ``arcs`` is already sorted, hence the associated 
    numeric labels are

    >>> arcs_labels = {arcs[i]: i for i in range(len(arcs))}
    >>> arcs_labels
    {(0, 1): 0, (1, 0): 1, (1, 2): 2, (1, 3): 3, (2, 1): 4, (2, 3): 5, (3, 1): 6, (3, 2): 7}

    The numeric labels are depicted in Figure 2.

    .. graphviz:: ../../graphviz/graph-arcs.dot
        :align: center
        :layout: neato
        :caption: Figure 2

    If we insert the labels of the arcs into the adjacency matrix,
    we obtain matrix ``adj_labels`` as follows:

    >>> adj_labels = [[arcs_labels[(i,j)] if (i,j) in arcs_labels
    ...                                   else '' for j in range(4)]
    ...               for i in range(4)]
    >>> adj_labels = np.matrix(adj_labels)
    >>> adj_labels
    matrix([['', '0', '', ''],
            ['1', '', '2', '3'],
            ['', '4', '', '5'],
            ['', '6', '7', '']], dtype='<U21')

    Note that, intuitively,
    the arcs are labeled in left-to-right and top-to-bottom fashion.
    """

    def __init__(self, adj_matrix):
        if all(hasattr(adj_matrix, attr) for attr in
               ['__len__', 'edges', 'nbunch_iter', 'subgraph',
                'is_directed']):
            adj_matrix = nx.convert_matrix.to_scipy_sparse_array(
                    adj_matrix).astype(np.int8)

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

    def arc_number(self, *args):
        r"""
        Return the numerical label of the arc.

        Parameters
        ----------
        *args:
            int:
                The arc's numerical label itself is passed
                as argument.
            (tail, head):
                Arc in arc notation.
            tail, head:
                Arc in arc notation,
                but ``tail`` and ``head`` are passed as
                different arguments, not as a tuple.

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
        >>> graph.arc_number(0, 1) #tail and head in separate arguments
        0
        """
        arc = (args[0], args[1]) if len(args) == 2 else args[0]

        if not hasattr(arc, '__iter__'):
            num_arcs = self.number_of_arcs()
            if arc < 0 and arc >= num_arcs:
                raise ValueError("Arc value out of range. "
                                 + "Expected arc value from 0 to "
                                 + str(num_arcs - 1))
            return int(arc)

        tail, head = arc
        arc_number = _binary_search(self._adj_matrix.indices, head,
                                   start = self._adj_matrix.indptr[tail],
                                   end = self._adj_matrix.indptr[tail + 1])
        if arc_number == -1:
            raise ValueError("Inexistent arc " + str(arc) + ".")
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

        The degree of a vertex :math:`u` in a graph, 
        which may include a loop, is the number of edges 
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
        :math:`v_0, \ldots, v_{n-1}`, the adjacency matrix of :math:`G(V, E)` is an 
        :math:`n`-dimensional matrix :math:`A`, defined as follows:
        
        .. math::
            A_{i,j} = \begin{cases}
                1, & \text{if } v_i \text{ is adjacent to } v_j,\\
                0, & \text{otherwise.}
            \end{cases}
    
        In weighted graphs, the entries of :math:`A` represent the weights of the edges.

        .. todo::
            Add other return types depending on the stored matrix type.
        """
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

        where :math:`D` is the degree matrix

        .. math::
            D_{i, j} = \begin{case}
                degree(v_i) &\text{if } i = j
                0 &\text{otherwise},
            \end{cases}

        and :math:`A` is the graph adjacency matrix.
        """
        A = self.adjacency_matrix()
        D = A.sum(axis=1)
        if len(D.shape) == 1:
            D = np.array(D.ravel())
        else:
            D = np.array(D.ravel())[0]
        D = diags(D)
        return D - A
