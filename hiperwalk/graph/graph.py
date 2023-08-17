import numpy as np
from scipy.sparse import issparse, csr_array

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
    Arbitrary graph.

    The graph on which a quantum walk takes place.

    Parameters
    ----------
    adj_matrix : :class:`scipy.sparse.csr_array` or :class:`numpy.ndarray`
        Adjacency matrix of the graph on
        which the quantum walk takes place.

    Raises
    ------
    TypeError
        if ``adj_matrix`` is not an square matrix.

    Notes
    -----
    .. todo::
        Check if it is more efficient to store the adjacency matrix as
        sparse or dense.

    A wide range of methods are available. 
    These methods can be used by a quantum walk model 
    to generate a valid quantum walk.
    
    This class can be passed as an argument to plotting functions.
    In this case,
    the default representation for the graph will be displayed.

    The recommended parameter type is
    :class:`scipy.sparse.csr_array` with ``dtype=np.int8``,
    where 1 denotes adjacency and 0 denotes non-adjacency.
    If any entry differs from 0 or 1,
    some methods may not operate as expected.

    Each edge of a given graph :math:`G(V, E)` is associated with
    two arcs in the symmetric digraph :math:`\vec{G}(V, A)`,
    where

    .. math::
        \begin{align*}
            A = \bigcup_{(v,u) \in E} \{(v, u), (u, v)\}.
        \end{align*}

    An arc can be described either in the arc notation as (tail,head) 
    or through a numerical label. The ordering of the labels in 
    the :obj:`Graph` class is as follows: 
    Consider two arcs, :math:`(v_1, u_1)` and :math:`(v_2, u_2')`, 
    with numerical labels :math:`a_1` and :math:`a_2` respectively. 
    Then, :math:`a_1 < a_2` if and only if either :math:`v_1 < v_2` 
    or :math:`v_1 = v_2` and :math:`u_1 < u_2`.

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
        # TODO:
        # * Check if valid adjacency matrix
        # * Add option: numpy dense matrix as parameters.
        # * Add option: networkx graph as parameter.
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
        Returns the numerical label of the arc.

        Parameters
        ----------
        *args:
            int:
                The numerical arc label itself is passed
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
            Arc label.

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
        Converts the arc number to arc notation.

        Given the arc number,
        returns the arc in the ``(tail, head)`` notation.

        Parameters
        ----------
        number: int
            The arc number

        Returns
        -------
        (int, int)
            Arc in the arc notation ``(tail, head)``.
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
        Returns all neighbors of the given vertex.
        """
        start = self._adj_matrix.indptr[vertex]
        end = self._adj_matrix.indptr[vertex + 1]
        return self._adj_matrix.indices[start:end]

    def arcs_with_tail(self, tail):
        r"""
        Returns all arcs that have the given tail.
        """
        arcs_lim = self._adj_matrix.indptr
        return np.arange(arcs_lim[tail], arcs_lim[tail + 1])

    def number_of_vertices(self):
        r"""
        Cardinality of vertex set.
        """
        return self._adj_matrix.shape[0]

    def number_of_arcs(self):
        r"""
        Cardinality of arc set.

        For simple graphs, the cardinality is twice the number of edges.
        """
        return self._adj_matrix.sum()

    def number_of_edges(self):
        r"""
        Cardinality of edge set.
        """
        return self._adj_matrix.sum() >> 1

    def degree(self, vertex):
        r"""
        Degree of given vertex.
        """
        indptr = self._adj_matrix.indptr
        return indptr[vertex + 1] - indptr[vertex]

    def vertex_number(self, vertex):
        r"""
        Returns vertex number given any vertex representation.

        By invoking this method,
        the vertex number is returned regardless of its representation.
        There are some graphs in which a vertex may have multiple
        representations.
        For example, coordinates in a grid.
        For general graphs,
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
                             "Expected integer value from 0 to" +
                             str(num_vert - 1))
        return vertex

    def adjacency_matrix(self):
        r"""
        Returns the graph adjacency matrix.

        Returns
        -------
        :class:`scipy.sparse.csr_array`.

        Notes
        -----
        .. todo::
            Add other return types depending on the stored matrix type.
        """
        return self._adj_matrix
