import numpy as np

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
    adj_matrix : :class:`scipy.sparse.csr_array`
        Adjacency matrix of the graph on
        which the quantum walk takes place.

    Raises
    ------
    TypeError
        if ``adj_matrix`` is not an instance of
        :class:`scipy.sparse.csr_array`.

    Notes
    -----
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
        self.adj_matrix = adj_matrix
        self.coloring = None

    def default_coin(self):
        r"""
        Returns the default coin for the given graph.

        The default coin for a coined quantum walk on an arbitrary
        graph is ``grover``.
        """
        return 'grover'

    def embeddable(self):
        r"""
        Returns ``True`` if the graph can be embedded on the plane,
        and ``False`` otherwise.

        If a graph can be embedded on the plane,
        we can assign meaningful right or left directions,
        and clockwise or counter-clockwise rotations to edges and arcs

        Notes
        -----
        The implementation is class-dependent.
        We do not inspect the structure of the graph to determine whether
        it is embeddable or not.
        """
        return False

    def arc_label(self, tail, head):
        r"""
        Returns the numerical label of the arc.

        Parameters
        ----------
        tail: int
            Tail of the arc.

        head: int
            Head of the arc.

        Returns
        -------
        label: int
            Arc label.
        """
        return _binary_search(self.adj_matrix.indices, head,
                              start = self.adj_matrix.indptr[tail],
                              end = self.adj_matrix.indptr[tail + 1])

    def arc(self, label):
        r"""
        Converts the numerical label to arc notation.

        Given the arc label as a number,
        returns the arc in the ``(tail, head)`` notation.

        Parameters
        ----------
        label: int
            Arc label (number)

        Returns
        -------
        (int, int)
            Arc in the arc notation ``(tail, head)``.
        """
        adj_matrix = self.adj_matrix
        head = adj_matrix.indices[label]
        # TODO: binary search
        for tail in range(len(adj_matrix.indptr)):
            if adj_matrix.indptr[tail + 1] > label:
                break
        return (tail, head)

    def next_arc(self, arc):
        r"""
        Next arc in an embeddable graph.

        Parameters
        ----------
        arc
            The arc in any of the following notations.

            * arc notation: tuple of vertices
                In ``(tail, head)`` format where
                ``tail`` and ``head`` must be valid vertices.
            * arc label: int.
                The arc label (number).

        Returns
        -------
        Next arc in the same notation as the ``arc`` argument.

        See Also
        --------
        arc
        arc_label
        """
        # implemented only if is embeddable
        raise AttributeError

    def previous_arc(self, arc):
        r"""
        Previous arc in an embeddable graph.

        Parameters
        ----------
        arc
            The arc in any of the following notations.

            * arc notation: tuple of vertices
                In ``(tail, head)`` format where
                ``tail`` and ``head`` must be valid vertices.
            * arc label: int.
                The arc label (number).

        Returns
        -------
        Previous arc in the same notation as the ``arc`` argument.

        See Also
        --------
        arc
        arc_label
        """
        # implemented only if is embeddable
        raise AttributeError

    def neighbors(self, vertex):
        r"""
        Returns all neighbors of the given vertex.
        """
        start = self.adj_matrix.indptr[vertex]
        end = self.adj_matrix.indptr[vertex + 1]
        return self.adj_matrix.indices[start:end]

    def arcs_with_tail(self, tail):
        r"""
        Returns all arcs that have the given tail.
        """
        arcs_lim = self.adj_matrix.indptr
        return np.arange(arcs_lim[tail], arcs_lim[tail + 1])

    def number_of_vertices(self):
        r"""
        Cardinality of vertex set.
        """
        return self.adj_matrix.shape[0]

    def number_of_arcs(self):
        r"""
        Cardinality of arc set.

        For simple graphs, the cardinality is twice the number of edges.
        """
        return self.adj_matrix.sum()

    def number_of_edges(self):
        r"""
        Cardinality of edge set.
        """
        return self.adj_matrix.sum() >> 1

    def degree(self, vertex):
        r"""
        Degree of given vertex.
        """
        indptr = self.adj_matrix.indptr
        return indptr[vertex + 1] - indptr[vertex]
