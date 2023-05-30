import numpy as np

def _binary_search(v, elem, start=0, end=None):
    r"""
    expects sorted array and executes binary search in the subarray
    v[start:end] searching for elem.
    Return the index of the element if found, otherwise returns -1
    Cormen's binary search implementation.
    Used to improve time complexity
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
    Graph on which a quantum walk occurs.

    Used for generic graphs.

    Parameters
    ----------
    adj_matrix : :class:`scipy.sparse.csr_array`
        Adjacency matrix of the graph on
        which the quantum walk occurs.

    Raises
    ------
    TypeError
        if ``adj_matrix`` is not an instance of
        :class:`scipy.sparse.csr_array`.

    Notes
    -----
    Makes a plethora of methods available.
    These methods may be used by a Quantum Walk model for
    generating a valid walk.
    The default arguments for a given model are given by the graph.
    The Quantum Walk model is ignorant with this regard.

    This class may be passed as argument to plotting methods.
    Then the default representation for the specific graph will be shown.

    .. todo::
        Check if valid adjacency matrix
        
    """

    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix
        self.coloring = None

    def default_coin(self):
        r"""
        Returns the default coin for the given graph.

        The default coin for the coined quantum walk on general
        graphs is ``grover``.
        """
        return 'grover'

    def embeddable(self):
        r"""
        Returns whether the graph can be embedded on the plane or not.

        If a graph can be embedded on the plane,
        we can assign directions to edges and arcs.

        Notes
        -----
        The implementation is class-dependent.
        We do not check the graph structure to determine whether
        it is embeddable or not.
        """
        return False

    def arc_label(self, tail, head):
        r"""
        Returns arc label (number).

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
        Arc in arc notation.

        Given the arc label,
        returns it in the ``(tail, head)`` notation.

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

        For simple graphs, this is twice the number of edges.
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
