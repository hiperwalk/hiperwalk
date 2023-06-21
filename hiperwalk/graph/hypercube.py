import numpy as np
import scipy.sparse
from .graph import Graph
from warnings import warn

class Hypercube(Graph):
    r"""
    Hypercube graph.

    The Hypercube has ``2**dimension`` vertices.

    Parameters
    ----------
    dimension : int
        Dimension of Hypercube.

    Notes
    -----
    Order of the arcs....
    """
    def __init__(self, dimension):
        num_vert = 1 << dimension
        num_arcs = dimension*num_vert

        data = np.ones(num_arcs, dtype=np.int8)
        indptr = np.arange(0, num_arcs + 1, dimension)
        indices = np.array([v ^ 1 << shift for v in range(num_vert)
                                           for shift in range(dimension)])
        # for v in range(num_vert):
        #     indices[v*dimension:(v + 1)*dimension].sort()

        adj_matrix = scipy.sparse.csr_array((data, indices, indptr),
                                            shape=(num_vert, num_vert))

        super().__init__(adj_matrix)
        self.dimension = int(dimension)

    def embeddable(self):
        return False

    def arc_direction(self, arc):
        r"""
        Return arc direction.

        Parameters
        ----------
        arc
            The arc in either arc notation (``(tail, head)``) or
            the arc label (``int``).

        Returns
        -------
        int
            The bit of the n-tuple that is equal to 1.
            From less significant to most significant counting from 0:
            00001 ~ 0
            00010 ~ 1
            etc.
        """
        try:
            tail, head = arc
        except TypeError:
            tail, head = self.arc(arc)

        direction = tail ^ head
        try:
            # python >= 3.10
            count = direction.bit_count()
        except:
            count = bin(direction).count('1')

        try:
            direction = direction.bit_length() - 1
        except:
            direction = np.log2(direction)

        if count != 1 or direction < 0 or direction >= self.dimension:
            raise ValueError("Arc " + str(arc) + " does not exist.")

        return direction

    def arc_label(self, tail, head):
        """
        Return arc label (number).

        Parameters
        ----------
        tail: int
            Arc tail.
        head: int
            Arc head.
        """

        direction = self.arc_direction((tail, head))
        return tail*self.dimension + direction

    def arc(self, label):
        r"""
        Return Arc in arc notation.

        Given the arc label (a number),
        return the arc in the ``(tail, head)`` notation.

        Parameters
        ----------
        label : int
            Arc label (number).

        Returns
        -------
        (tail, head)
        """
        tail = label // self.dimension
        direction = label - tail*self.dimension
        head = tail ^ (1 << direction)
        return (tail, head)

    def degree(self, vertex):
        return self.dimension

    def dimension(self):
        r"""
        Hypercube dimension.

        Returns
        -------
        int
        """
        return self.dimension
