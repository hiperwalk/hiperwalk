import numpy as np
import scipy.sparse
from .graph import Graph
from warnings import warn

class Hypercube(Graph):
    r"""
    Hypercube graph.

    The hypercube has ``2**dimension`` vertices.

    Parameters
    ----------
    dimension : int
        Dimension of hypercube.

    Notes
    -----
    The vertex :math:`v` is adjacent to all vertices that
    have Hamming distance of 1.
    That is, :math:`v` is adjacent to
    :math:`v \oplus 2^0`, :math:`v \oplus 2^1`, :math:`\ldots`,
    :math:`v \oplus 2^{n - 2}`, and :math:`v \oplus 2^{n - 1}`,
    where :math:`\oplus` denotes bitwise XOR operation and
    :math:`n` is the hypercube dimension.

    The order of the arcs depends on on the XOR operation.
    Consider two arc,
    :math:`(u, u \oplus 2^i)` and :math:`(v, v \oplus 2^j`),
    with numerical labels :math:`a_1` and :math:`a_2`, respectively.
    Then, :math:`a_1 < a_2` if and only if either :math:`u < v` or
    :math:`u = v` and :math:`i < j`.
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

        The arc direction of ``(tail, head)`` is the number ``i``
        such that ``tail == head ^ 2**i``.

        Parameters
        ----------
        arc
            The arc in either arc notation (``(tail, head)``) or
            the arc label (``int``).

        Returns
        -------
        int
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
        direction = self.arc_direction((tail, head))
        return tail*self.dimension + direction

    def arc(self, label):
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
