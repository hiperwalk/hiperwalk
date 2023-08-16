import numpy as np
import scipy.sparse
from .graph import *
from warnings import warn

class Hypercube(Graph):
    r"""
    Hypercube graph.

    The hypercube graph consists of ``2**n`` vertices,
    where ``n`` is the hypercube *dimension*.
    The numerical labels of these vertices  are
    ``0``, ``1``, ..., ``2**n - 1``.
    Two vertices are adjacent
    if and only if the corresponding binary tuples
    differ by only one bit, indicating a Hamming distance of 1.

    Parameters
    ----------
    dimension : int
        The dimension of the hypercube.

    Notes
    -----
    A vertex :math:`v` in the hypercube is adjacent to all other vertices
    that have a Hamming distance of 1. To put it differently, :math:`v`
    is adjacent to :math:`v \oplus 2^0`, :math:`v \oplus 2^1`,
    :math:`\ldots`, :math:`v \oplus 2^{n - 2}`, and
    :math:`v \oplus 2^{n - 1}`.
    Here, :math:`\oplus` represents the bitwise XOR operation,
    and :math:`n` signifies the dimension of the hypercube.

    The order of the arcs is determined by the XOR operation.
    Consider two arcs, :math:`(u, u \oplus 2^i)` and :math:`(v, v \oplus 2^j)`,
    labeled numerically as :math:`a_1` and :math:`a_2`, respectively.
    The condition :math:`a_1 < a_2` is true if and only
    if either :math:`u < v` is true, or
    both :math:`u = v` and :math:`i < j` are true.
    """
    def __init__(self, dimension):
        num_vert = 1 << dimension
        num_arcs = dimension*num_vert

        data = np.ones(num_arcs, dtype=np.int8)
        indptr = np.arange(0, num_arcs + 1, dimension)
        indices = np.array([v ^ 1 << shift for v in range(num_vert)
                                           for shift in range(dimension)])

        adj_matrix = scipy.sparse.csr_array((data, indices, indptr),
                                            shape=(num_vert, num_vert))

        super().__init__(adj_matrix)
        self._dimension = int(dimension)

    def arc_direction(self, arc):
        r"""
        Returns the arc direction.

        The arc direction of ``(tail, head)`` is the number ``i``
        such that ``tail == head ^ 2**i``.

        Parameters
        ----------
        arc
            The arc can be represented in either arc notation (``(tail, head)``)
            or by using the numerical label  (``int``).

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

        if count != 1 or direction < 0 or direction >= self._dimension:
            raise ValueError("Arc " + str(arc) + " does not exist.")

        return direction

    def arc_number(self, *args):
        arc = (args[0], args[1]) if len(args) == 2 else args[0]

        if not hasattr(arc, '__iter__'):
            return super().arc_number(arc)

        tail, head = arc
        direction = self.arc_direction((tail, head))
        return tail*self._dimension + direction

    def arc(self, number):
        tail = number // self._dimension
        direction = number - tail*self._dimension
        head = tail ^ (1 << direction)
        return (tail, head)

    def degree(self, vertex):
        return self._dimension

    def dimension(self):
        r"""
        Hypercube dimension.

        Returns
        -------
        int
        """
        return self._dimension
