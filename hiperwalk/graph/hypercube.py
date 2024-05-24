from .multigraph import Multigraph
from .weighted_graph import WeightedGraph

def __adjacent(self, u, v):
    x = u ^ v #bitwise xor
    return x != 0 and x & (x - 1) == 0

    # TODO: check if the following strategy is faster
    # try:
    #     # python >= 3.10
    #     count = x.bit_count()
    # except:
    #     count = bin(x).count('1')
    # return count == 1

def __neighbor_index(self, vertex, neigh):
    # TODO: how to use __debug__?
    # how to unable __debug__ when uploading to pip?
    # TODO: throw value error if not adjacent
    if __debug__:
        assert self.adjacent(vertex, neigh)

    # it is supposed that vertex and neigh are adjacent
    x = vertex ^ neigh

    # numpy integers do not have bit_length
    # TODO: check if it is faster fo convert or to calculate
    # np.ceil(np.log2(x + 1)).astype(int)
    x = int(x)
    return x.bit_length() - 1

def __degree(self, vertex):
    return self._dimension

def __number_of_vertices(self):
    return 1 << self._dim

def __number_of_edges(self):
    return (1 << (self._dim - 1)) * self._dim

def __degree(self, vertex):
    return self._dim

def dimension(self):
    r"""
    The dimension of the Hypercube.

    Returns
    -------
    int

    Examples
    --------
    .. testsetup::

        from sys import path
        path.append('..')
        import hiperwalk as hpw

    .. doctest::

        >>> n = 10
        >>> g = hpw.Hypercube(10)
        >>> g.dimension() == n
        True
    """
    return self._dim

# graph constructor
import numpy as np
from scipy.sparse import csr_array
from types import MethodType
from .graph import Graph

def Hypercube(dim, multiedges=None, weights=None, copy=False):
    r"""
    Hypercube graph constructor.

    The hypercube graph consists of ``2**dim`` vertices.
    The numerical labels of these vertices  are
    ``0``, ``1``, ..., ``2**dim - 1``.
    Two vertices are adjacent
    if and only if the corresponding binary tuples
    differ by only one bit, indicating a Hamming distance of 1.

    Parameters
    ----------
    dim : int
        The dimension of the hypercube.

    multiedges, weights: matrix or dict, default=None
        See :ref:`graph_constructors`.

    copy : bool, default=False
        See :ref:`graph_constructors`.

    Returns
    -------
    :class:`hiperwalk.Graph`
        See :ref:`graph_constructors` for details.

    See Also
    --------
    :ref:`graph_constructors`.

    Notes
    -----
    A vertex :math:`v` in the hypercube is adjacent to all other vertices
    that have a Hamming distance of 1. To put it differently, :math:`v`
    is adjacent to :math:`v \oplus 2^0`, :math:`v \oplus 2^1`,
    :math:`\ldots`, :math:`v \oplus 2^{n - 2}`, and
    :math:`v \oplus 2^{n - 1}`.
    Here, :math:`\oplus` represents the bitwise XOR operation,
    and :math:`n` signifies the dimension of the hypercube.

    The **order of neighbors** is determined by the XOR operation.
    The neighbors of vertex :math:`u` are given in the following order:
    :math:`u \oplus 2^0`, :math:`u \oplus 2^1, \ldots,`
    :math:`u \oplus 2^{n - 1}`.
    For example,

    .. testsetup::

        import hiperwalk as hpw

    .. doctest::

        >>> g = hpw.Hypercube(4)
        >>> u = 10
        >>> bin(u)
        '0b1010'
        >>> neigh = g.neighbors(u)
        >>> neigh
        array([11,  8, 14,  2])
        >>> [bin(v) for v in neigh]
        ['0b1011', '0b1000', '0b1110', '0b10']
        >>> [u^v for v in neigh]
        [1, 2, 4, 8]
    """
    if weights is not None and multiedges is not None:
        raise ValueError(
            "Both `weights` and `multiedges` arguments were set. "
            + "Cannot decide whether to create a weighted graph or "
            + "a multigraph."
        )

    # adjacency matrix
    num_vert = 1 << dim
    num_arcs = dim*num_vert

    data = np.ones(num_arcs, dtype=np.int64)
    indptr = np.arange(0, num_arcs + 1, dim)
    indices = np.array([v ^ 1 << shift for v in range(num_vert)
                                       for shift in range(dim)])
    adj_matrix = csr_array((data, indices, indptr),
                           shape=(num_vert, num_vert))
    g = Graph(adj_matrix, copy=False)

    data = weights if weights is not None else multiedges
    if data is not None:
        if hasattr(data, 'keys'):
            data = g._dict_to_adj_matrix(data)
            copy = False
        else:
            g._rearrange_matrix_indices(data)

        del g

    if weights is not None:
        g = WeightedGraph(data, copy=copy)

    elif multiedges is not None:
        g = Multigraph(data, copy=copy)

    if multiedges is None:
        # bind specific simple/weighted graph methods
        g.degree = MethodType(__degree, g)
        g.number_of_edges = MethodType(__number_of_edges, g)

    # Binding common attributes and methods
    g._dim = int(dim)
    g._num_loops = 0

    g.adjacent = MethodType(__adjacent, g)
    g._neighbor_index = MethodType(__neighbor_index, g)
    g.number_of_vertices = MethodType(__number_of_vertices, g)
    g.dimension = MethodType(dimension, g)

    return g
