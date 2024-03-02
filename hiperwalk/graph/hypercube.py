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
    if __debug__:
        assert self.adjacent(vertex, neigh)

    # it is supposed that vertex and neigh are adjacent
    x = vertex ^ neigh
    return x.bit_length() - 1

def __degree(self, vertex):
    return self._dimension

def __number_of_vertices(self):
    return 1 << self._dim

def __number_of_edges(self):
    return 1 << (self._dim - 1) * self._dim

def __degree(self, vertex):
    return self._dim

def __dimension(self):
    r"""
    Hypercube dimension.

    .. todo::
        How to add to docs?

    Returns
    -------
    int
    """
    return self._dim

# graph constructor
import numpy as np
from scipy.sparse import csr_array
from types import MethodType
from .graph import Graph
def Hypercube(dim, weights=None, multiedges=None):
    r"""
    Hypercube graph.

    The hypercube graph consists of ``2**n`` vertices,
    where ``n`` is the hypercube *dimension*.
    The numerical labels of these vertices  are
    ``0``, ``1``, ..., ``2**n - 1``.
    Two vertices are adjacent
    if and only if the corresponding binary tuples
    differ by only one bit, indicating a Hamming distance of 1.

    .. todo::
        Update docs.
        Create generic constructor docs.

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
    if weights is not None or multiedges is not None:
        raise NotImplementedError()

    # adjacency matrix
    num_vert = 1 << dim
    num_arcs = dim*num_vert

    data = np.ones(num_arcs, dtype=np.int8)
    indptr = np.arange(0, num_arcs + 1, dim)
    indices = np.array([v ^ 1 << shift for v in range(num_vert)
                                       for shift in range(dim)])
    adj_matrix = csr_array((data, indices, indptr),
                           shape=(num_vert, num_vert))

    g = Graph(adj_matrix, copy=False)

    # Binding particular attributes and methods
    # TODO: add to docs
    g._dim = int(dim)

    g.adjacent = MethodType(__adjacent, g)
    g._neighbor_index = MethodType(__neighbor_index, g)
    g.degree = MethodType(__degree, g)
    g.number_of_vertices = MethodType(__number_of_vertices, g)
    g.number_of_edges = MethodType(__number_of_edges, g)
    g.degree = MethodType(__degree, g)
    g.dimension = MethodType(__dimension, g)

    return g
