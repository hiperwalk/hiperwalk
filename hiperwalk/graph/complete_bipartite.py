import numpy as np
from .graph import Graph
from types import MethodType
from scipy.sparse import eye

def adjacent(self, u, v):
    if u >= self._num_vert or v >= self._num_vert:
        raise ValueError("Received vertices " + str(u) + " and " + str(v)
                         + ". Maximum expected value is "
                         + str(self._num_vert - 1) + ".")
    return ((u < self._num_vert1 and v >= self._num_vert1)
            or (v < self._num_vert1 and u >= self._num_vert2))

def _entry(self, lin, col):
    entry = 1
    if lin < self._num_vert1:
        entry += lin*self._num_vert2
        entry += col - self._num_vert1
        return entry

    entry += self.number_of_edges()
    entry += (lin - self._num_vert1)*self._num_vert1
    entry += col
    return entry


def _find_entry(self, entry):
    num_edges = self.number_of_edges()
    if entry < num_edges:
        lin = entry // self._num_vert1
        col = entry % self._num_vert1
        return (lin, col)

    entry -= num_edges
    lin = entry // self._num_vert2
    col = entry % self._num_vert2
    return (lin, col)

def _neighbor_index(self, vertex, neigh):
    if neigh < self._num_vert1:
        return neigh
    return neigh - self._num_vert1

def neighbors(self, vertex):
    if vertex >= self._num_vert1:
        return np.arange(self._num_vert1)
    return np.arange(self._num_vert1, self._num_vert)

def number_of_vertices(self):
    return self._num_vert

def number_of_edges(self):
    return self._num_vert1*self._num_vert2

def degree(self, vertex):
    if vertex >= 0 and vertex < self._num_vert1:
        return self._num_vert2
    if vertex >= self._num_vert1 and vertex < self._num_vert: 
        return self._num_vert1

    raise ValueError("Vertex out of range. Expected value from 0 to "
                     + str(self._num_vert - 1) + ". But received "
                     + str(vertex) + " instead.")

def adjacency_matrix(self):
    # it is more efficient to store the dense adj matrix than
    # the sparse one when explicit values are used
    A = np.zeros((self._num_vert1, self._num_vert1), dtype=np.int8)
    B = np.ones((self._num_vert1, self._num_vert2), dtype=np.int8)
    C = np.ones((self._num_vert2, self._num_vert1), dtype=np.int8)
    D = np.zeros((self._num_vert2, self._num_vert2), dtype=np.int8)
    return np.block([[A, B],
                     [C, D]])

def laplacian_matrix(self):
    raise NotImplementedError()

def CompleteBipartite(num_vert1, num_vert2, weights=None, multiedges=None):
    r"""
    Complete bipartite graph.

    Parameters
    ----------
    num_vert1: int
        Number of vertices of the first partition.
    num_vert2: int
        Number of vertices of the second partition.

    Notes
    -----
    The vertices in the first partition are labeled from
    0 to ``num_vert1 - 1`` and the vertices of the
    second partition are labeled from
    ``num_vert1`` to ``num_vert1 + num_vert2 - 1``.
    """

    if weights is not None or multiedges is not None:
        raise NotImplementedError()

    if num_vert1 <= 0 or num_vert2 <= 0:
        raise ValueError("There must be at least one vertex "
                         + "in each partition.")

    # toy graph
    g = Graph(eye(num_vert1 + num_vert2))

    # changes attributes
    del g._adj_matrix
    g._adj_matrix = None

    g._num_vert1 = int(num_vert1)
    g._num_vert2 = int(num_vert2)
    g._num_vert = g._num_vert1 + g._num_vert2

    g.adjacent = MethodType(adjacent, g)
    g._entry = MethodType(_entry, g)
    g._find_entry = MethodType(_find_entry, g)
    g._neighbor_index = MethodType(_neighbor_index, g)
    g.neighbors = MethodType(neighbors, g)
    g.number_of_vertices = MethodType(number_of_vertices, g)
    g.number_of_edges = MethodType(number_of_edges, g)
    g.degree = MethodType(degree, g)
    g.adjacency_matrix = MethodType(adjacency_matrix, g)
    g.laplacian_matrix = MethodType(laplacian_matrix, g)

    return g
