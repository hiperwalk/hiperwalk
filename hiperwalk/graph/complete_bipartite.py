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

def _entry(self, row, col):
    entry = 1
    if row < self._num_vert1:
        entry += row*self._num_vert2
        entry += col - self._num_vert1
        return entry

    entry += self.number_of_edges()
    entry += (row - self._num_vert1)*self._num_vert1
    entry += col
    return entry


def _find_entry(self, entry):
    num_edges = self.number_of_edges()
    if entry < num_edges:
        row = entry // self._num_vert1
        col = entry % self._num_vert1
        return (row, col)

    entry -= num_edges
    row = entry // self._num_vert2
    col = entry % self._num_vert2
    return (row, col)

def _neighbor_index(self, vertex, neigh):
    # TODO: throw value error if not adjacent
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
    A = self._num_vert2 * np.eye(self._num_vert1, dtype=np.int32)
    B = -np.ones((self._num_vert1, self._num_vert2), dtype=np.int32)
    C = -np.ones((self._num_vert2, self._num_vert1), dtype=np.int32)
    D = self._num_vert1 * np.eye(self._num_vert2, dtype=np.int32)
    return np.block([[A, B],
                     [C, D]])

def CompleteBipartite(num_vert1, num_vert2, multiedges=None, weights=None):
    r"""
    Constructs a complete bipartite graph.

    A complete bipartite graph is a graph whose vertices can be divided 
    into two disjoint and independent sets such that every vertex of 
    the first set is connected to every vertex of the second set, 
    and no vertex within the same set is connected.

    Parameters
    ----------
    num_vert1 : int
        The number of vertices in the first part of the graph.
    num_vert2 : int
        The number of vertices in the second part of the graph.
    multiedges : scipy.sparse.csr_array, optional
        Allows multiple edges between the same pair of vertices. 
        Defaults to None.
    weights : scipy.sparse.csr_array, optional
        Assigns weights to the edges of the graph. 
        Defaults to None.

    Returns
    -------
    :class:`hiperwalk.Graph`
        Returns an instance of a complete bipartite graph. 
        Refer to :ref:`graph_constructors` for more details.

    See Also
    --------
    :ref:`graph_constructors`
        More information on various types of graph constructors.

    Notes
    -----
    The vertices in the first part are labeled from 0 to `num_vert1 - 1`, 
    and the vertices in the second part are labeled from `num_vert1` 
    to `num_vert1 + num_vert2 - 1`.
    """
    if weights is not None or multiedges is not None:
        raise NotImplementedError()

    if num_vert1 <= 0 or num_vert2 <= 0:
        raise ValueError("There must be at least one vertex "
                         + "in each part.")

    # toy graph
    g = Graph(eye(num_vert1 + num_vert2).tocsr())

    # changes attributes
    del g._adj_matrix
    g._adj_matrix = None
    g._num_loops = 0

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
