import numpy as np
from .graph import Graph
from types import MethodType
from scipy.sparse import eye

def adjacent(self, u, v):
    return u != v

def _entry(self, row, col):
    entry = row*(self._num_vert - 1) + col
    if col < row:
        entry += 1

    return entry

def _find_entry(self, entry):
    # row = (entry - 1) % (self._num_vert - 1)
    # col = entry - row*(self._num_vert - 1)
    # if row >= col:
    #     col -= 1

    # return (row, col)
    tail = entry // (self._num_vert - 1)
    head = entry % (self._num_vert - 1)
    if head >= tail:
        head += 1
    return (tail, head)

def _neighbor_index(self, vertex, neigh):
    return neigh - 1 if neigh > vertex else neigh

def neighbors(self, vertex):
    neigh = np.arange(self._num_vert)
    return np.delete(neigh, vertex)

def number_of_vertices(self):
    return self._num_vert

def number_of_edges(self):
    return self._num_vert * (self._num_vert - 1) >> 1

def degree(self, vertex):
    return self._num_vert - 1

def vertex_number(self, vertex):
    vertex = int(vertex)
    if vertex < 0 or vertex >= self._num_vert:
        raise ValueError("Vertex label out of range. " +
                         "Expected integer value from 0 to" +
                         str(self._num_vert - 1))
    return vertex

def adjacency_matrix(self):
    adj_matrix = np.ones((self._num_vert, self._num_vert),
                         dtype=np.int8)
    for i in range(self._num_vert):
        adj_matrix[i, i] = 0

    return adj_matrix

def laplacian_matrix(self):
    lpl_matrix = - np.ones((self._num_vert, self._num_vert),
                           dtype=np.int32)
    for i in range(self._num_vert):
        lpl_matrix[i, i] = self._num_vert - 1

    return lpl_matrix

def Complete(num_vert, multiedges=None, weights=None):
    r"""
    Complete graph constructor.

    A complete graph is one in which every vertex is connected 
    to every other vertex. Each pair of distinct vertices is 
    connected by a unique edge 
    unless multiedges are specified.

    Parameters
    ----------
    num_vert : int
        The number of vertices in the complete graph.
    multiedges : scipy.sparse.csr_array, optional
        Specifies if multiple edges between the same 
        pair of vertices are allowed. Defaults to None.
    weights : scipy.sparse.csr_array, optional
        Assigns weights to the edges of the graph. 
        Defaults to None.

    Returns
    -------
    :class:`hiperwalk.Graph`
        Returns an instance of a complete graph. 
        See :ref:`graph_constructors` for more details.

    See Also
    --------
    :ref:`graph_constructors`
        More information on graph constructors and how they are implemented.
    """
    if weights is not None or multiedges is not None:
        raise NotImplementedError()

    if num_vert <= 0:
        raise ValueError("Expected positive value of vertices."
                         + " Received " + str(num_vert) + "instead.")

    # toy graph
    g = Graph(eye(num_vert).tocsr())

    # changes attributes
    del g._adj_matrix
    g._adj_matrix = None
    g._num_vert = num_vert
    g._num_loops = 0

    g.adjacent = MethodType(adjacent, g)
    g._entry = MethodType(_entry, g)
    g._find_entry = MethodType(_find_entry, g)
    g._neighbor_index = MethodType(_neighbor_index, g)
    g.neighbors = MethodType(neighbors, g)
    g.number_of_vertices = MethodType(number_of_vertices, g)
    g.number_of_edges = MethodType(number_of_edges, g)
    g.degree = MethodType(degree, g)
    g.vertex_number = MethodType(vertex_number, g)
    g.adjacency_matrix = MethodType(adjacency_matrix, g)
    g.laplacian_matrix = MethodType(laplacian_matrix, g)

    return g
