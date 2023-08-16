import numpy as np
from .graph import Graph

class Complete(Graph):
    r"""
    Complete graph.

    The graph on which any vertex is connected to
    every other vertex.

    Parameters
    ----------
    num_vert : int
        Number of vertices in the complete graph.
    """

    def __init__(self, num_vert):
        # the adjacency matrix structure is easy.
        # It is not stored to save space
        if num_vert <= 0:
            raise ValueError("Expected positive value of vertices."
                             + " Received " + str(num_vert) + "instead.")
        self._num_vert = int(num_vert)
        self._adj_matrix = None
        self._coloring = None

    def arc_number(self, *args):
        arc = (args[0], args[1]) if len(args) == 2 else args[0]

        if not hasattr(arc, '__iter__'):
            num_arcs = self.number_of_arcs()
            if arc < 0 and arc >= num_arcs:
                raise ValueError("Arc value out of range. "
                                 + "Expected arc value from 0 to "
                                 + str(num_arcs - 1))
            return int(arc)

        tail, head = arc
        arc_number = (self._num_vert - 1) * tail + head
        return arc_number if tail > head else arc_number - 1

    def arc(self, number):
        tail = number // (self._num_vert - 1)
        head = number % (self._num_vert - 1)
        if head >= tail:
            head += 1
        return (tail, head)

    def neighbors(self, vertex):
        neigh = np.arange(self._num_vert)
        return np.delete(neigh, vertex)

    def arcs_with_tail(self, tail):
        return np.arange((self._num_vert - 1)*tail,
                         (self._num_vert - 1)*(tail + 1))

    def number_of_vertices(self):
        return self._num_vert

    def number_of_arcs(self):
        return self._num_vert * (self._num_vert - 1)

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
