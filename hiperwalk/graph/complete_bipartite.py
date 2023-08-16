import numpy as np
from .graph import Graph

class CompleteBipartite(Graph):
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

    def __init__(self, num_vert1, num_vert2):
        self._adj_matrix = None
        self._coloring = None
        if num_vert1 <= 0 or num_vert2 <= 0:
            raise ValueError("There must be at least one vertex "
                             + "in each partition.")

        self._num_vert1 = int(num_vert1)
        self._num_vert2 = int(num_vert2)
        self._num_vert = self._num_vert1 + self._num_vert2

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
        if tail >= self._num_vert or head >= self._num_vert:
            raise ValueError("Vertices should range from 0 to "
                             + str(self._num_vert - 1)
                             + ". But values " + str(tail) + " and " +
                             str(head) + " were received.")

        arc_number = None
        if tail < self._num_vert1 and head >= self._num_vert1:
            head -= self._num_vert1
            arc_number = tail*self._num_vert2 + head
        elif tail >= self._num_vert1 and head < self._num_vert1: 
            tail -= self._num_vert1
            arc_number = tail*self._num_vert1 + head
            arc_number += self.number_of_edges()
        else:
            raise ValueError("Inexistent arc " + str(arc)
                             + ". Tail and head in the same partition.")

        return arc_number

    def arc(self, number):
        if number < 0 or number >= self.number_of_arcs():
            raise ValueError("Inexistent arc. Expected value from 0 to"
                             + str(self.number_of_arcs()) + ".")

        tail = None
        head = None
        num_edges = self.number_of_edges()
        if number < num_edges:
            # tail in V_1 and head in V_2
            tail = number // self._num_vert2
            head = number % self._num_vert2
            head += self._num_vert1
        else:
            # tail in V_2 and head in V_1
            number -= num_edges
            tail = number // self._num_vert1
            tail += self._num_vert1
            head = number % self._num_vert1

        return (tail, head)

    def neighbors(self, vertex):
        if vertex >= self._num_vert1:
            return np.arange(self._num_vert1)
        return np.arange(self._num_vert1, self._num_vert)

    def arcs_with_tail(self, tail):
        if tail < self._num_vert1:
            start = tail*self._num_vert2
            end = start + self._num_vert2
            return np.arange(start, end)

        tail -= self._num_vert1
        start = tail*self._num_vert1 + self.number_of_edges()
        end = start + self._num_vert1
        return np.arange(start, end)

    def number_of_vertices(self):
        return self._num_vert

    def number_of_arcs(self):
        return 2*self._num_vert1*self._num_vert2

    def number_of_edges(self):
        return self._num_vert1*self._num_vert2

    def degree(self, vertex):
        if vertex > 0 and vertex < self._num_vert1:
            return self._num_vert2
        if vertex >= self._num_vert1 and vertex < self._num_vert: 
            return self._num_vert1

        raise ValueError("Vertex out of range. Expected value from 0 to"
                         + str(self._num_vert) + ". But received "
                         + str(vertex) + " instead.")

    def adjacency_matrix(self):
        A = np.zeros((self._num_vert1, self._num_vert1), dtype=np.int8)
        B = np.ones((self._num_vert1, self._num_vert2), dtype=np.int8)
        C = np.ones((self._num_vert2, self._num_vert1), dtype=np.int8)
        D = np.zeros((self._num_vert2, self._num_vert2), dtype=np.int8)
        return np.block([[A, B],
                         [C, D]], dtype=np.int8)
