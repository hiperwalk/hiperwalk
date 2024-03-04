from scipy.sparse import csr_array
from .graph import Graph
from .square_lattice import SquareLattice

class Cycle(Graph):
    r"""
    Cycle graph.

    Parameters
    ----------
    num_vert : int
        Number of vertices in the cycle.

    Notes
    -----
    The cycle can be interpreted as being embedded on the line
    with a cyclic boundary condition.
    In this context,
    we assign the direction ``0`` to the right and ``1`` to the left.
    This assignment alters the order of the arcs.
    Any arc with a tail denoted by :math:`v`
    has the numerical label :math:`2v` if it points to the right,
    and the numerical label :math:`2v + 1` if it points to the left.
    Figure 1 illustrates the arc numbers of a cycle with 3 vertices.

    .. graphviz:: ../../graphviz/cycle-arcs.dot
        :align: center
        :layout: neato
        :caption: Figure 1.

    """

    def __init__(self, num_vert):
        # Creating adjacency matrix
        # Every vertex is adjacent to two vertices.
        data = np.ones(2*num_vert, dtype=np.int8)
        indptr = 2*np.arange(num_vert + 1)

        indices = np.zeros(len(data))
        right = 1
        left = num_vert - 1
        for i in range(num_vert):
            indices[2*i] = right
            indices[2*i + 1] = left
            right = (right + 1) % num_vert
            left = (left + 1) % num_vert

        adj_matrix = csr_array((data, indices, indptr))
        self._adj_matrix = adj_matrix
    
        # initializing
        #super().__init__(adj_matrix)

    def arc_number(self, arc):
        if not hasattr(arc, '__iter__'):
            return super().arc_number(arc)

        tail, head = arc
        num_vert = self.number_of_vertices()
        arc = (2*tail if (head - tail == 1
                          or tail - head == num_vert - 1)
               else 2*tail + 1)
        return arc

    def arc(self, number):
        tail = number//2
        remainder = number % 2
        head = (tail + (-1)**remainder) % self.number_of_vertices()
        return (tail, head)

    def next_arc(self, arc):
        try:
            tail, head = arc
            num_vert = self.number_of_vertices()
            if (head - tail) % num_vert == 1:
                # clockwise
                return ((tail + 1) % num_vert, (head + 1) % num_vert)
            elif (tail - head) % num_vert == 1:
                # anticlockwise
                return ((tail - 1) % num_vert, (head - 1) % num_vert)
            else:
                raise ValueError('Invalid arc.')
        except TypeError:
            # could not unpack. Arc label passed
            num_arcs = self.number_of_arcs()
            return ((arc + 2) % num_arcs if arc % 2 == 0
                    else (arc - 2) % num_arcs)

    def previous_arc(self, arc):
        try:
            tail, head = arc
            num_vert = self.number_of_vertices()
            if (head - tail) % num_vert == 1:
                # previous clockwise
                return ((tail - 1) % num_vert, (head - 1) % num_vert)
            elif (tail - head) % num_vert == 1:
                # previous anticlockwise
                return ((tail + 1) % num_vert, (head + 1) % num_vert)
            else:
                raise ValueError('Invalid arc.')
        except TypeError:
            # could not unpack. Arc label passed
            num_arcs = self.number_of_arcs()
            return ((arc - 2) % num_arcs if arc % 2 == 0
                    else (arc + 2) % num_arcs)
