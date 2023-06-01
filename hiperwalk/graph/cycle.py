from scipy.sparse import csr_array
from .graph import *

class Cycle(Graph):
    r"""
    Cycle Graph.

    Parameters
    ----------
    num_vert : int
        Number of vertices in the cycle.

    Notes
    -----
    The cycle may be interpreted as being embedded on the line
    with cyclic boundary condition.


    .. todo::
        
        update below

    The edge order respects the default vertex-coin notation.
    In other words, 0 corresponds to the coin pointing rightwards,
    and 1 to the coin pointing leftwards.
    Therefore, the arcs are sorted with respect to this order
    (vertex has precedence over direction and
    right has precedence over left):

    .. math::
        \begin{align*}
            \ket{(v, c)} = \ket{2v + c}
        \end{align*}

    where :math:`v \in V` and :math:`c \in \{0, 1\}`.
    Figure 1 illustrates the arcs of a 3 vertices cycle.

    .. graphviz:: ../../graphviz/coined-cycle-edges-labels.dot
        :align: center
        :layout: neato
        :caption: Figure 1.

    """

    def __init__(self, num_vert):
        # Creating adjacency matrix
        # Every vertex is adjacent to two vertices.
        data = np.ones(2*num_vert, dtype=np.int8)
        # upper diagonal
        row_ind = [lin for lin in range(num_vert)
                       for twice in range(2)]
        # lower digonal
        col_ind = [(col-shift) % num_vert for col in range(num_vert)
                             for shift in [-1, 1]]
        adj_matrix = csr_array((data, (row_ind, col_ind)))
    
        # initializing
        super().__init__(adj_matrix)

    def embeddable(self):
        return True

    def default_coin(self):
        r"""
        Returns the default coin name.
        """
        return 'hadamard'

    def arc_label(self, tail, head):
        num_vert = self.number_of_vertices()
        arc = (2*tail if (head - tail == 1
                          or tail - head == num_vert - 1)
               else 2*tail + 1)
        return arc

    def arc(self, label):
        tail = label//2
        remainder = label % 2
        head = (tail + (-1)**remainder) % self.number_of_vertices()
        return (tail, head)

    def next_arc(self, arc):
        # implemented only if is embeddable
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
        # implemented only if is embeddable
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
