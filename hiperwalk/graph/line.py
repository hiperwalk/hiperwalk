import numpy as np
from scipy.sparse import csr_array
from sys import path as sys_path
from .graph import Graph

class Line(Graph):
    r"""
    Finite Line Graph (Path Graph).

    Parameters
    ----------
    num_vert : int
        Number of vertices on the line.

    Notes
    -----
    On the Line,
    we can assign directions to the arcs.
    An arc pointing rightward has direction 0 --
    i.e., (1, 2), and
    an arc pointing leftward has direction 1 --
    i.e., (2, 1).

    We use the directions of the arcs to define their order.
    Thus, for a vertex :math:`v \in V`,
    the arcs :math:`(v, v + 1)` and :math:`(v, v - 1)`
    have labels :math:`a_0` and :math:`a_1` (respectively)
    such that  :math:`a_0 < a_1`.
    This order is aligned with the description of coined quantum walks on the line in Reference [1]_.
    The only exceptions to this rule are the extreme vertices
    0 and :math:`|V| - 1` because
    they only have one arc.
    Aside from these exceptions,
    for any two vertices :math:`v_1 < v_2`,
    any arc with tail :math:`v_1` has a label smaller than
    the label of any arc with tail :math:`v_2`.

    For example, Figure 1 depicts the labels of the arcs of a path graph
    with 4 vertices.
    
    .. graphviz:: ../../graphviz/line-arcs.dot
        :align: center
        :layout: neato
        :caption: Figure 1.

    References
    ----------
    .. [1] R. Portugal. "Quantum walks and search algorithms",
        2nd edition, Springer, 2018.
    """

    def __init__(self, num_vert):
        # Creating adjacency matrix
        # The end vertices are only adjacent to 1 vertex.
        # Every other vertex is adjacent to two vertices.
        data = np.ones(2*(num_vert-1), dtype=np.int8)
        # upper diagonal
        row_ind = [lin+shift for shift in range(2)
                             for lin in range(num_vert - 1)]
        # lower digonal
        col_ind = [col+shift for shift in [1, 0]
                             for col in range(num_vert - 1)]
        adj_matrix = csr_array((data, (row_ind, col_ind)))
    
        # initializing
        super().__init__(adj_matrix)

    def embeddable(self):
        return True

    def default_coin(self):
        r"""
        Returns the default coin name.

        The default coin for the coined quantum walk on the
        segment is ``'hadamard'``.
        """
        return 'hadamard'

    def arc_label(self, tail, head):
        diff = head - tail
        if diff != 1 and diff != -1:
            raise ValueError('Invalid arc.')

        num_vert = self.number_of_vertices()
        return (2*tail - 1 if (diff == 1 and tail != 0
                               or head == num_vert - 1)
                else tail*2)

    def arc(self, label):
        label += 1
        tail = label//2
        num_vert = self.number_of_vertices()
        head = (tail + (-1)**(label % 2)
                if tail != 0 and tail != num_vert - 1
                else tail - (-1)**(label % 2))
        return (tail, head)

    def next_arc(self, arc):
        # implemented only if is embeddable
        try:
            tail, head = arc
            diff = head - tail
            if diff != 1 and diff != -1:
                raise ValueError('Invalid arc')

            if head == 0 or head == self.number_of_vertices() - 1:
                return (head, tail)
            
            return ((tail + 1, head + 1) if diff == 1
                    else (tail - 1, head - 1))
        except TypeError:
            if arc == 0:
                return 1
            num_arcs = self.number_of_arcs() 
            if arc == num_arcs - 1:
                return num_arcs - 2
            return arc + 2 if arc % 2 == 1 else arc - 2

    def previous_arc(self, arc):
        # implemented only if is embeddable
        try:
            tail, head = arc
            diff = head - tail
            if diff != 1 and diff != -1:
                raise ValueError('Invalid arc')

            if tail == 0 or tail == self.number_of_vertices() - 1:
                return (head, tail)
            
            return ((tail - 1, head - 1) if diff == 1
                    else (tail + 1, head + 1))
        except TypeError:
            num_arcs = self.number_of_arcs()
            if arc < 0 or arc >= num_arcs:
                raise ValueError('Invalid arc')

            arc = arc + 2 if arc % 2 == 0 else arc - 2
            if arc < 0:
                arc += 1
            elif arc >= num_arcs:
                arc -= 1

            return arc
