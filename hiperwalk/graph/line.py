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
    .. todo::

        update


    Since :class:`Segment` is built on top of :class:`Graph`,
    operators and states respect the edge order
    (See :class:`Graph` notes for more details).
    As a consequence, for any vertex :math:`v \in V`,
    the state :math:`\ket{2v - d}` for :math:`d \in \{0, 1\}`
    corresponds to the walker being on vertex :math:`v` and the
    coin pointing rightwars (:math:`d = 0`) or leftwards (:math:`d = 1`).
    For example, the edges labels of the 4-vertices :class:`Segment`
    are represented in Figure 1.
    
    .. graphviz:: ../../graphviz/coined-segment-edges-labels.dot
        :align: center
        :layout: neato
        :caption: Figure 1.
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

        return tail*2 if diff == 1 else tail*2 - 1 

    def arc(self, label):
        tail = (label + 1)//2
        head = tail + (-1)**(label % 2)
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
            if arc == 1:
                return 0
            num_arcs = self.number_of_arcs() 
            if arc == num_arcs - 2:
                return num_arcs - 1
            return arc + 2 if arc % 2 == 0 else arc - 2

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
            if arc == 0:
                return 1
            num_arcs = self.number_of_arcs() 
            if arc == num_arcs - 1:
                return num_arcs - 2
            return arc - 2 if arc % 2 == 0 else arc + 2
