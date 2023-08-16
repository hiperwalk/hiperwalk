import numpy as np
from scipy.sparse import csr_array
from sys import path as sys_path
from .lattice import Lattice

class Line(Lattice):
    r"""
    Finite line graph (path graph).

    Parameters
    ----------
    num_vert : int
        The number of vertices on the line.

    Notes
    -----
    In the :obj:`Line` class, directions can be assigned to the arcs. 
    An arc pointing to the right has direction 0 (e.g., (1, 2)), 
    and an arc pointing to the left has direction 1 (e.g., (2, 1)).

    The order of the arcs is determined by their direction. 
    Thus, for a vertex :math:`v \in V`, 
    the arcs :math:`(v, v + 1)` and :math:`(v, v - 1)` 
    have labels :math:`a_0` and :math:`a_1` respectively, 
    with :math:`a_0 < a_1`. 
    The only exceptions to this rule are the extreme vertices 0 
    and :math:`|V| - 1`, as they have outdegree 1. 

    Apart from these exceptions, 
    for any two vertices :math:`v_1 < v_2`, 
    any arc with tail :math:`v_1` will have a label smaller than the 
    label of any arc with tail :math:`v_2`.

    For instance, Figure 1 illustrates the labels of the arcs of a path graph with 4 vertices.
    
    .. graphviz:: ../../graphviz/line-arcs.dot
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

    def arc_number(self, *args):
        arc = (args[0], args[1]) if len(args) == 2 else args[0]

        if not hasattr(arc, '__iter__'):
            return super().arc_number(arc)

        tail, head = arc
        diff = head - tail
        if diff != 1 and diff != -1:
            raise ValueError('Invalid arc.')

        num_vert = self.number_of_vertices()
        return (2*tail - 1 if (diff == 1 and tail != 0
                               or tail == num_vert - 1)
                else tail*2)

    def arc(self, number):
        number += 1
        tail = number//2
        num_vert = self.number_of_vertices()
        head = (tail + (-1)**(number % 2)
                if tail != 0 and tail != num_vert - 1
                else tail - (-1)**(number % 2))
        return (tail, head)

    def next_arc(self, arc):
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
