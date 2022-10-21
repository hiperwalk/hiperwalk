import numpy as np
from scipy.sparse import csr_array
from .coined import *

class Segment(Coined):
    r"""
    Class for managing quantum walks on the segment.
    In other words, a finite one-dimensional lattice.

    Notes
    -----
    Since :class:`Segment` is built on top of :class:`Coined`,
    operators and states respect the edge order
    (See :class:`Coined` notes for more details).
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
