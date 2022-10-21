import numpy as np
from scipy.sparse import csr_array
from ._coined import *

class Segment(Coined):

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
        # print(adj_matrix.data)
        # print(adj_matrix.indptr)
        # print(adj_matrix.indices)
    
        # initializing
        super().__init__(adj_matrix)
