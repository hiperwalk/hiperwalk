from scipy.sparse import csr_array
from .coined import *

class Cycle(Coined):
    r"""
    Class for managing quantum walk on the cycle.

    Parameters
    ----------
    num_vert : int
        Number of vertices in the cycle.
    """

    def __init__(self, num_vert):
        # Creating adjacency matrix
        # Every vertex is adjacent to two vertices.
        data = np.ones(2*num_vert, dtype=np.int8)
        # upper diagonal
        row_ind = [lin for lin in range(num_vert)
                       for twice in range(2)]
        print(row_ind)
        # lower digonal
        col_ind = [(col-shift) % num_vert for col in range(num_vert)
                             for shift in [-1, 1]]
        print(col_ind)
        adj_matrix = csr_array((data, (row_ind, col_ind)))

        print('hello')
    
        # initializing
        super().__init__(adj_matrix)
