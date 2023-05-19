import numpy as np
import scipy.sparse
from .graph import Graph

class Lattice(Graph):
    def __init__(self, dimension, periodic=True, diagonal=False):
        r"""
        Two-dimensional lattice.

        The lattice may have boundary conditions or not.
        Its adjacency may be either natural or diagonal.

        Parameters
        ----------
        dimension : int or tuple of int
            Lattice dimension in ``(x_dim, y_dim)`` format.
            If ``dimension`` is an integer, creates a square lattice.


        periodic : bool, default=True
            Whether the lattice has boundary conditions or not.

        diagonal : bool, default=False
            If ``False`` the natural adjacency is used.
            Otherwise, diagonal adjacency is used.
        """

        try:
            x_dim, y_dim = dimension
        except TypeError:
            x_dim = y_dim = dimension

        num_vert = x_dim * y_dim
        num_arcs = (4*num_vert if periodic
                    else 4*num_vert - 2*(x_dim + y_dim))
        num_edges = num_arcs >> 1

        data = np.ones(num_arcs, np.int8)
        indptr = np.zeros(num_vert + 1)
        indices = np.zeros(num_arcs)
        arc_count = 0

        if not periodic and not diagonal:
            for v in range(num_vert):
                #indptr
                indptr[v + 1] = indptr[v] + 4
                x = v % x_dim
                y = v // x_dim
                if x == 0 or x == x_dim - 1:
                    indptr[v + 1] -= 1
                if y == 0 or y == y_dim - 1:
                    indptr[v + 1] -= 1

                #indices
                for coin in [3, 1, 0, 2]: #stay in order
                    head = v + (-1)**(coin % 2) * x_dim**(coin // 2)

                    if (head >= 0 and head < num_vert
                        and not (head % x_dim == 0 and head - v == 1)
                        and not (v % x_dim == 0 and v - head == 1)
                    ):
                        indices[arc_count] = head
                        arc_count += 1

        if not periodic and diagonal:
            raise NotImplementedError

        if periodic:
            indptr = np.arange(0, num_arcs + 1, 4)

            for v in range(num_vert):
                cols = (np.array(
                            [(v + (-1)**(coin % 2) * x_dim**(coin // 2))
                             % num_vert for coin in range(4)])
                        if not diagonal
                        else np.array(
                            [((v % x_dim + (-1)**(coin // 2)) % x_dim
                              + x_dim*(v // x_dim + (-1)**(coin % 2)))
                             % num_vert
                             for coin in range(4)]))
                cols.sort()
                indices[arc_count:arc_count + 4] = cols
                arc_count += 4

            # TODO: use spdiags

        adj_matrix = scipy.sparse.csr_array((data, indices, indptr),
                                            shape=(num_vert, num_vert))
        super().__init__(adj_matrix)
