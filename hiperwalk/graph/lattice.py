import numpy as np
import scipy.sparse
from .graph import Graph
from warnings import warn

class Lattice(Graph):
    def __init__(self, dimensions, periodic=True, diagonal=False):
        r"""
        Two-dimensionsal lattice.

        The lattice may have boundary conditions or not.
        Its adjacency may be either natural or diagonal.

        Parameters
        ----------
        dimensions : int or tuple of int
            Lattice dimensions in ``(x_dim, y_dim)`` format.
            If ``dimensions`` is an integer, creates a square lattice.


        periodic : bool, default=True
            Whether the lattice has boundary conditions or not.

        diagonal : bool, default=False
            If ``False`` the natural adjacency is used.
            Otherwise, diagonal adjacency is used.
        """

        try:
            x_dim, y_dim = dimensions
        except TypeError:
            x_dim = y_dim = dimensions

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
                            [v - v % x_dim
                             + (v % x_dim + (-1)**(coin % 2)) % x_dim
                             if coin < 2 else
                             (v + x_dim*(-1)**(coin % 2)) % num_vert
                             for coin in range(4)])
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
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.periodic = periodic
        self.diagonal = diagonal

    def embeddable(self):
        return True

    def vertex_coordinates(self, label):
        return (label % self.x_dim, label // self.x_dim)


    def vertex_label(self, x, y):
        return (x + self.x_dim*y) % self.number_of_vertices()

    def arc_direction(self, arc):
        r"""
        Notes
        -----
        Does not check if arc exists.
        """
        # dealing with coordinates
        try:
            tail, head = arc
            if not hasattr(tail, '__iter__'):
                tail = self.vertex_coordinates(tail)
            if not hasattr(head, '__iter__'):
                head = self.vertex_coordinates(head)
        except TypeError:
            tail, head = self.arc(arc)

        if self.diagonal:
            x_diff = head[0] - tail[0]
            y_diff = head[1] - tail[1]
            x = 0 if (x_diff == 1 or x_diff == -self.x_dim + 1) else 1
            y = 0 if (y_diff == 1 or y_diff == -self.y_dim + 1) else 1
            return (x << 1) + y

        y = tail[1] != head[1]
        x = ((tail[1] - head[1]) % self.y_dim == 1
             if y else
             (tail[0] - head[0]) % self.x_dim == 1)

        return (y << 1) + x


    def arc_label(self, tail, head):
        try:
            tail = self.vertex_label(tail[0], tail[1])
            head = self.vertex_label(head[0], head[1])
        except TypeError:
            pass
        except IndexError:
            pass

        if not self.periodic:
            raise NotImplementedError
        if not self.diagonal:
            raise NotImplementedError

        return 4*tail + self.arc_direction((tail, head)) 

    def arc(self, label, coordinates=True):
        if not self.periodic:
            raise NotImplementedError
        if not self.diagonal:
            raise NotImplementedError

        tail = label // 4
        coin = label % 4
        num_vert = self.number_of_vertices()
        x_dim = self.x_dim
        head = ((tail % x_dim + (-1)**(coin // 2)) % x_dim
                 + x_dim*(tail // x_dim + (-1)**(coin % 2))) % num_vert
        arc = (tail, head)

        if not coordinates:
            return arc
        return (self.vertex_coordinates(arc[0]),
                self.vertex_coordinates(arc[1]))

    def neighbors(self, vertex):
        iterable = hasattr(vertex, '__iter__')
        if iterable:
            vertex = self.vertex_label(vertex[0], vertex[1])

        neigh = super().neighbors(vertex)

        if iterable:
            return list(map(self.vertex_coordinates, neigh))
        return neigh

    def arcs_with_tail(self, tail):
        try:
            tail = self.vertex_label(tail[0], tail[1])
        except TypeError:
            pass

        return super().arcs_with_tail(tail)

    def degree(self, vertex):
        try:
            vertex = self.vertex_label(vertex[0], vertex[1])
        except TypeError:
            pass

        return super().degree(vertex)

    def next_arc(self, arc):
        if not self.periodic:
            raise NotImplementedError

        try:
            tail, head = arc
            iterable = hasattr(tail, '__iter__')
            
            if not iterable:
                tail = self.vertex_coordinates(tail)
                head = self.vertex_coordinates(head)

            # get direction
            direction = self.arc_direction(arc)
            if self.diagonal:
                x = direction // 2
                y = direction % 2
                tail = head
                head = ((head[0] + (-1)**x) % self.x_dim,
                        (head[1] + (-1)**y) % self.y_dim)

                if iterable:
                    return (tail, head)
                return (self.vertex_label(tail[0], tail[1]),
                        self.vertex_label(head[0], head[1]))
            else:
                raise NotImplementedError

        except TypeError:
            # arc label
            raise NotImplementedError

    def previous_arc(self, arc):
        if not self.periodic:
            raise NotImplementedError

        arc_iterable = hasattr(arc, '__iter__')
        if arc_iterable:
            tail, head = arc
        else:
            tail, head = self.arc(arc)
        vertex_iterable = hasattr(tail, '__iter__')
        if not vertex_iterable:
            tail = self.vertex_coordinates(tail)
            head = self.vertex_coordinates(head)

        direction = self.arc_direction(arc)
        if self.diagonal:
            x = direction // 2
            y = direction % 2
            head = tail
            tail = ((tail[0] - (-1)**x) % self.x_dim,
                    (tail[1] - (-1)**y) % self.y_dim)
        else:
            raise NotImplementedError

        if not arc_iterable:
            return self.arc_label(tail, head)
        if vertex_iterable:
            return (tail, head)
        return (self.vertex_label(tail[0], tail[1]),
                self.vertex_label(head[0], head[1]))

    def dimensions(self):
        return (self.x_dim, self.y_dim)

    def get_central_vertex(self):
        warn('`get_central_vertex` is deprecated. '
             + 'Use `central_vertex` instead.',
             DeprecationWarning)
        return self.central_vertex()

    def central_vertex(self):
        if self.x_dim % 2 != 1 or self.y_dim % 2 != 1:
            raise ValueError(
                "One of lattice dimensions is even. "
                + "Hence it does not have a single central vertex."
            )
        return np.array([self.x_dim // 2, self.y_dim // 2])
