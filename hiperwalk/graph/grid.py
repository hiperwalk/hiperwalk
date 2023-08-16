import numpy as np
import scipy.sparse
from .lattice import Lattice
from warnings import warn

class Grid(Lattice):
    r"""
    Two-dimensionsal grid.

    The grid can be designed with either
    cyclic boundary conditions or borders.
    Moreover, the grid's representation can be either
    *natural* or *diagonal*.
    In the *natural* representation,
    neighboring vertices lie along the X and Y axes,
    while in the *diagonal* representation,
    they lie along the diagonals.

    Parameters
    ----------
    dimensions : int or tuple of int
        Grid dimensions in ``(x_dim, y_dim)`` format.
        If ``dimensions`` is an integer, creates a square grid.

    periodic : bool, default=True
        ``True`` if the grid has cyclic boundary conditions,
        ``False`` if it has borders.

    diagonal : bool, default=False
        ``True`` if the grid has the diagonal representation,
        ``False`` if it has the natural representation.

    Notes
    -----
    The order of the arcs is determined according to
    the order of the vertices.
    The order of the vertices is defined as follows:
    if :math:`(x_1, y_1)` and :math:`(x_2, y_2)` are two valid vertices,
    we say that :math:`(x_1, y_1) < (x_2, y_2)` if :math:`y_1 < y_2` or
    if :math:`y_1 = y_2` and :math:`x_1 < x_2`.
    The order of the arcs also depends on the grid representation
    (natural or diagonal).

    Natural Grid:
        In the natural grid,
        the directions are described as follows:

        * 00 = 0: right;
        * 01 = 1: left;
        * 10 = 2: up;
        * 11 = 3: down.
        
        The most significant bit corresponds to the axis:
        0 represents the X-axis and 1 represents the Y-axis.
        The least significant bit indicates the direction of
        movement along the given axis, with 0 signifying
        forward movement and 1 signifying backward movement.

        The order of arcs corresponds with the order of vertices
        and their respective directions. For example,
        consider a vertex :math:`(x, y)`. Then,
        :math:`(x \pm 1, y)` and :math:`(x, y \pm 1)`
        are adjacent vertices.
        The order of these arcs is

        .. math::
            ((x, y), (x + 1, y)) &< ((x, y), (x - 1, y)) \\
                                 &< ((x, y), (x, y + 1)) \\
                                 &< ((x, y), (x, y - 1)).

        The directions are depicted in :ref:`fig-natural-dir`.

        .. graphviz:: ../../graphviz/grid/natural-directions.dot
            :align: center
            :layout: neato
            :name: fig-natural-dir
            :caption: Figure: Directions in the natural representation.

        For example, the labels of the arcs for
        the :math:`3 \times 3`-grid with periodic boundary
        conditions are depicted in :ref:`fig-periodic-natural-grid`.

        .. graphviz:: ../../graphviz/grid/periodic-natural.dot
            :align: center
            :layout: neato
            :name: fig-periodic-natural-grid
            :caption: Figure: 3x3-grid with cyclic boundary conditions.

        In the case of a natural grid with borders, the labels of the
        arcs maintain the same sequence but with some modifications
        due to the presence of vertices with degrees 2 and 3.
        Figure :ref:fig-bounded-natural-Grid provides an illustration
        of a bounded natural grid.

        .. graphviz:: ../../graphviz/grid/bounded-natural.dot
            :align: center
            :layout: neato
            :name: fig-bounded-natural-grid
            :caption: Figure: Bounded natural 3x3-grid.

    Diagonal Grid:
        In the diagonal grid,
        the directions are described as follows:

        * 00 = 0: right, up;
        * 01 = 1: right, down;
        * 10 = 2: left, up;
        * 11 = 3: left, down.

        Each binary value indicates the direction of movement,
        with 0 representing forward motion and 1 representing
        backward motion. The most significant bit corresponds to
        movement along the X-axis, while the least significant bit
        corresponds to movement along the Y-axis.

        The order of arcs corresponds with the order of vertices
        and their respective directions. For example,
        consider a vertex :math:`(x, y)`. Then,
        :math:`(x \pm 1, y)` and :math:`(x, y \pm 1)`
        are adjacent vertices.
        The order of these arcs is
        
        .. math::
            ((x, y), (x + 1, y + 1)) &< ((x, y), (x + 1, y - 1)) \\
                                     &< ((x, y), (x - 1, y + 1)) \\
                                     &< ((x, y), (x - 1, y - 1)).

        The directions are depicted in :ref:`fig-diagonal-dir`.

        .. graphviz:: ../../graphviz/grid/diagonal-directions.dot
            :align: center
            :layout: neato
            :name: fig-diagonal-dir
            :caption: Figure: Directions in the diagonal representation.

        For example, the labels of the arcs for
        the :math:`3 \times 3`-grid with periodic boundary
        conditions are depicted in :ref:`fig-periodic-diagonal-grid`

        .. graphviz:: ../../graphviz/grid/periodic-diagonal.dot
            :align: center
            :layout: neato
            :name: fig-periodic-diagonal-grid
            :caption: Figure: Periodic diagonal 3x3-grid.

        In the case of a diagonal grid with borders, the labels of the
        arcs maintain the same sequence but with some modifications.
        Figure :ref:`fig-bounded-diagonal-Grid` provides
        an illustration of a bounded diagonal grid.

        .. graphviz:: ../../graphviz/grid/bounded-diagonal.dot
            :align: center
            :layout: neato
            :name: fig-bounded-diagonal-grid
            :caption: Figure: Bounded 3x3-grid in the diagonal representation.

        Note that in this context,
        there exist two independent subgrids.
        In other words, a vertex in one subgrid is not accessible from
        a vertex in the other subgrid. This situation also arises
        if the diagonal grid has periodic boundary conditions and both
        dimensions are even.
        Figure :ref:`fig-even-dim-diagonal` illustrates
        an example of this case.

        .. graphviz:: ../../graphviz/grid/even-dim-diagonal.dot
            :align: center
            :layout: neato
            :name: fig-even-dim-diagonal
            :caption: Figure: 4x4-grid with cyclic boundary conditions.
    """
    def __init__(self, dimensions, periodic=True, diagonal=False):
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

    def vertex_coordinates(self, label):
        r"""
        Returns vertex (x, y)-coordinates given its label.

        Parameters
        ----------
        label : int
            Vertex label.

        Returns
        -------
        x : int
            Vertex X-coordinate.
        y : int
            Vertex Y-coordinate.

        See Also
        --------
        vertex_number
        """
        return (label % self.x_dim, label // self.x_dim)


    def vertex_number(self, vertex):
        r"""
        Returns vertex number given any vertex representation.

        By invoking this method,
        the vertex number is returned regardless of its representation.
        The representation may be the vertex number itself or
        the vertex coordinates.

        Parameters
        ----------
        vertex: int or tuple of int
            * int :
                The vertex number.
            * (int, int):
                The vertex coordinates in ``(x, y)`` format.

        Returns
        -------
        int
            Vertex label.

        See Also
        --------
        vertex_coordinates
        """
        if not hasattr(vertex, '__iter__'):
            return super().vertex_number(vertex)

        x, y = vertex
        return (x + self.x_dim*y) % self.number_of_vertices()

    def arc_direction(self, arc):
        r"""
        Return arc direction.

        Parameters
        ----------
        arc
            Any of the following notations are acceptable.
            
            * ((int, int), (int, int))
                Arc notation with vertices' coordinates.
            * (int, int)
                Arc notation with vertices' labels.
            * int
                Arc label.

        Returns
        -------
        int
            If natural (not diagonal) grid:
                * 0: right
                * 1: left
                * 2: up
                * 3: down

            If diagonal grid:
                * 0: right, up
                * 1: right, down
                * 2: left, up
                * 3: left, down

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


    def arc_number(self, *args):
        arc = (args[0], args[1]) if len(args) == 2 else args[0]

        if not hasattr(arc, '__iter__'):
            return super().arc_number(arc)

        tail, head = arc
        tail = self.vertex_number(tail)
        head = self.vertex_number(head)

        if self._adj_matrix[tail, head] == 0:
            raise ValueError('Inexistent arc ' + str((tail, head)) + '.')

        if self.periodic:
            return 4*tail + self.arc_direction((tail, head))

        label = self._adj_matrix.indptr[tail]
        direction = self.arc_direction((tail, head))
        if self.diagonal:
            raise NotImplementedError

        sub_x = (1 if ((tail % self.x_dim == 0
                        or tail % self.x_dim == self.x_dim - 1)
                       and direction > 0)
                 else 0)
            
        sub_y = (1 if ((tail // self.x_dim == 0
                        or tail // self.x_dim == self.x_dim - 1)
                       and direction > 2)
                 else 0)
        return label + direction - sub_x - sub_y

    def arc(self, number, coordinates=True):
        r"""
        Arc in arc notation.

        Given the numerical arc label,
        returns the arc in the ``(tail, head)`` notation.

        Parameters
        ----------
        number : int
            Numerical arc label.

        coordinates : bool, default=True
            Whether the vertices are returned as coordinates or as
            a single number.

        Returns
        -------
        (tail, head)
            There are two possible formats for the vertices
            ``tail`` and ``head``.

            (vertex_x, vertex_y) : (int, int)
                If ``coordinates=True``.
            number : int
                If ``coordinates=False``.
        """
        if not self.periodic and self.diagonal:
            raise NotImplementedError

        if self.periodic:
            tail = number // 4
            coin = number % 4
            num_vert = self.number_of_vertices()
            x_dim = self.x_dim
            if self.diagonal:
                head = (((tail % x_dim + (-1)**(coin // 2)) % x_dim
                          + x_dim*(tail // x_dim + (-1)**(coin % 2)))
                        % num_vert)
            else:
                head = (tail - tail % x_dim
                        + (tail % x_dim + (-1)**(coin % 2)) % x_dim
                        if coin < 2 else
                        (tail + x_dim*(-1)**(coin % 2)) % num_vert)

        else:
            # not diagonal
            tail, _ = super().arc(number)
            diff = number - self._adj_matrix.indptr[tail]
            num_vert = self.number_of_vertices()

            for coin in range(4):
                head = tail + (-1)**(coin % 2) * self.x_dim**(coin // 2)

                if (head >= 0 and head < num_vert
                    and not (head % self.x_dim == 0 and head - tail == 1)
                    and not (tail % self.x_dim == 0 and tail - head == 1)
                ):
                    if diff == 0:
                        break
                    diff -= 1

        arc = (tail, head)

        if not coordinates:
            return arc
        return (self.vertex_coordinates(arc[0]),
                self.vertex_coordinates(arc[1]))

    def neighbors(self, vertex):
        iterable = hasattr(vertex, '__iter__')
        if iterable:
            vertex = self.vertex_number(vertex)

        neigh = super().neighbors(vertex)

        if iterable:
            return list(map(self.vertex_coordinates, neigh))
        return neigh

    def arcs_with_tail(self, tail):
        try:
            tail = self.vertex_number(tail)
        except TypeError:
            pass

        return super().arcs_with_tail(tail)

    def degree(self, vertex):
        try:
            vertex = self.vertex_number(vertex)
        except TypeError:
            pass

        return super().degree(vertex)

    def next_arc(self, arc):
        try:
            tail, head = arc
        except TypeError:
            tail, head = self.arc(arc)

        iterable = hasattr(tail, '__iter__')
        
        if not iterable:
            tail = self.vertex_coordinates(tail)
            head = self.vertex_coordinates(head)

        # get direction
        direction = self.arc_direction(arc)

        if self.diagonal:
            if not self.periodic:
                raise NotImplementedError

            tail = head
            x = direction // 2
            y = direction % 2
            head = ((head[0] + (-1)**x) % self.x_dim,
                    (head[1] + (-1)**y) % self.y_dim)
        else:
            y_axis = direction // 2
            exp = direction % 2
            if self.periodic:
                tail = head
                head = ((head[0], (head[1] + (-1)**exp) % self.x_dim)
                        if y_axis else
                        ((head[0] + (-1)**exp) % self.y_dim, head[1]))
            else:
                new_head = ((head[0], head[1] + (-1)**exp) if y_axis
                            else (head[0] + (-1)**exp, head[1]))

                if (new_head[0] < 0 or new_head[0] >= self.x_dim
                    or new_head[1] < 0 or new_head[1] >= self.y_dim
                ):
                    # out of bounds. Rebound
                    new_head = tail

                tail = head
                head = new_head

        if not iterable:
            tail = self.vertex_number(tail)
            head = self.vertex_number(head)
        return (tail, head)

    def previous_arc(self, arc):
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
            if not self.periodic:
                raise NotImplementedError

            x = direction // 2
            y = direction % 2
            head = tail
            tail = ((tail[0] - (-1)**x) % self.x_dim,
                    (tail[1] - (-1)**y) % self.y_dim)
        else:
            y_axis = direction // 2
            exp = direction % 2
            if self.periodic:
                head = tail
                tail = ((tail[0], (tail[1] - (-1)**exp) % self.x_dim)
                        if y_axis else
                        ((tail[0] - (-1)**exp) % self.y_dim, tail[1]))
            else:
                new_tail = ((tail[0], tail[1] - (-1)**exp) if y_axis
                            else (tail[0] - (-1)**exp, tail[1]))

                if (new_tail[0] < 0 or new_tail[0] >= self.x_dim
                    or new_tail[1] < 0 or new_tail[1] >= self.y_dim
                ):
                    # out of bounds. Rebound
                    new_tail = head

                head = tail
                tail = new_tail

        if not vertex_iterable:
            tail = self.vertex_number(tail),
            head = self.vertex_number(head)

        if not arc_iterable:
            return self.arc_number((tail, head))
        return (tail, head)

    def dimensions(self):
        r"""
        Grid dimensions.

        Returns
        -------
        x_dim : int
            Dimension alongside de X axis.
        y_dim : int
            Dimension alongside de Y axis.
        """
        return (self.x_dim, self.y_dim)
