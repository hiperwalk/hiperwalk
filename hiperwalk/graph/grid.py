from numpy import array as np_array
from .integer_lattice import IntegerLattice
from types import MethodType

def Grid(dim, periodic=True, diagonal=False,
         multiedges=None, weights=None, copy=False):
    r"""
    Two-dimensionsal grid constructor.

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
    dim : int or tuple of int
        Grid dimensions in ``(dim[0], dim[1])`` format,
        where ``dim[0]`` is the number of vertices in the X-axis,
        and ``dim[1]`` is the number of vertices in the Y-axis.
        If ``dim`` is an integer, creates a square grid.

    periodic : bool, default=True
        ``True`` if the grid has cyclic boundary conditions,
        ``False`` if it has borders.

    diagonal : bool, default=False
        ``True`` if the grid has the diagonal representation,
        ``False`` if it has the natural representation.

    multiedges, weights: matrix or dict, default=None
        See :ref:`graph_constructors`.

    copy : bool, default=False
        See :ref:`graph_constructors`.

    Returns
    -------
    :class:`hiperwalk.Graph`
        See :ref:`graph_constructors` for details.

    See Also
    --------
    :ref:`graph_constructors`.

    Notes
    -----
    The **order of neighbors** depends on the grid.

    .. testsetup::

        import hiperwalk as hpw

    Natural Grid:
        The natural grid is created when ``diagonal=False``.
        The neighbors are given in the following order.

        * 00 = 0: right;
        * 01 = 1: left;
        * 10 = 2: up;
        * 11 = 3: down.
        
        The most significant bit corresponds to the axis:
        0 represents the X-axis and 1 represents the Y-axis.
        The least significant bit indicates the direction
        along the given axis, with 0 signifying
        forward and 1 signifying backward.

        Consider a vertex :math:`(x, y)`. Then,
        :math:`(x \pm 1, y)` and :math:`(x, y \pm 1)`
        are adjacent vertices.
        The order of neighbors is depicted in
        :ref:`fig-natural-grid-order`.

        .. graphviz:: ../../graphviz/grid/natural-grid-neigh-order.dot
            :align: center
            :layout: neato
            :name: fig-natural-grid-order
            :caption: Figure: The order of neighbors in the natural grid.

        For example,
        consider the :math:`3 \times 3` periodic natural grid
        (:ref:`fig-3x3-natural-grid`).

        .. graphviz:: ../../graphviz/grid/periodic-natural.dot
            :align: center
            :layout: neato
            :name: fig-3x3-natural-grid
            :caption: Figure: Periodic natural 3x3-grid.

        The neighbors of :math:`(0, 0)` and :math:`(1, 1)` with
        respect to the order of neighbors are

        .. doctest::

            >>> nat = hpw.Grid(3, diagonal=False, periodic=True)
            >>> neigh = nat.neighbors((0, 0))
            >>> [tuple(nat.vertex_coordinates(v)) for v in neigh]
            [(1, 0), (2, 0), (0, 1), (0, 2)]
            >>> 
            >>> neigh = nat.neighbors((1, 1))
            >>> [tuple(nat.vertex_coordinates(v)) for v in neigh]
            [(2, 1), (0, 1), (1, 2), (1, 0)]

    Diagonal Grid:
        The diagonal grid is created when ``diagonal=True``.
        The neighbors are given in the following order.

        * 00 = 0: right, up;
        * 01 = 1: right, down;
        * 10 = 2: left, up;
        * 11 = 3: left, down.

        Each binary value indicates the direction along
        a given axis,
        with 0 representing forward and 1 representing backward.
        The most significant bit corresponds to the
        direction along the X-axis,
        while the least significant bit corresponds to the
        direction along the Y-axis.

        Consider a vertex :math:`(x, y)`. Then,
        its four neighbors are :math:`(x \pm 1, y \pm 1)`.
        The order of neighbors is depicted in
        :ref:`fig-diagonal-grid-order`.

        .. graphviz:: ../../graphviz/grid/diagonal-grid-neigh-order.dot
            :align: center
            :layout: neato
            :name: fig-diagonal-grid-order
            :caption: Figure: The order of neighbors in the diagonal grid.

        For example,
        consider the :math:`3 \times 3` periodic diagonal grid
        (:ref:`fig-3x3-diagonal-grid`).

        .. graphviz:: ../../graphviz/grid/periodic-diagonal.dot
            :align: center
            :layout: neato
            :name: fig-3x3-diagonal-grid
            :caption: Figure: Periodic diagonal 3x3-grid.

        The neighbors of :math:`(0, 0)` and :math:`(1, 1)` with
        respect to the order of neighbors are

        .. doctest::

            >>> diag = hpw.Grid(3, diagonal=True, periodic=True)
            >>> neigh = diag.neighbors((0, 0))
            >>> [tuple(diag.vertex_coordinates(v)) for v in neigh]
            [(1, 1), (1, 2), (2, 1), (2, 2)]
            >>> 
            >>> neigh = diag.neighbors((1, 1))
            >>> [tuple(diag.vertex_coordinates(v)) for v in neigh]
            [(2, 2), (2, 0), (0, 2), (0, 0)]

        In the case of a diagonal grid with borders,
        there exist two independent subgrids.
        In other words, a vertex in one subgrid is not accessible from
        a vertex in the other subgrid.

        .. graphviz:: ../../graphviz/grid/bounded-diagonal.dot
            :align: center
            :layout: neato
            :name: fig-bounded-diagonal-grid
            :caption: Figure: Bounded 3x3-grid in the diagonal representation.

        Two independent subgrids also occur if
        the diagonal grid has periodic boundary conditions and both
        dimensions are even.
        Figure :ref:`fig-even-dim-diagonal` illustrates
        an example of this case.

        .. graphviz:: ../../graphviz/grid/even-dim-diagonal.dot
            :align: center
            :layout: neato
            :name: fig-even-dim-diagonal
            :caption: Figure: 4x4-grid with cyclic boundary conditions.
    """
    try:
        if len(dim) != 2:
            raise ValueError("Expected 2-dimensional tuple. "
                             + "Received " + str(dim)
                             + " instead.")
    except TypeError:
        # then int
        dim = (dim, dim)

    if not diagonal:
        basis = [1, -1, 2, -2]
    else:
        basis = np_array([[1, 1], [1, -1],
                          [-1, 1], [-1, -1]])

    g = IntegerLattice(dim, basis=basis, periodic=periodic,
                       multiedges=multiedges, weights=weights,
                       copy=copy)
    g.diagonal = diagonal

    return g
