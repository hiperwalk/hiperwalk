from numpy import array as np_array
from .square_lattice import SquareLattice
from types import MethodType

def Grid(dim, periodic=True, diagonal=False,
         weights=None, multiedges=None):
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
        Grid dimensions in ``(_dim[0], _dim[1])`` format.
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

    g = SquareLattice(dim, basis, periodic, weights, multiedges)
    # g._neighbor_index = MethodType(_neighbor_index, g)
    g.diagonal = diagonal

    return g
