from .square_lattice import SquareLattice

def Line(num_vert, multiedges=None, weights=None):
    r"""
    Finite line graph (path graph).

    Parameters
    ----------
    num_vert : int
        The number of vertices on the line.

    multiedges, weights: :class:`scipy.sparse.csr_array`, default=None
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
    The **order of neighbors** is
    the neighbor to the right first,
    followed by the neighbor to the left.
    In other words, for any vertex :math:`v`,
    the neighbors are given in the order :math:`[v + 1, v - 1]`.

    .. testsetup::

        >>> import hiperwalk as hpw

    .. doctest::

        >>> g = hpw.Line(10)
        >>> g.neighbors(0) # 0 and 9 are not adjacent
        [1]
        >>> g.neighbors(1)
        [2, 0]
        >>> g.neighbors(8)
        [9, 7]
        >>> g.neighbors(9) # 0 and 9 are not adjacent
        [8]

    """

    basis = [1, -1]
    g = SquareLattice(num_vert, basis, False, weights, multiedges)
    return g
