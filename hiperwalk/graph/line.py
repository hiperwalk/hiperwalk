from .integer_lattice import IntegerLattice

def Line(num_vert, multiedges=None, weights=None, copy=False):
    r"""
    Finite line graph (path graph) constructor.

    Parameters
    ----------
    num_vert : int
        The number of vertices on the line.

    multiedges, weights: scipy.sparse.csr_array, default=None
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
    The **order of neighbors** is
    the neighbor to the right first,
    followed by the neighbor to the left.
    In other words, for any vertex :math:`v`,
    the neighbors are given in the order :math:`[v + 1, v - 1]`.

    .. testsetup::

        import hiperwalk as hpw

    .. doctest::

        >>> g = hpw.Line(10)
        >>> list(g.neighbors(0)) # 0 and 9 are not adjacent
        [1]
        >>> list(g.neighbors(1))
        [2, 0]
        >>> list(g.neighbors(8))
        [9, 7]
        >>> list(g.neighbors(9)) # 0 and 9 are not adjacent
        [8]

    """

    basis = [1, -1]
    g = IntegerLattice(num_vert, basis=basis, periodic=False,
                       multiedges=multiedges, weights=weights,
                       copy=copy)
    return g
