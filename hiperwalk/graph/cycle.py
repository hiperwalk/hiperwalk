from .square_lattice import SquareLattice

def Cycle(num_vert, multiedges=None, weights=None):
    r"""
    Cycle graph.

    Parameters
    ----------
    num_vert : int
        Number of vertices in the cycle.
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
    The cycle can be interpreted as being embedded on the line
    with a cyclic boundary condition.
    In this context,
    the **order of the neighbors** is
    the neighbor to the right first,
    followed by the neighbor to the left.
    In other words, for any vertex :math:`v`,
    the neighbors are given in the order :math:`[v + 1, v - 1]`.

    .. testsetup::

        >>> import hiperwalk as hpw

    .. doctest::

        >>> g = hpw.Cycle(10)
        >>> g.neighbors(0)
        [1, 9]
        >>> g.neighbors(1)
        [2, 0]
        >>> g.neighbors(8)
        [9, 7]
        >>> g.neighbors(9)
        [0, 8]

    """
    basis = [1, -1]
    g = SquareLattice(num_vert, basis, True, weights, multiedges)
    return g
