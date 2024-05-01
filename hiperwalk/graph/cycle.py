from .integer_lattice import IntegerLattice

def Cycle(num_vert, multiedges=None, weights=None):
    r"""
    Cycle graph.

    Parameters
    ----------
    num_vert : int
        Number of vertices in the cycle.
    multiedges, weights: scipy.sparse.csr_array, default=None
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
    the **order of neighbors** is
    the neighbor to the right first,
    followed by the neighbor to the left.
    In other words, for any vertex :math:`v`,
    the neighbors are given in the order :math:`[v + 1, v - 1]`.

    .. testsetup::

        import hiperwalk as hpw

    .. doctest::

        >>> g = hpw.Cycle(10)
        >>> g.neighbors(0)
        array([1, 9])
        >>> g.neighbors(1)
        array([2, 0])
        >>> g.neighbors(8)
        array([9, 7])
        >>> g.neighbors(9)
        array([0, 8])

    """
def Cycle(num_vert, multiedges=None, weights=None):
    r"""
    Construct a cycle graph.

    A cycle graph is a graph that forms a single closed loop, 
    where each vertex is connected to exactly two other vertices, 
    forming a circular structure.

    Parameters
    ----------
    num_vert : int
        The number of vertices in the cycle. 
    multiedges : scipy.sparse.csr_array, optional
        Specifies the number of multiple edges between the same 
        pair of vertices. 
        Defaults to None.
    weights : scipy.sparse.csr_array, optional
        Assigns weights to the edges of the graph. 
        Defaults to None.

    Returns
    -------
    :class:`hiperwalk.Graph`
        Returns an instance of a cycle graph. 
        Refer to :ref:`graph_constructors` for more details.

    See Also
    --------
    :ref:`graph_constructors`
        Further information on graph constructors.

    Notes
    -----
    The cycle is conceptually embedded on a line with cyclic 
    boundary conditions. The **order of neighbors** for any 
    vertex :math:`v` is :math:`[v + 1, v - 1]`, where the right 
    neighbor is listed first, followed by the left neighbor.

    .. testsetup::

        import hiperwalk as hpw

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
    g = IntegerLattice(num_vert, basis, True, weights, multiedges)
    return g
