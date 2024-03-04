from .square_lattice import SquareLattice

def Cycle(num_vert, weights=None, multiedges=None):
    r"""
    Cycle graph.

    Parameters
    ----------
    num_vert : int
        Number of vertices in the cycle.

    Notes
    -----
    The cycle can be interpreted as being embedded on the line
    with a cyclic boundary condition.
    In this context,
    we assign the direction ``0`` to the right and ``1`` to the left.
    This assignment alters the order of the arcs.
    Any arc with a tail denoted by :math:`v`
    has the numerical label :math:`2v` if it points to the right,
    and the numerical label :math:`2v + 1` if it points to the left.
    Figure 1 illustrates the arc numbers of a cycle with 3 vertices.

    .. graphviz:: ../../graphviz/cycle-arcs.dot
        :align: center
        :layout: neato
        :caption: Figure 1.

    """
    basis = [1, -1]
    g = SquareLattice(num_vert, basis, True, weights, multiedges)
    return g
