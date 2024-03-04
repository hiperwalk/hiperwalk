from .square_lattice import SquareLattice

def Line(num_vert):
    r"""
    Finite line graph (path graph).

    Parameters
    ----------
    num_vert : int
        The number of vertices on the line.

    Notes
    -----
    In the :obj:`Line` class, directions can be assigned to the arcs. 
    An arc pointing to the right has direction 0 (e.g., (1, 2)), 
    and an arc pointing to the left has direction 1 (e.g., (2, 1)).

    The order of the arcs is determined by their direction. 
    Thus, for a vertex :math:`v \in V`, 
    the arcs :math:`(v, v + 1)` and :math:`(v, v - 1)` 
    have labels :math:`a_0` and :math:`a_1` respectively, 
    with :math:`a_0 < a_1`. 
    The only exceptions to this rule are the extreme vertices 0 
    and :math:`|V| - 1`, as they have outdegree 1. 

    Apart from these exceptions, 
    for any two vertices :math:`v_1 < v_2`, 
    any arc with tail :math:`v_1` will have a label smaller than the 
    label of any arc with tail :math:`v_2`.

    For instance, Figure 1 illustrates the labels of the arcs of a path graph with 4 vertices.
    
    .. graphviz:: ../../graphviz/line-arcs.dot
        :align: center
        :layout: neato
        :caption: Figure 1.
    """

    basis = [1, -1]
    g = SquareLattice(num_vert, basis, False, weights, multiedges)
    return g
