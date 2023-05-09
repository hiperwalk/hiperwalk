from scipy.sparse import csr_array
from .graph import *

class Cycle(Graph):
    r"""
    Cycle Graph.

    Parameters
    ----------
    num_vert : int
        Number of vertices in the cycle.

    Notes
    -----
    The cycle may be interpreted as being embedded on the line
    with cyclic boundary condition.


    .. todo::
        
        update below

    The edge order respects the default vertex-coin notation.
    In other words, 0 corresponds to the coin pointing rightwards,
    and 1 to the coin pointing leftwards.
    Therefore, the arcs are sorted with respect to this order
    (vertex has precedence over direction and
    right has precedence over left):

    .. math::
        \begin{align*}
            \ket{(v, c)} = \ket{2v + c}
        \end{align*}

    where :math:`v \in V` and :math:`c \in \{0, 1\}`.
    Figure 1 illustrates the arcs of a 3 vertices cycle.

    .. graphviz:: ../../graphviz/coined-cycle-edges-labels.dot
        :align: center
        :layout: neato
        :caption: Figure 1.

    """

    def __init__(self, num_vert):
        # Creating adjacency matrix
        # Every vertex is adjacent to two vertices.
        data = np.ones(2*num_vert, dtype=np.int8)
        # upper diagonal
        row_ind = [lin for lin in range(num_vert)
                       for twice in range(2)]
        # lower digonal
        col_ind = [(col-shift) % num_vert for col in range(num_vert)
                             for shift in [-1, 1]]
        adj_matrix = csr_array((data, (row_ind, col_ind)))
    
        # initializing
        super().__init__(adj_matrix)

    def has_persistent_shift_operator(self):
        r"""
        See :meth:`Graph.has_persistent_shift_operator`.
        """
        return True

    def get_default_coin(self):
        r"""
        Returns the default coin name.

        The default coin for the coined quantum walk on general
        graphs is ``grover``.
        """
        return 'hadamard'

    def _state_vertex_dir(self, state, entries):
        r"""
        Overrides Coined model method so the directions respect
        the default coin directions.
        In other words:
        0 pointing rightwards and 1 pointing leftwards.
        """
        num_vert = self.adj_matrix.shape[0]

        for amplitude, src, coin_dir in entries:
            if coin_dir != 0 and coin_dir != 1:
                raise ValueError(
                    "Invalid entry coin direction for vertex " + str(src)
                    + ". Expected either 0 (rightwards) or 1 (leftwards),"
                    + " but received " + str(coin_dir) + " instead."
                )

            arc = 2*src + coin_dir
            # if src == 0 or src == num_vert - 1:
            #     arc = 2*src + coin_dir 

            print(arc)
            state[arc] = amplitude

        return state

    def _state_arc_notation(self, state, entries):
        num_vert = self.adj_matrix.shape[0]

        for amplitude, src, dst in entries:
            if (dst != (src - 1) % num_vert and
                dst != (src + 1) % num_vert):
                raise ValueError (
                    "Vertices " + str(src) + " and " + str(dst)
                    + " are not adjacent."
                )
            
            arc = (2*src if dst - src == 1 or src - dst == num_vert - 1
                   else 2*src + 1)
            state[arc] = amplitude

        return state
