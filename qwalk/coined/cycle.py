from scipy.sparse import csr_array
from .coined import *

class Cycle(Coined):
    r"""
    Class for managing quantum walk on the cycle.

    Parameters
    ----------
    num_vert : int
        Number of vertices in the cycle.

    Notes
    -----
    The cycle may be interpreted as being embedded on the line
    with cyclic boundary condition.

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
        print(row_ind)
        # lower digonal
        col_ind = [(col-shift) % num_vert for col in range(num_vert)
                             for shift in [-1, 1]]
        print(col_ind)
        adj_matrix = csr_array((data, (row_ind, col_ind)))
    
        # initializing
        super().__init__(adj_matrix)

    def has_persistent_shift_operator(self):
        return True

    def persistent_shift_operator(self):
        r"""
        Create the persistent shift operator (:math:`S`) based on the
        ``adj_matrix`` atribute.

        Returns
        -------
        :class:`scipy.sparse.csr_matrix`
            Persistent shift operator.

        Notes
        -----
        The persistent shift operator :math:`S`
        for any vertex :math:`v \in V` is defined by

        .. math::
            \begin{align*}
                S \ket{2v} &= \ket{2v + 2 \mod 2|E|} \\
                S \ket{2v+1} &= \ket{2v+1 - 2 \mod 2|E|}. 
            \end{align*}

        Hence, if the walker reaches a boundary vertex
        :math:`u \in \{0, |V| - 1\}`,
        the coin starts pointing in the opposite direction.

        .. todo::
            Add option to implement boundary vertices as sinks.
        """

        num_edges = 2*self.adj_matrix.shape[0]

        data = np.ones(num_edges, np.int8)
        indptr = np.arange(num_edges + 1)
        indices = np.zeros(num_edges)

        indices = [(i - 2) % num_edges if i % 2 == 0
                   else (i + 2) % num_edges
                   for i in range(num_edges)]

        S = scipy.sparse.csr_array((
            data, indices, indptr        
        ))

        return S

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

    def coin_operator(self, coin='hadamard'):
        """
        Same as :meth:`Coined.coin_operator`,
        but uses Hadamard as default coin.
        """
        return super().coin_operator(coin)

    def evolution_operator(self, persistent_shift=True, hpc=False,
                           coin='hadamard'):
        """
        Same as :meth:`Coined.evolution_operator` with
        overriden default arguments.
        """
        return super().evolution_operator(persistent_shift, hpc, coin)
