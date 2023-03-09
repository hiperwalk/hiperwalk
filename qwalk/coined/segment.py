import numpy as np
from scipy.sparse import csr_array
from sys import path as sys_path
from .coined import Coined

class Segment(Coined):
    r"""
    Class for managing quantum walks on the segment.
    In other words, a finite one-dimensional lattice.

    Parameters
    ----------
    num_vert : int
        Number of vertices in the segment.

    Notes
    -----
    Since :class:`Segment` is built on top of :class:`Coined`,
    operators and states respect the edge order
    (See :class:`Coined` notes for more details).
    As a consequence, for any vertex :math:`v \in V`,
    the state :math:`\ket{2v - d}` for :math:`d \in \{0, 1\}`
    corresponds to the walker being on vertex :math:`v` and the
    coin pointing rightwars (:math:`d = 0`) or leftwards (:math:`d = 1`).
    For example, the edges labels of the 4-vertices :class:`Segment`
    are represented in Figure 1.
    
    .. graphviz:: ../../graphviz/coined-segment-edges-labels.dot
        :align: center
        :layout: neato
        :caption: Figure 1.
    """

    def __init__(self, num_vert):
        # Creating adjacency matrix
        # The end vertices are only adjacent to 1 vertex.
        # Every other vertex is adjacent to two vertices.
        data = np.ones(2*(num_vert-1), dtype=np.int8)
        # upper diagonal
        row_ind = [lin+shift for shift in range(2)
                             for lin in range(num_vert - 1)]
        # lower digonal
        col_ind = [col+shift for shift in [1, 0]
                             for col in range(num_vert - 1)]
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
            Shift operator.

        Notes
        -----
        The persistent shift operator :math:`S`
        for any vertex :math:`v \in V` is defined by

        .. math::
            \begin{align*}
                S \ket{2v} &= \ket{\min(2v + 2, 2|V| - 1)} \\
                S \ket{2v-1} &= \ket{\max(2v - 1 - 2, 0)}. 
            \end{align*}

        Hence, if the walker reaches a boundary vertex
        :math:`u \in \{0, |V| - 1\}`,
        the coin starts pointing in the opposite direction.

        .. todo::
            Add option to implement boundary vertices as sinks.
        """

        num_edges = 2*self.adj_matrix.shape[0] - 2

        data = np.ones(num_edges, np.int8)
        indptr = np.arange(num_edges + 1)
        indices = np.zeros(num_edges)

        indices = [i + 2 if i % 2 == 1 else i - 2
                   for i in range(num_edges)]
        indices[0] = 1
        indices[num_edges - 1] = num_edges - 2

        S = csr_array((
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
        indices = self.adj_matrix.indices
        indptr = self.adj_matrix.indptr
        num_vert = self.adj_matrix.shape[0]

        for amplitude, src, coin_dir in entries:
            if coin_dir != 0 and coin_dir != 1:
                raise ValueError(
                    "Invalid entry coin direction for vertex " + str(src)
                    + ". Expected either 0 (rightwards) or 1 (leftwards),"
                    + " but received " + str(coin_dir) + " instead."
                )

            # bound conditions
            if src == 0 and coin_dir != 0:
                raise ValueError(
                    "Invalid entry coin direction for vertex 0."
                    + "Since 0 is the leftmost vertex, "
                    + "the only available direction is 0 (rightwards). "
                    + "But received " + str(coin_dir) + " instead."
                )
            if src == num_vert - 1 and coin_dir != 1:
                raise ValueError(
                    "Invalid entry coin direction for vertex " + str(src)
                    + ". Since " + str(src) + " is the rightmost vertex, "
                    + "the only available direction is 1 (leftwards). "
                    + "But received " + str(coin_dir) + " instead."
                )

            arc = indptr[src] + 1 - coin_dir if src != 0 else 0

            state[arc] = amplitude

        return state

    def state(self, entries, type='vertex_dir'):
        r"""
        Similar to :meth:`qwalk.Coined.state`.

        However, when ``type='vertex_dir'``,
        the default coin directions are used.
        In other words, 0 for rightwards and 1 for leftwards.

        Raises
        ------
        ValueError
            If ``type='vertex_dir'`` and one of the following occurs
            for some ``entries[i] = (vertices[i], coin_dir[i])``.

            * ``coin_dir[i]`` is different from 0 and 1:
                the only valid directions are 0 and 1.
            * ``vertices[i]=0`` and ``coin_dir=1``:
                0 is the leftmost vertex,
                hence the only possible direction is rightwards.
            * ``vertices[i]=``:math:`|V|-1` and ``coin_dir=0``:
                :math:`|V|-1` is the rightmost vertex,
                hence the only possible direction is leftwards.

        Notes
        -----
        The default of :meth:`qwalk.Coined.state` is to use the
        sorted arcs order.
        Hence, the directions would be shifted.
        """
        return super().state(entries, type)
