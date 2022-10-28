import numpy as np
import scipy
import scipy.sparse
import networkx as nx
from .base_walk import *
from constants import DEBUG

if DEBUG:
    from time import time as now
    from guppy import hpy # used to check memory usage

def _binary_search(v, elem, start=0, end=None):
    r"""
    expects sorted array and executes binary search in the subarray
    v[start:end] searching for elem.
    Return the index of the element if found, otherwise returns -1
    Cormen's binary search implementation.
    Used to improve time complexity
    """
    if end == None:
        end = len(v)
    
    while start < end:
        mid = int((start + end)/2)
        if elem <= v[mid]:
            end = mid
        else:
            start = mid + 1

    return end if v[end] == elem else -1

class Coined(BaseWalk):
    r"""
    Manage an instance of the coined quantum walk model
    on general unweighted graphs.

    Methods for managing, simulating and generating operators of
    the coined quantum walk model for general graphs are available.

    For implementation details, see the Notes Section.

    Parameters
    ----------
    adj_matrix : :class:`scipy.sparse.csr_array`
        Adjacency matrix of the graph on
        which the quantum walk occurs.

    Raises
    ------
    TypeError
        if ``adj_matrix`` is not an instance of
        :class:`scipy.sparse.csr_array`.

    Notes
    -----
    The recommended parameter type is
    :class:`scipy.sparse.csr_array` using ``dtype=np.int8``
    with 1 denoting adjacency and 0 denoting non-adjacency.
    If any entry is different from 0 or 1,
    some methods may not work as expected.

    For more information about the general general Coined Quantum Walk Model,
    check Quantum Walks and Search Algorithms's
    Section 7.2: Coined Walks on Arbitrary Graphs [1]_.

    The Coined class uses the arc notation
    and the Hilbert space :math:`\mathcal{H}^{2|E|}` for general Graphs.
    That is, for a given graph :math:`G(V, E)`,
    the walk occurs in the graph :math:`\vec{G}(V, A)`
    where

    .. math::
        \begin{align*}
            A = \bigcup_{(v,u) \in E} \{(v, u), (u, v)\}.
        \end{align*}

    Matrices and states respect the sorted arcs order,
    i.e. :math:`(v, u) < (v', u')` if either :math:`v < v'` or
    :math:`v = v'` and :math:`u < u'`
    where :math:`(v, u), (v', u')` are valid arcs.

    For example, the graph :math:`G(V, E)` shown in
    Figure 1 has adjacency matrix ``adj_matrix``.

    >>> import numpy as np
    >>> adj_matrix = np.matrix([[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]])
    >>> adj_matrix
    matrix([[0, 1, 0, 0],
            [1, 0, 1, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 0]])

     .. graphviz:: ../../graphviz/coined-model-sample.dot
        :align: center
        :layout: neato
        :caption: Figure 1

    The corresponding arcs are

    >>> arcs = [(i, j) for i in range(4) for j in range(4) if A[i,j] == 1]
    >>> arcs
    [(0, 1), (1, 0), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]

    Note that ``arcs`` is already sorted, hence the labels are

    >>> arcs_labels = {arcs[i]: i for i in range(len(arcs))}
    >>> arcs_labels
    {(0, 1): 0, (1, 0): 1, (1, 2): 2, (1, 3): 3, (2, 1): 4, (2, 3): 5, (3, 1): 6, (3, 2): 7}

    The arcs labels are illustrated in Figure 2.

    .. graphviz:: ../../graphviz/coined-model-edges-labels.dot
        :align: center
        :layout: neato
        :caption: Figure 2

    If we would write the arcs labels respecting the the adjacency matrix format,
    we would have the matrix ``adj_labels``.
    Intuitively, the arcs are labeled in left-to-right top-to-bottom fashion.

    >>> adj_labels = [[arcs_labels[(i,j)] if (i,j) in arcs_labels else '' for j in range(4)]
    ...             for i in range(4)]
    >>> adj_labels = np.matrix(adj_labels)
    >>> adj_labels
    matrix([['', '0', '', ''],
            ['1', '', '2', '3'],
            ['', '4', '', '5'],
            ['', '6', '7', '']], dtype='<U21')

    For consistency, any state :math:`\ket\psi \in \mathcal{H}^{2|E|}`
    is such that :math:`\ket\psi = \sum_{i = 0}^{2|E| - 1} \psi_i \ket{i}`
    where :math:`\ket{i}` is the computational basis state
    associated to the :math:`i`-th arc.
    In our example, the state

    >>> psi = np.matrix([1/np.sqrt(2), 0, 1j/np.sqrt(2), 0, 0, 0, 0, 0]).T
    >>> psi
    matrix([[0.70710678+0.j        ],
            [0.        +0.j        ],
            [0.        +0.70710678j],
            [0.        +0.j        ],
            [0.        +0.j        ],
            [0.        +0.j        ],
            [0.        +0.j        ],
            [0.        +0.j        ]])

    corresponds to the walker being at vertex 0
    and the coin pointing to vertex 1 with
    associated amplitude of :math:`\frac{1}{\sqrt 2}`, and
    to the walker being at vertex 1
    and the coin pointing to vertex 2 with
    associated amplitude of :math:`\frac{\text{i}}{\sqrt 2}`.

    .. todo::
        * Add option: numpy dense matrix as parameters.
        * Add option: networkx graph as parameter.

    References
    ----------
    .. [1] Portugal, Renato. "Quantum walks and search algorithms".
        Vol. 19. New York: Springer, 2013.
    """

    def __init__(self, adj_matrix):
        super().__init__(adj_matrix)

        # Expects adjacency matrix with only 0 and 1 as entries
        self.hilb_dim = self.adj_matrix.sum()

    def flip_flop_shift_operator(self):
        r"""
        Create the flip-flop shift operator (:math:`S`) based on
        the ``adj_matrix`` attribute.

        Returns
        -------
        :class:`scipy.sparse.csr_matrix`
            Flip-flop shift operator.

        Notes
        -----

        .. todo::
            - If `adj_matrix` parameter is not sparse,
                throw exception of convert to sparse.

        .. note::
            Check :class:`Coined` Notes for details
            about the order and dimension of the computational basis.


        The flip-flop shift operator :math:`S` is defined such that

        .. math::
            \begin{align*}
                S \ket{(v, u)} &= \ket{(u, v)} \\
                \implies S\ket i &= \ket j
            \end{align*}

        where :math:`i` is the label of the edge :math:`(v, u)` and
        :math:`j` is the label of the edge :math:`(u, v)`.


        For more information about the general flip-flop shift operator,
        check "Quantum Walks and Search Algorithms"
        Section 7.2: Coined Walks on Arbitrary Graphs [1]_.
        

        References
        ----------
        .. [1] Portugal, Renato. "Quantum walks and search algorithms".
            Vol. 19. New York: Springer, 2013.

        Examples
        --------
        Consider the Graph presented in the
        :class:`Coined` Notes Section example.
        The corresponding flip-flop shift operator is

        >>> from scipy.sparse import csr_array
        >>> import CoinedModel as qcm
        >>> A = csr_array([[0, 1, 0, 0],
        ...                [1, 0, 1, 1],
        ...                [0, 1, 0, 1],
        ...                [0, 1, 1, 0]])
        >>> S = qcm.flip_flop_shift_operator(A)
        >>> Sd = S.todense()
        >>> Sd
        array([[0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0]], dtype=int8)

        Note that as required, :math:`S^2 = I`,
        :math:`S \ket 0 = \ket 1`, :math:`S \ket 1 = \ket 0`,
        :math:`S \ket 2 = \ket 4`, :math:`S \ket 4 = \ket 2`, etc.

        >>> (Sd @ Sd == np.eye(8)).all() # True by definition
        True
        >>> Sd @ np.array([1, 0, 0, 0, 0, 0, 0, 0]) # S|0> = |1>
        array([0., 1., 0., 0., 0., 0., 0., 0.])
        >>> Sd @ np.array([0, 1, 0, 0, 0, 0, 0, 0]) # S|1> = |0>
        array([1., 0., 0., 0., 0., 0., 0., 0.])
        >>> Sd @ np.array([0, 0, 1, 0, 0, 0, 0, 0]) # S|2> = |4>
        array([0., 0., 0., 0., 1., 0., 0., 0.])
        >>> Sd @ np.array([0, 0, 0, 0, 1, 0, 0, 0]) # S|4> = |2>
        array([0., 0., 1., 0., 0., 0., 0., 0.])
        """

        if DEBUG:
            start_time = now()

        # expects weights to be 1 if adjacent
        num_edges = self.adj_matrix.sum()

        # Storing edges' indeces in data.
        # Obs.: for some reason this does not throw exception,
        #   so technically it is a sparse matrix that stores a zero entry
        orig_dtype = self.adj_matrix.dtype
        self.adj_matrix.data = np.arange(num_edges)

        # Calculating flip_flop_shift columns
        # (to be used as indices of a csr_array)
        row = 0
        S_cols = np.zeros(num_edges)
        for edge in range(num_edges):
            if edge >= self.adj_matrix.indptr[row + 1]:
                row += 1
            # Column index (in the adj_matrix struct) of the current edge
            col_index = _binary_search(self.adj_matrix.data, edge,
                                        start=self.adj_matrix.indptr[row],
                                        end=self.adj_matrix.indptr[row+1])
            # S[edge, S_cols[edge]] = 1
            # S_cols[edge] is the edge_id such that
            # S|edge> = |edge_id>
            S_cols[edge] = self.adj_matrix[
                self.adj_matrix.indices[col_index], row
            ]

        # Using csr_array((data, indices, indptr), shape)
        # Note that there is only one entry per row and column
        S = scipy.sparse.csr_array(
            ( np.ones(num_edges, dtype=np.int8),
              S_cols, np.arange(num_edges+1) ),
            shape=(num_edges, num_edges)
        )

        # restores original data to adj_matrix
        self.adj_matrix.data = np.ones(num_edges, dtype=orig_dtype)

        # TODO: compare with old approach for creating S

        if DEBUG:
            print("flip_flop_shift_operator Memory: "
                  + str(hpy().heap().size))
            print("flip_flop_shift_operator Time: "
                  + str(now() - start_time))

        return S

    def coin_operator(self, coin='grover'):
        """
        Generate a coin operator based on the graph structure.

        Parameters
        ----------
        coin : {'grover', 'fourier', 'hadamard'}
            Type of the coin to be used.
        
        Returns
        -------
        :class:`scipy.sparse.csr_matrix`

        .. todo::
            Check if return automatically changed to
            :class:`scipy.sparse.csr_array`.

        Raises
        ------
        ValueError
            If ``coin='hadamard'`` and any vertex of the graph
            has a non-power of two degree.

        Notes
        -----
        Due to the chosen computational basis
        (see :class:`Coined` Notes),
        the resulting operator is a block diagonal where
        each block is the :math:`\deg(v)`-dimensional ``coin``.
        Consequently, there are :math:`|V|` blocks.

        .. todo::
            Implement general fourier and hadamard coins.
        """
        # dict with valid coins as keys and the respective
        # function pointers.
        coin_funcs = {
            'fourier': Coined._fourier_coin,
            'grover': Coined._grover_coin,
            'hadamard': Coined._hadamard_coin
        }

        if coin not in coin_funcs.keys():
            raise ValueError(
                'Invalid coin. Expected any of '
                + str(list(coin_funcs.keys())) + ', '
                + "but received '" + str(coin) + "'."
            )

        num_vert = self.adj_matrix.shape[0]
        degrees = self.adj_matrix.sum(1) # sum rows

        blocks = (coin_funcs[coin](degrees[v]) for v in range(num_vert))

        return scipy.sparse.block_diag(blocks, format='csr')

    @staticmethod
    def _fourier_coin(dim):
        import scipy.linalg
        return scipy.linalg.dft(dim, scale='sqrtn')

    @staticmethod
    def _grover_coin(dim):
        return np.array(2/dim * np.ones((dim, dim)) - np.identity(dim))

    @staticmethod
    def _hadamard_coin(dim):
        import scipy.linalg
        return scipy.linalg.hadamard(dim) / np.sqrt(dim)

    def oracle(self, vertices):
        r"""
        Create the oracle that marks the given vertices.

        The oracle flips the phase of every entry associated
        with the marked vertices.

        .. todo::
            - Search about different valid oracles for the coined model.

        Parameters
        ----------
        vertices : array_like
            ID(s) of the vertex (vertices) to be marked.

        Returns
        -------
        :class:`scipy.sparse.csr_array`

        Notes
        -----
        The oracle is described by

        .. math::
            R = I - 2 \sum_{v \in M}\sum_{u \in \text{adj}(v)}
                \ketbra{(v,u)}{(v,u)}

        where :math:`M` is the set of marked vertices and
        :math:`\text{adj}(v)` is the set of all vertices adjacent
        to :math:`v \in V`.
        See :class:`Coined` Notes for more details about
        the used computational basis.
        """
        try:
            iter_ = iter(vertices)
        except:
            vertices = [vertices]
            iter_ = iter(vertices)

        R = scipy.sparse.identity(self.hilb_dim)

        for vertex_id in iter_:
            first_edge = self.adj_matrix.ptr[vertex_id]
            last_edge = self.adj_matrix.ptr[vertex_id + 1]

            for edge in range(first_edge, last_edge):
                R[edge, edge] = -1

        return np.matrix(R)

    def evolution_operator(self, hpc=False, coin='grover'):
        """
        Create the standard evolution operator.

        Parameters
        ----------
        hpc : bool, default=False
            Whether or not evolution operator should be
            constructed using nelina's high-performance computating.

        coin : str, default='grover'
            The coin to be used as diffusion operator.
            See :obj:`coin_operator`'s ``coin``
            attribute for valid options.

        Returns
        -------
        U_w : :class:`scipy.sparse.csr_array`
            The evolution operator.

        Notes
        -----
        The evolution operator is

        .. math::
            U_w = SC

        where :math`S` is the flip-flop shift operator and
        :math:`C` is the coin operator.

        See Also
        --------
        flip_flop_shift_operator
        coin_operator
        """
        S = self.flip_flop_shift_operator()
        C = self.coin_operator(coin=coin)

        if hpc:
            #TODO: import neblina and implement
            raise NotImplementedError (
                'Calculating the evolution operator via'
                + 'hpc (high-performance computing)'
                + 'is not supported yet.'
            )
            return None
        return S@C

    def search_evolution_operator(self, vertices, hpc=False,
                                  coin='grover'):
        """
        Create the search evolution operator.

        Parameters
        ----------
        vertices : array_like
            The marked vertex (vertices) IDs.
            See :obj:`oracle`'s ``vertices`` parameter.

        hpc : bool, default=False
            Whether or not evolution operator should be
            constructed using nelina's high-performance computating.

        coin : str, default='grover'
            The coin to be used as diffusion operator.
            See :obj:`evolution_operator` and :obj:`coin_operator`.

        Returns
        -------
        U : :class:`scipy.sparse.csr_array`
            The search evolution operator.

        See Also
        --------
        coin_operator
        evolution_operator
        oracle

        Notes
        -----
        The search evolution operator is

        .. math::
            U = U_w R = S C R

        where :math:`U_w` is the coined quantum walk evolution operator
        and :math:`R` is the oracle [1]_.

        References
        ----------
        .. [1] Portugal, Renato. "Quantum walks and search algorithms".
            Vol. 19. New York: Springer, 2013.
        """
        evol_op = self.evolution_operator(coin=coin, hpc=hpc)
        oracle = self.oracle(vertices)

        if hpc:
            # TODO: check if sending SCR instead of U_wR
            # to neblina is more efficient
            raise NotImplementedError (
                'Calculating the search evolution operator via'
                + 'hpc (high-performance computing)'
                + 'is not supported yet.'
            )
            return None
        return evol_op @ oracle

    def probability_distribution(self, states):
        """
        Compute the probability distribution of given states.

        The probability of the walker being found on each vertex
        for the given states.

        Parameters
        ----------
        states : :class:`numpy.ndarray`
            The states used to compute the probabilities.

        Returns
        -------
        probabilities : :class:`numpy.ndarray`

        See Also
        --------
        simulate_walk
        """
        # TODO: test with nonregular graph
        # TODO: test with nonuniform condition
        if DEBUG:
            start = now()

        if len(states.shape) == 1:
            states = [states]

        # TODO: check if dimensions match and throw exception if necessary
        # TODO: check if just creates reference (no hard copy)
        edges_indices = self.adj_matrix.indptr 

        # TODO: check it is more efficient on demand or
        # using extra memory (aux_prob)
        # aux_prob = ElementwiseProbability(state)
        # first splits state per vertex,
        # then calculates probability of each vertex direction,
        # then sums the probabilities resulting in
        # the vertix final probability
        prob = np.array([[
                Coined._elementwise_probability(
                    states[i][edges_indices[j]:edges_indices[j + 1]]
                ).sum()
                for j in range(len(edges_indices) - 1)
            ] for i in range(len(states)) ])

        if DEBUG:
            end = now()
            print("probability_distribution: " + str(end - start) + 's')
        # TODO: benchmark (time and memory usage)
        return prob

    def state(self, entries, amplitudes=None, type='vertex_dir'):
        """
        Generates a valid state.

        The state corresponds to the walker being in a superposition
        of ``entries[i]`` with amplitudes ``amplitudes[i]``, respectively.

        The ``amplitudes`` are normalized so the state is unitary.

        Parameters
        ----------
        entries :
            There are three types of accaptable entries:
            `(vertices, coin_dir)`, `(vertices, target_vertices)`,
            and `arc_number`.


            vertices : :class:`numpy.array`
                The vertices corresponding to the positions of the walker
                in the superposition.
            coin_dir : :class:`numpy.array`
                The direction to which the coin is pointing to.
                Each array entry is expected to be a value between
                0 and degree(``vertices[i]``) - 1,
                which respect the sorted arcs order.
            target_vertices : :class:`numpy.array`
                The target vertex for the given positions.
                This respects the arc notation,
                i.e. (``vertices[i]``, ``target_vertices[i]``).
            arc_number : :class:`numpy.array`
                The arc number with respect to the sorted arcs order.

        amplitudes : :class:`numpy.array`, default=None
            The amplitudes of each entry.
            If ``none``, computes the entries' uniform superposition.

        type : {'vertex_dir', 'arc_notation', 'arc_order'}
            The type of the ``entries`` argument.
            The (default) value 'vertex_dir' corresponds to
            the (`vertices, coin_dir`) entry.
            The 'arc_notation' value corresponds to the
            `(vertices, target_vertices)` entry.
            The 'arc_order' value corresponds to the `arc_number` entry.


        Raises
        ------
        ValueError
            If ``type`` has invalid value.

            If ``type='arc_notation'`` and there exists an entry such that
            ``vertices[i]`` and ``target_vertices[i]`` are not adjacent.

        IndexError
            If ``type='vertex_dir'`` and ``coin_dir`` is not a value in
            the valid interval (from 0 to degree(``vertices[i]``) - 1).

        Notes
        -----
        If entries are repeated, they are overwritten by the last one.

        More efficient implementation of state construction is desirable.

        """

        def _normalize(state, error=1e-16):
            norm = np.linalg.norm(state)
            if 1 - error <= norm and norm <= 1 + error:
                return state
            return state / norm

        def _vertex_dir(adj_matrix, state, entries, amplitudes):
            sources, directions = entries
            indices = adj_matrix.indices
            indptr = adj_matrix.indptr
            print(indices)
            print(indptr)

            for i in range(len(sources)):
                src = sources[i]
                arc = indptr[src] + directions[i]

                if arc < indptr[src] or arc >= indptr[src+1]:
                    raise IndexError(
                        "The " + str(i) + "-th entry (vertex "
                        + str(src)
                        + ") expected a direction value in the [0, "
                        + str(indptr[src+1] - indptr[src] - 1)
                        + "] interval. But received the value "
                        + str(directions[i]) + " instead."
                    )

                state[arc] = amplitudes[i] if amplitudes is not None else 1

            return state

        def _arc_notation(adj_matrix, state, entries, amplitudes):
            sources, targets = entries
            indices = adj_matrix.indices
            indptr = adj_matrix.indptr

            for i in range(len(sources)):
                src = sources[i]
                
                if adj_matrix[src, targets[i]] == 0:
                    raise ValueError(
                        "At the " + str(i) + "-th entry: vertices "
                        + str(src) + " and " + str(targets[i])
                        + " are not adjacent."
                    )

                arc = _binary_search(indices, targets[i],
                                     start=indptr[src],
                                     end=indptr[src+1])

                state[arc] = amplitudes[i] if amplitudes is not None else 1

            return state


        def _arc_order(adj_matrix, state, entries, amplitudes):
            if amplitudes is None:
                for i in entries:
                    state[i] = 1
            else:
                for i in range(len(entries)):
                    state[entries[i]] = amplitudes[i]

            return state
        
        funcs = {'vertex_dir' : _vertex_dir,
                 'arc_notation' : _arc_notation,
                 'arc_order' : _arc_order}

        if type not in list(funcs.keys()):
            raise ValueError(
                    'Invalid `type` argument. Expected any from '
                    + str(list(funcs.keys()))
            )

        state = (np.zeros(self.hilb_dim) if amplitudes is None else
                 np.zeros(self.hilb_dim, amplitudes.dtype))
        state = funcs[type](self.adj_matrix, state, entries, amplitudes)

        return _normalize(state)
