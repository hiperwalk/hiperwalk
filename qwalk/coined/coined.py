import numpy as np
import scipy
import scipy.sparse
import networkx as nx
from ..base_walk import BaseWalk
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

    def coin_operator(self, coin='grover', coin2=None, vertices2=[]):
        """
        Generate a coin operator based on the graph structure.

        Constructs coin operator depending on the degree of each vertex.
        A single coin type may be applied to all vertices,
        or two coins may be applied to selected vertices.

        Parameters
        ----------
        coin : {'grover', 'fourier', 'hadamard', 'minus_identity'}
            Type of the coin to be used.

        coin2 : {'grover', 'fourier', 'hadamard', 'minus_identity'}, default=None
            Type of the coin to be used for ``vertices2``.

        vertices2 :
            Vertices to use ``coin2`` instead of ``coin``.
        
        Returns
        -------
        :class:`scipy.sparse.csr_matrix`

        .. todo::
            Check if return automatically changed to
            :class:`scipy.sparse.csr_array`.

        Raises
        ------
        ValueError
            If ``coin`` or ``coin2`` values are invalid.

            If ``coin='hadamard'`` and any vertex of the graph
            has a non-power of two degree.

        Notes
        -----
        Due to the chosen computational basis
        (see :class:`Coined` Notes),
        the resulting operator is a block diagonal where
        each block is the :math:`\deg(v)`-dimensional ``coin``.
        Consequently, there are :math:`|V|` blocks.

        """
        # dict with valid coins as keys and the respective
        # function pointers.
        coin_funcs = {
            'fourier': Coined._fourier_coin,
            'grover': Coined._grover_coin,
            'hadamard': Coined._hadamard_coin,
            'minus_identity': Coined._minus_identity
        }

        if coin not in coin_funcs.keys() or (coin2 != None and
            coin2 not in coin_funcs.keys()):
            raise ValueError(
                'Invalid coin. Expected any of '
                + str(list(coin_funcs.keys())) + ', '
                + "but received '" + str(coin) + "'."
            )

        num_vert = self.adj_matrix.shape[0]
        degrees = self.adj_matrix.sum(1) # sum rows
        blocks = []

        if coin2 is None:
            blocks = [coin_funcs[coin](degrees[v])
                      for v in range(num_vert)]
        else:
            not_vertex2 = [True] * num_vert
            for v in vertices2:
                not_vertex2[v] = False

            blocks = [coin_funcs[coin](degrees[v]) if not_vertex2[v]
                      else coin_funcs[coin2](degrees[v])
                      for v in range(num_vert)]

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

    @staticmethod
    def _minus_identity(dim):
        return -np.identity(dim)

    def oracle(self, vertices, oracle_type="standard"):
        r"""
        Create the oracle that marks the given vertices.


        Parameters
        ----------
        vertices : array_like
            ID(s) of the vertex (vertices) to be marked.

        oracle_type : {'standard', 'phase_flip'}
            Oracle type to be used.

        Returns
        -------
        :class:`scipy.sparse.csr_array`

        Notes
        -----
        There are two types of oracle possible.
        The standard oracle applies Grover coin to unmarked vertices and
        flips the phase of marked vertices.
        The other oracle flips the phase of every entry associated
        with the marked vertices,
        leaving the unmarked vertices unchanged.

        The standard oracle is given by

        .. math::
            R &= \sum_{v \in M^C} \sum_{u \in \text{adj}(v)}
                G_{\deg(v)} \ketbra{(v,u)}{(v,u)} \\
            &- \sum_{v \in M}\sum_{u \in \text{adj}(v)}
                \ketbra{(v,u)}{(v,u)},

        and the phase flip oracle is described by

        .. math::
            R = I - 2 \sum_{v \in M}\sum_{u \in \text{adj}(v)}
                \ketbra{(v,u)}{(v,u)}

        where :math:`M` is the set of marked vertices,
        :math:`\text{adj}(v)` is the set of all vertices adjacent
        to :math:`v \in V`, and
        :math:`G_{\deg(v)}` is the Grover matrix of
        dimension :math:`\deg(v)`.
        See :class:`Coined` Notes for more details about
        the used computational basis.
        """
        try:
            iter_ = iter(vertices)
        except:
            vertices = [vertices]
            iter_ = iter(vertices)

        if oracle_type == 'standard':
            return self.coin_operator('grover', 'minus_identity',
                                      vertices)

        elif oracle_type == 'phase_flip':
            R = scipy.sparse.eye(self.hilb_dim, format='csr')

            for vertex_id in iter_:
                first_edge = self.adj_matrix.indptr[vertex_id]
                last_edge = self.adj_matrix.indptr[vertex_id + 1]

                for edge in range(first_edge, last_edge):
                    R.data[edge] = -1

            return R

        raise ValueError(
            "Invalid oracle_type value. Expected one of"
            + "{'standard', 'phase_flip'} but received '"
            + str(oracle_type) + "' instead."
        )


    def has_persistent_shift_operator(self):
        return False

    def evolution_operator(self, persistent_shift=False, hpc=False,
                           coin='grover'):
        """
        Create the standard evolution operator.

        Parameters
        ----------
        persistent_shift : bool, default=False
            Wheter to use persistent shift operator
            (``persistent_shift=True``) or
            the flip flop shift operator (``persistent_shift=False``).

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

        Raises
        ------
        NotImplementedError
            If ``persistent_shift=True`` and the
            persistent shift operator is not defined.

        See Also
        --------
        flip_flop_shift_operator
        coin_operator
        has_persistent_shift_operator

        Notes
        -----
        The evolution operator is

        .. math::
            U_w = SC

        where :math`S` is the flip-flop shift operator and
        :math:`C` is the coin operator.

        The persistent shift operator is only defined for specific graphs
        that can be embedded into the plane.
        Hence, a direction can be inferred (left, right, up, down).
        """
        if persistent_shift and not self.has_persistent_shift_operator():
            raise NotImplementedError (
                "Quantum walk has no persistent shift operator."
            )

        S = (self.persistent_shift_operator() if persistent_shift
             else self.flip_flop_shift_operator())
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

    def search_evolution_operator(self, vertices, persistent_shift=False,
                                  hpc=False, coin='grover'):
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


    #########################################################
    ######### Auxiliary Methods for state() method ##########
    #########################################################

    def _state_vertex_dir(self, state, entries):
        indices = self.adj_matrix.indices
        indptr = self.adj_matrix.indptr

        for amplitude, src, coin_dir in entries:
            arc = indptr[src] + coin_dir

            if arc < indptr[src] or arc >= indptr[src+1]:
                raise IndexError(
                    "For vertex " + str(src) + ", "
                    + "a coin direction value from 0 to "
                    + str(indptr[src+1] - indptr[src] - 1)
                    + " was expected. But the value "
                    + str(coin_dir) + " was received instead."
                )

            state[arc] = amplitude

        return state

    def _state_arc_notation(self, state, entries):
        indices = self.adj_matrix.indices
        indptr = self.adj_matrix.indptr

        for amplitude, src, dst in entries:
            
            if self.adj_matrix[src, dst] == 0:
                raise ValueError(
                    "Vertices " + str(src) + " and " + str(dst)
                    + " are not adjacent."
                )

            arc = _binary_search(indices, dst, start=indptr[src],
                                 end=indptr[src+1])

            state[arc] = amplitude

        return state


    def _state_arc_order(self, state, entries):
        for amplitude, arc in entries:
            state[arc] = amplitude

        return state

    def state(self, entries, type='vertex_dir'):
        """
        Generates a valid state.

        The state corresponds to the walker being in a superposition
        of the ``entries``.

        The final state is normalized in order to be unitary.

        Parameters
        ----------
        entries :
            Each entry is a tuple (or array).
            There are three types of accaptable entries:
            `(amplitude, vertex, coin_dir)`,
            `(amplitude, vertex, dst_vertex)`,
            `(amplitude, arc)`.

            amplitude :
                The amplitudes of the given entry.
            vertex :
                The vertex corresponding to the position of the walker
                in the superposition.
            coin_dir :
                The direction towards which the coin is pointing.
                A value between 0 and degree(``vertices[i]``) - 1
                is expected, respecting the sorted arcs order.
            dst_vertex : 
                The vertex which the coin is pointing to.
                In other words, the tuple
                (vertex, dst_vertex) must be a valid arc.
            arc_number :
                The arc number with respect to the sorted arcs order.

        type : {'vertex_dir', 'arc_notation', 'arc_order'}
            The type of the ``entries`` argument.

            * `'vertex_dir'` (default): corresponds to
                the `(amplitude, vertex, coin_dir)` entry type;
            * `'arc_notation'` : corresponds to the
                `(amplitude, vertex, dst_vertex)` entry type;
            * `arc_order` : corresponds to the
                `(amplitude, arc)` entry type.


        Raises
        ------
        ValueError
            If ``type`` has invalid value.

            If ``type='arc_notation'`` and there exists an entry such that
            `vertex` and `dst_vertex` are not adjacent.

        IndexError
            If ``type='vertex_dir'`` and `coin_dir` is not a value in
            the valid interval (from 0 to degree(`vertex`) - 1).

        Notes
        -----
        If entries are repeated (except by the amplitude),
        they are overwritten by the last one.

        .. todo::
            * Allow real states (only complex allowed at the moment).
            * More efficient implementation of
                state construction is desirable.
            * Turn simple `entries` in iterable format. For example,
                (1, 0, 0) into [(1, 0, 0)]
        """

        def _normalize(state, error=1e-16):
            norm = np.linalg.norm(state)
            if 1 - error <= norm and norm <= 1 + error:
                return state
            return state / norm
        
        funcs = {'vertex_dir' : self._state_vertex_dir,
                 'arc_notation' : self._state_arc_notation,
                 'arc_order' : self._state_arc_order}

        if type not in list(funcs.keys()):
            raise ValueError(
                    'Invalid `type` argument. Expected any from '
                    + str(list(funcs.keys()))
            )

        state = np.zeros(self.hilb_dim, dtype=complex)
        state = funcs[type](state, entries)

        return _normalize(state)
