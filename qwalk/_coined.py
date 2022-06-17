import numpy as np
import scipy
import scipy.sparse
import networkx as nx
from constants import DEBUG

if DEBUG:
    from time import time as now
    from guppy import hpy # used to check memory usage


class Coined:
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

    The Coined class uses the position-coin notation
    and the Hilbert space :math:`\mathcal{H}^{2|E|}` for general Graphs.
    Matrices and states respect the sorted edges order,
    i.e. :math:`(v, u) < (v', u')` if either :math:`v < v'` or
    :math:`v = v'` and :math:`u < u'`
    where :math:`(v, u), (v', u')` are valid edges.

    For example, the graph :math:`G(V, E)` shown in Figure 1 has adjacency matrix `A`.

    >>> import numpy as np
    >>> A = np.matrix([[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]])
    >>> A
    matrix([[0, 1, 0, 0],
            [1, 0, 1, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 0]])

     .. graphviz:: ../../graphviz/coined-model-sample.dot
        :align: center
        :layout: neato
        :caption: Figure 1

    Letting :math:`(v, u)` denote the edge from vertex :math:`v` to :math:`u`,
    the `edges` of :math:`G` are

    >>> edges = [(i, j) for i in range(4) for j in range(4) if A[i,j] == 1]
    >>> edges
    [(0, 1), (1, 0), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]

    Note that `edges` is already sorted, hence the labels are

    >>> edges_labels = {edges[i]: i for i in range(len(edges))}
    >>> edges_labels
    {(0, 1): 0, (1, 0): 1, (1, 2): 2, (1, 3): 3, (2, 1): 4, (2, 3): 5, (3, 1): 6, (3, 2): 7}

    The edges labels are illustrated in Figure 2.

    .. graphviz:: ../../graphviz/coined-model-edges-labels.dot
        :align: center
        :layout: neato
        :caption: Figure 2

    If we would write the edges labels respecting the the adjacency matrix format,
    we would have the matrix `A_labels`.
    Intuitively, the edges are labeled in left-to-right top-to-bottom fashion.

    >>> A_labels = [[edges_labels[(i,j)] if (i,j) in edges_labels else '' for j in range(4)]
    ...             for i in range(4)]
    >>> A_labels = np.matrix(A_labels)
    >>> A_labels
    matrix([['', '0', '', ''],
            ['1', '', '2', '3'],
            ['', '4', '', '5'],
            ['', '6', '7', '']], dtype='<U21')

    For consistency, any state :math:`\ket\psi \in \mathcal{H}^{2|E|}`
    is such that :math:`\ket\psi = \sum_{i = 0}^{2|E| - 1} \psi_i \ket{i}`
    where :math:`\ket{i}` is the computational basis state
    associated to the :math:`i`-th edge.
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
        self._initial_condition = None
        self._evolution_operator = None
        self._num_steps = 0

        ##############################
        ### Simulation attributes. ###
        ##############################
        # Matrix object used during simulation.
        # It may by a scipy matrix or a neblina matrix.
        # Should be different from None during simulation only.
        self._simul_mat = None
        # Vector object used during simulation.
        # Should be different from None during simulation only.
        self._simul_vec = None

        # TODO: create sparse matrix from graph or dense adjacency matrix
        if isinstance(adj_matrix, scipy.sparse.csr_array):
            self.adj_matrix = adj_matrix
        else:
            raise TypeError(
                "Invalid 'adj_matrix' type."
                + " Expected 'scipy.sparse.csr_array',"
                + " but received " + str(type(adj_matrix)) + '.'
            )

        # Expects adjacency matrix with only 0 and 1 as entries
        self.hilb_dim = self.adj_matrix.sum()

    def uniform_state(self):
        r"""
        Generate the uniform state.

        The state is constructed based on the ``adj_matrix`` attribute.
        The uniform initial condition is the state where
        all entries have the same amplitude.

        Returns
        -------
        :obj:`numpy.ndarray`

        Notes
        -----
        The uniform initial condition is the state

        .. math::

            \ket{d} = \frac{1}{\sqrt{N}} \sum_{i = 0}^{N - 1} \ket{i}

        where :math:`N` is the dimension of the Hilbert space.
        For a graph :math:`G(V, E)`,
        in the general case, :math:`N = 2|E|`.

        Examples
        --------
        >>> # Importing and generating adj_matrix
        >>> # of Coined Notes example
        >>> coined_model = hpw.Coined(adj_matrix)
        >>> coined_model.uniform_state()
        array([0.35355339, 0.35355339, 0.35355339, 0.35355339, 0.35355339,
               0.35355339, 0.35355339, 0.35355339])
        """
        return (np.ones(self.hilb_dim, dtype=float)
                / np.sqrt(self.hilb_dim))

    def uniform_initial_condition(self):
        """
        Alias for :obj:`uniform_state`.
        """
        return self.uniform_state()

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

        # expects sorted array and executes binary search in the subarray
        # v[start:end] searching for elem.
        # Return the index of the element if found, otherwise returns -1
        # Cormen's binary search implementation.
        # Used to improve time complexity
        def __binary_search(v, elem, start=0, end=None):
            if end == None:
                end = len(v)
            
            while start < end:
                mid = int((start + end)/2)
                if elem <= v[mid]:
                    end = mid
                else:
                    start = mid + 1

            return end if v[end] == elem else -1

        # Calculating flip_flop_shift columns
        # (to be used as indices of a csr_array)
        row = 0
        S_cols = np.zeros(num_edges)
        for edge in range(num_edges):
            if edge >= self.adj_matrix.indptr[row + 1]:
                row += 1
            # Column index (in the adj_matrix struct) of the current edge
            col_index = __binary_search(self.adj_matrix.data, edge,
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

    def evolution_operator(self, coin='grover', hpc=False):
        """
        Create the standard evolution operator.

        Parameters
        ----------
        coin : str, default='grover'
            The coin to be used as diffusion operator.
            See :obj:`coin_operator`'s ``coin``
            attribute for valid options.

        hpc : bool, default=False
            Whether or not evolution operator should be
            constructed using nelina's high-performance computating.

        Returns
        -------
        U_w : :class:`scipy.sparse.csr_array` or
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

    def search_evolution_operator(self, vertices, coin='grover',
                                  hpc=False):
        """
        Create the search evolution operator.

        Parameters
        ----------
        vertices : array_like
            The marked vertex (vertices) IDs.
            See :obj:`oracle`'s ``vertices`` parameter.

        coin : str, default='grover'
            The coin to be used as diffusion operator.
            See :obj:`evolution_operator` and :obj:`coin_operator`.

        hpc : bool, default=False
            Whether or not evolution operator should be
            constructed using nelina's high-performance computating.

        Returns
        -------
        U : :class:`scipy.sparse.csr_array` or
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

    @staticmethod
    def __elementwise_probability(elem):
        # this is more efficient than:
        #(np.conj(elem) * elem).real
        # elem.real**2 + elem.imag**2
        return elem.real*elem.real + elem.imag*elem.imag

    def probability_distribution(self, adj_matrix, states):
        # TODO: test with nonregular graph
        # TODO: test with nonuniform condition
        if DEBUG:
            start = now()

        if len(states.shape) == 1:
            states = [states]

        # TODO: check if dimensions match and throw exception if necessary
        # TODO: check if just creates reference (no hard copy)
        edges_indices = adj_matrix.indptr 

        # TODO: check it is more efficient on demand or
        # using extra memory (aux_prob)
        # aux_prob = ElementwiseProbability(state)
        # first splits state per vertex,
        # then calculates probability of each vertex direction,
        # then sums the probabilities resulting in
        # the vertix final probability
        prob = np.array([[
                Coined.__elementwise_probability(
                    states[i][edges_indices[j]:edges_indices[j + 1]]
                ).sum()
                for j in range(len(edges_indices) - 1)
            ] for i in range(len(states)) ])

        if DEBUG:
            end = now()
            print("probability_distribution: " + str(end - start) + 's')
        # TODO: benchmark (time and memory usage)
        return prob

    def prepare_walk(self, evolution_operator,
                     initial_condition, num_steps):
        """
        Set all information needed for simulating a quantum walk.

        Parameters
        ----------
        evolution_operator
            Operator that describes the quantum walk.

        initial_coidition
            The initial state.

        num_steps : int
            Number of times to apply the ``evolution_operator`` on
            the ``initial_condition``.

        See Also
        --------
        simulate_walk

        Notes
        -----
        .. todo::
            Implement assertion of arguments.
            For example: check if evolution operator is unitary.
        """

        self._evolution_operator = evolution_operator
        self._initial_condition = initial_condition
        self._num_steps = num_steps
        # TODO: add initial_condition in states matrix
        # and create states matrix as np.zeros

    def simulate_walk(self, save_interval=0, hpc=False):
        r"""
        Simulates quantum walk.
        
        It is necessary to call :obj:`prepare_walk` beforehand.

        Parameters
        ----------
        save_interval : int, default=0
            Number of applications of the evolution operation
            before saving an intermediate state.
            If ``save_interval=0``, returns only the final state.
            Otherwise, returns the initial state, the intermediate
            states and the final state.
        hpc : bool, default=False
            Whether or not to use neblina's high-performance computing
            to perform matrix multiplications.
            If ``hpc=False`` uses python.

        Returns
        -------
        returns array with saved states

        See Also
        --------
        prepare_walk

        Examples
        --------
        If ``num_steps=10`` and ``save_interval=3``,
        the returned saved states are:
        the initial state, the intermediate states (3, 6, and 9),
        and the final state (10).

        >>> cqw.prepare_walk(U, psi0, 10)
        >>> cqw.simulate_walk(save_interval=3)
        """

        if hpc:
            from . import _pyneblina_interface as nbl

        def _prepare_engine(self):
            if hpc:
                self._simul_mat = nbl.send_sparse_matrix(
                    self._evolution_operator)
                self._simul_vec = nbl.send_vector(
                    self._initial_condition)

            else:
                self._simul_mat = self._evolution_operator
                self._simul_vec = self._initial_condition

        def _simulate_steps(self, num_steps):
            if hpc:
                # TODO: request multiple multiplications at once
                #       to neblina-core
                # TODO: check if intermediate states are being freed
                for i in range(num_steps):
                    self._simul_vec = nbl.multiply_sparse_matrix_vector(
                        self._simul_mat, self._simul_vec)
            else:
                for i in range(num_steps):
                    self._simul_vec = self._simul_mat @ self._simul_vec

                # TODO: compare with numpy.linalg.matrix_power

        def _save_simul_vec(self):
            if hpc:
                # TODO: check if vector must be deleted or
                #       if it can be reused via neblina-core commands.
                return nbl.retrieve_vector(
                    self._simul_vec, self._initial_condition.shape[0],
                    delete_vector=True
                )
            return self._simul_vec

        _prepare_engine(self)

        # number of states to save
        num_states = (int(np.ceil(self._num_steps / save_interval))
                      if save_interval >= 0 else 1)
        if save_interval > 0:
            # saves initial state
            num_states += 1

        # create saved states matrix
        dtype = (self._initial_condition.dtype if
            self._initial_condition.dtype == self._evolution_operator.dtype
            else complex
        )
        saved_states = np.zeros(
            (num_states, self._initial_condition.shape[0]), dtype=dtype
        )
        state_index = 0 # index of the state to be saved

        # if save_initial_state:
        if save_interval > 0:
            saved_states[0] = self._initial_condition
            state_index += 1
        else:
            save_interval = 1 # saves only final state

        # simulate walk / apply evolution operator
        for i in range(int(self._num_steps / save_interval)):
            _simulate_steps(self, save_interval)
            saved_states[state_index] = _save_simul_vec(self)
            state_index += 1

        if self._num_steps % save_interval > 0:
            _simulate_steps(self, self._num_steps % save_interval)
            saved_states[state_index] = _save_simul_vec(self)

        # TODO: free vector from neblina core
        self._simul_mat = None
        self._simul_vec = None

        return saved_states

    def plot_probability(self, **kwargs):
        print("automatically call plot_probability distribution")
        return None

    # TODO:
    def plot_states(self):
        # function to plot each state and the direction it is pointing to.
        return None
