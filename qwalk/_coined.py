import numpy as np
import scipy
from scipy.linalg import block_diag as scipy_block_diag
import networkx
from constants import DEBUG

if DEBUG:
    from time import time as now
    from guppy import hpy # used to check memory usage


class Coined:
    r"""
    Manage an instance of the general coined quantum walk model.

    Methods for managing, simulating and generating operators of
    the coined quantum walk model for general and
    specific graphs are available.

    For implementation details, see the Notes Section.

    Parameters
    ----------
    adj_matrix : :class:`scipy.sparse.csr_array`
        Adjacency matrix of the graph on
        which the quantum walk occurs.

    Notes
    -----
    The preferable parameter type is

    #. scipy.sparse.array using ``dtype=np.int8``
    #. numpy.array using ``dtype=np.int8``
    #. networkx graph

    .. todo::
        * Add option: numpy dense matrix as parameters.
        * Add option: networkx graph as parameter.

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


    References
    ----------
    .. [1] Portugal, Renato. "Quantum walks and search algorithms".
        Vol. 19. New York: Springer, 2013.
    """

    def __init__(self, adj_matrix):
        self._initial_condition = None
        self._evolution_operator = None
        self._num_steps = 0
        # TODO: create sparse matrix from graph or dense adjacency matrix
        self.adj_matrix = adj_matrix

    def uniform_state(self):
        hilb_dim = self.adj_matrix.sum()
        return np.ones(hilb_dim, dtype=float)/np.sqrt(hilb_dim)

    def flip_flop_shift_operator(self, adj_matrix):
        r"""
        Creates flip-flop shift operator (:math:`S`) based on
        an adjacency matrix.

        Parameters
        ----------
        adj_matrix : :class:`scipy.sparse.csr_matrix`
            Adjacency Matrix of an unweighted undirected graph.

        Returns
        -------
        :class:`scipy.sparse.csr_matrix`
            Flip-flop shift operator.

        Notes
        -----

        .. todo::
            - If `adj_matrix` parameter is not sparse,
                throw exception of convert to sparse.
            - Change :class:`scipy.sparse.csr_matrix` to
                :class:`scipy.sparse.csr_array`.

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

        >>> from scipy.sparse import csr_matrix
        >>> import CoinedModel as qcm
        >>> A = csr_matrix([[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]])
        >>> S = qcm.flip_flop_shift_operator(A)
        >>> Sd = S.todense()
        >>> Sd
        matrix([[0, 1, 0, 0, 0, 0, 0, 0],
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

        num_edges = adj_matrix.sum() # expects weights to be 1 if adjacent

        # Storing edges' indeces in data.
        # Obs.: for some reason this does not throw exception,
        #   so technically it is a sparse matrix that stores a zero entry
        orig_dtype = adj_matrix.dtype
        adj_matrix.data = np.arange(num_edges)

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
        # (to be used as indices of a csr_matrix)
        row = 0
        S_cols = np.zeros(num_edges)
        for edge in range(num_edges):
            if edge >= adj_matrix.indptr[row + 1]:
                row += 1
            # Column index (in the adj_matrix struct) of the current edge
            col_index = __binary_search(adj_matrix.data, edge,
                                        start=adj_matrix.indptr[row],
                                        end=adj_matrix.indptr[row+1])
            # S[edge, S_cols[edge]] = 1
            # S_cols[edge] is the edge_id such that
            # S|edge> = |edge_id>
            S_cols[edge] = adj_matrix[adj_matrix.indices[col_index], row]

        # Using csr_matrix((data, indices, indptr), shape)
        # Note that there is only one entry per row and column
        S = scipy.sparse.csr_matrix(
            ( np.ones(num_edges, dtype=np.int8),
              S_cols, np.arange(num_edges+1) ),
            shape=(num_edges, num_edges)
        )

        # restores original data to adj_matrix
        adj_matrix.data = np.ones(num_edges, dtype=orig_dtype)

        # TODO: compare with old approach for creating S

        if DEBUG:
            print("flip_flop_shift_operator Memory: "
                  + str(hpy().heap().size))
            print("flip_flop_shift_operator Time: "
                  + str(now() - start_time))

        return S

    def coin_operator(self, adj_matrix, coin='grover'):
        n = adj_matrix.shape[0]
        G = networkx.from_numpy_matrix(adj_matrix)
        if coin == 'grover':
            L = [self.grover_operator(G.degree(i)) for i in range(n)]
        elif coin == 'hadamard':
            L = [self.hadamard_operator() for i in range(n)]
        else:
            return None
        return scipy.sparse.csr_matrix(scipy_block_diag(*L))

    def grover_operator(self, N):
        return np.matrix(2/N*np.ones(N) - np.identity(N))

    def hadamard_operator(self):
        return 1/np.sqrt(2) * np.matrix([[1, 1], [1, -1]])

    def oracle(self, N):
        """
        Create the oracle that marks the first element (vertex 0)
        """
        R = np.identity(N)
        R[0,0] = -1
        return np.matrix(R)

    def evolution_operator(self, adj_matrix, coin=None):
        # TODO: should these matrix multiplication be performed by neblina?
        if coin is None:
            return (self.flip_flop_shift_operator(adj_matrix)
                    @ self.coin_operator(adj_matrix))
        return self.flip_flop_shift_operator(adj_matrix) @ coin

    def search_evolution_operator(self, adj_matrix):
        """
        Creates the search evolution operator for the graph described by a
        given adjacency matrix.

        Parameters
        ----------
        adj_matrix : :class:`scipy.sparse.csr_matrix`
            Adjacency matrix of the graph where the walk is performed.

        Returns
        -------
        :class:`scipy.sparse.csr_matrix`
            Search evolution operator

        See Also
        --------
        evolution_operator
        oracle


        Notes
        -----
        The search evolution operator is

        .. math::
            U = U_w R

        where :math:`U_w` is the coined quantum walk evolution operator
        and :math:`R` is the oracle [1]_.

        References
        ----------
        .. [1] Portugal, Renato. "Quantum walks and search algorithms".
            Vol. 19. New York: Springer, 2013.
        """
        S = self.flip_flop_shift_operator(adj_matrix)
        C = self.coin_operator(adj_matrix)
        N = S.shape[0]
        # TODO: should this matrix multiplication be performed by neblina?
        return S @ C @ self.oracle(N)

    # TODO: check numpy vectorize documentation
    # TODO: move to auxiliary functions?
    # TODO: test with complex state
    @staticmethod
    def __elementwise_probability(elem):
        # this is more efficient than:
        #(np.conj(elem) * elem).real
        # elem.real**2 + elem.imag**2
        return elem.real*elem.real + elem.imag*elem.imag

    # TODO: documentation
    # TODO: test with nonregular graph
    # TODO: test with nonuniform condition
    def probability_distribution(self, adj_matrix, states):
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
            Operator that describes the quantum walk

        initial_coidition
            The initial state

        num_steps : int
            Numbert of times to apply the ``evolution_operator`` on
            the ``initial_condition``

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

    # Simulating walk. Needed: U, state, stop_steps
    # num_steps: int. Number of iterations to be simulated,
    # i.e. U^num_steps |initial_state>
    # save_interval: int. Number of steps to execute before
    # saving the state.
    #   For example if num_steps = 10 and save_interval = 5,
    #   the states at iterations
    #   5 and 10 will be saved and case save_interval = 3,
    #   the states at iterations
    #   3, 6, 9, and 10 will be saved.
    #   Default: None, i.e. saves only the final state
    # save_initial_condition: boolean.
    #   If True, adds the initial condition into the saved states.
    # returns array with saved states
    def simulate_walk(self, U, initial_state, num_steps,
                      save_interval=None, save_initial_state=False):
        from . import _pyneblina_interface as nbl
        # preparing walk
        nbl_matrix = nbl.send_sparse_matrix(U)
        nbl_vec = nbl.send_vector(initial_state)

        # number of states to save
        num_states = (int(np.ceil(num_steps/save_interval))
                      if save_interval is not None else 1)
        if save_initial_state:
            num_states += 1
        save_final_state = (save_interval is None
                            or num_steps % save_interval != 0)

        # TODO: change dtype accordingly
        saved_states = np.zeros(
            (num_states, initial_state.shape[0]), dtype=complex
        )
        state_index = 0 # index of the state to be saved
        if save_initial_state:
            saved_states[0] = initial_state
            state_index += 1

        # simulating walk
        # TODO: request multiple multiplications at once to neblina-core
        # TODO: check if intermediate states are being freed from memory
        for i in range(1, num_steps + 1):
            # TODO: request to change parameter order
            nbl_vec = nbl.multiply_sparse_matrix_vector(
                nbl_matrix, nbl_vec)

            if save_interval is not None and i % save_interval == 0:
                saved_states[state_index] = nbl.retrieve_vector(
                    nbl_vec, initial_state.shape[0], delete_vector=True
                )
                state_index += 1

        if save_final_state:
            saved_states[state_index] = nbl.retrieve_vector(
                nbl_vec, initial_state.shape[0], delete_vector=True
            )

        return saved_states
