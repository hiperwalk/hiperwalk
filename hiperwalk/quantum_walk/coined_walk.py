import numpy as np
import scipy
import scipy.sparse
import networkx as nx
from ..base_walk import BaseWalk
from warnings import warn
from sys import path as sys_path
sys_path.append('../..')
from constants import __DEBUG__, PYNEBLINA_IMPORT_ERROR_MSG

if __DEBUG__:
    from time import time as now

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

class CoinedWalk(QuantumWalk):
    r"""
    Manage an instance of the coined quantum walk model
    on general unweighted graphs.

    Methods for managing, simulating and generating operators of
    the coined quantum walk model for general graphs are available.

    For implementation details, see the Notes Section.

    Parameters
    ----------
    graph
        Graph on which the quantum walk occurs.
        It can be the graph itself (:class:`hiperwalk.graph.Graph`) or
        its adjacency matrix (:class:`scipy.sparse.csr_array`).

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

    The Graph class uses the arc notation
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

    .. testsetup::

        import numpy as np

    >>> adj_matrix = np.array([
    ...     [0, 1, 0, 0],
    ...     [1, 0, 1, 1],
    ...     [0, 1, 0, 1],
    ...     [0, 1, 1, 0]])
    >>> adj_matrix
    array([[0, 1, 0, 0],
           [1, 0, 1, 1],
           [0, 1, 0, 1],
           [0, 1, 1, 0]])

    .. graphviz:: ../../graphviz/coined-model-sample.dot
        :align: center
        :layout: neato
        :caption: Figure 1

    The corresponding arcs are

    >>> arcs = [(i, j) for i in range(4)
    ...                for j in range(4) if adj_matrix[i,j] == 1]
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

    >>> adj_labels = [[arcs_labels[(i,j)] if (i,j) in arcs_labels
    ...                                   else '' for j in range(4)]
    ...               for i in range(4)]
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

    >>> psi = np.array([1/np.sqrt(2), 0, 1j/np.sqrt(2), 0, 0, 0, 0, 0])
    >>> psi
    array([0.70710678+0.j        , 0.        +0.j        ,
           0.        +0.70710678j, 0.        +0.j        ,
           0.        +0.j        , 0.        +0.j        ,
           0.        +0.j        , 0.        +0.j        ])

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

    def __init__(self, graph=None):

        if graph is None:
            raise ValueError('graph is None')

        super().__init__(graph)
        self._shift_operator = None
        self._coin_operator = None

        # Expects adjacency matrix with only 0 and 1 as entries
        self.hilb_dim = self._graph.adj_matrix.sum()

        self._valid_kwargs = dict()

        self._valid_kwargs['shift'] = self._get_valid_kwargs(
            self.shift_operator)

        self._valid_kwargs['coin'] = self._get_valid_kwargs(
            self.coin_operator)

        self._valid_kwargs['oracle'] = self._get_valid_kwargs(
            self.oracle)

        if __DEBUG__:
            methods = list(self._valid_kwargs)
            params = [p for m in methods
                        for p in self._valid_kwargs[m]]
            if len(params) != len(set(params)):
                raise AssertionError

    def _flipflop_shift(self):
        r"""
        Create the flipflop shift operator (:math:`S`) based on
        the ``adj_matrix`` attribute.

        The operator is set for future usage.
        If an evolution operator was set previously,
        it is unset for coherence.

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
            Check :class:`Graph` Notes for details
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
        :class:`Graph` Notes Section example.
        The corresponding flip-flop shift operator is

        .. testsetup::

            import numpy as np
            import scipy.sparse

        .. doctest::

            >>> import qwalk.coined as cnqw
            >>> A = scipy.sparse.csr_array([[0, 1, 0, 0],
            ...                             [1, 0, 1, 1],
            ...                             [0, 1, 0, 1],
            ...                             [0, 1, 1, 0]])
            >>> g = cnqw.Graph(A)
            >>> S = g.flipflop_shift_operator()
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

        .. doctest::

            >>> (Sd @ Sd == np.eye(8)).all() # True by definition
            True
            >>> Sd @ np.array([1, 0, 0, 0, 0, 0, 0, 0]) # S|0> = |1>
            array([0, 1, 0, 0, 0, 0, 0, 0])
            >>> Sd @ np.array([0, 1, 0, 0, 0, 0, 0, 0]) # S|1> = |0>
            array([1, 0, 0, 0, 0, 0, 0, 0])
            >>> Sd @ np.array([0, 0, 1, 0, 0, 0, 0, 0]) # S|2> = |4>
            array([0, 0, 0, 0, 1, 0, 0, 0])
            >>> Sd @ np.array([0, 0, 0, 0, 1, 0, 0, 0]) # S|4> = |2>
            array([0, 0, 1, 0, 0, 0, 0, 0])
        """

        if __DEBUG__:
            start_time = now()

        # expects weights to be 1 if adjacent
        num_edges = self.adj_matrix.sum()

        # Storing edges' indeces in data.
        # Obs.: for some reason this does not throw exception,
        #   so technically it is a sparse matrix that stores a zero entry
        orig_dtype = self.adj_matrix.dtype
        self.adj_matrix.data = np.arange(num_edges)

        # Calculating flipflop_shift columns
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

        if __DEBUG__:
            print("flipflop_shift Time: "
                  + str(now() - start_time))

        self._shift_operator = S
        self._evolution_operator = None
        return S

    def has_persistent_shift(self):
        r"""
        Returns if the persistent shift operator is defined
        for the current graph.

        The persistent shift operator is only defined for specific graphs
        that can be embedded into the plane.
        Hence, a direction can be inferred --
        e.g. left, right, up, down.
        """
        return self._graph.is_embeddable()

    def _persistent_shift(self):
        raise NotImplementedError('Inexistent or not overriden.')


    def shift(self, shift='default'):
        r"""
        Create the shift operator.

        Create either the flipflop or the persistent shift operator.

        The created shift operator is saved to be used
        for generating the evolution operator.
        If an evolution operator was set previously,
        it is unset for coherence.

        Parameters
        ----------
        shift: {'default', 'flipflop', 'persistent', 'ff', 'p'}
            Whether to create the flip flop or the persistent shift.
            By default, creates the persistent shift if it is defined;
            otherwise creates the flip flop shift.
            Argument ``'ff'`` is an alias for ``'flipflop'``.
            Argument ``'p'`` is an alias for ``'persistent'``.

        Raises
        ------
        AttributeError
            If ``shift='persistent'`` and
            the persistent shift operator is not implemented.

        Returns
        -------
        :class:`scipy.sparse.csr_matrix`
            Shift operator

        See Also
        --------
        has_persistent_shift_operator
        """
        valid_keys = ['default', 'flipflop', 'persistent', 'ff', 'p']
        if shift not in valid_keys:
            raise ValueError(
                "Invalid `shift` value. Expected one of "
                + str(valid_keys) + ". But received '"
                + str(shift) + "' instead."
            )

        if shift == 'default':
            shift = 'p' if self.has_persistent_shift_operator() else 'ff'

        if shift == 'ff':
            shift = 'flipflop'
        elif shift == 'p':
            shift = 'persistent'

        
        S = (self._flipflop_shift() if shift == 'flipflop'
             else self._persistent_shift())

        if __DEBUG__:
            if self._shift_operator is None: raise AssertionError
            if self._evolution_operator is not None: raise AssertionError

        return S

    def coin_operator(self, coin='default', coin2=None, vertices2=[]):
        """
        Generate a coin operator based on the graph structure.

        Constructs coin operator depending on the degree of each vertex.
        A single coin type may be applied to all vertices
        (``coin2 is None``),
        or two coins may be applied to selected vertices
        (``coin2 is not None``).

        Parameters
        ----------
        coin : {'default', 'fourier', 'grover', 'hadamard', 'minus_identity'}
            Type of the coin to be used.

        coin2 : default=None
            Type of the coin to be used for ``vertices2``.
            Accepts the same inputs as ``coin``.

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

        See Also
        --------
        get_default_coin

        Notes
        -----
        Due to the chosen computational basis
        (see :class:`Graph` Notes),
        the resulting operator is a block diagonal where
        each block is the :math:`\deg(v)`-dimensional ``coin``.
        Consequently, there are :math:`|V|` blocks.

        """
        # dict with valid coins as keys and the respective
        # function pointers.
        coin_funcs = {
            'fourier': Graph._fourier_coin,
            'grover': Graph._grover_coin,
            'hadamard': Graph._hadamard_coin,
            'minus_identity': Graph._minus_identity
        }

        if coin == 'default':
            coin = self.get_default_coin()
        if coin2 == 'default':
            coin2 = self.get_default_coin()

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

        C = scipy.sparse.block_diag(blocks, format='csr')
        self._coin_operator = C
        self._evolution_operator = None
        return C

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

    def set_marked(self, vertices=None, change_coin='minus_identity'):
        raise NotImplementedError

    def get_marked(self):
        return self._marked

    def evolution_operator(self, marked_vertices=[], hpc=True, **kwargs):
        """
        Create the standard evolution operator.

        The evolution operator is created using the
        shift operator, coin operator and oracle.
        If an operator was set previously, it is used unless
        ``**kwargs`` specifies arguments for creating new ones.
        If an operator was not set previously,
        it is created using ``**kwargs`` and ``marked_vertices``
        accordingly.
        In this case, if ``**kwargs`` is empty,
        the default arguments are used.

        The created evolution operator is set to be used in the
        quantum walk simulation.

        Parameters
        ----------
        marked_vertices : array_like, default=[]
            The marked vertices.
            See :obj:`oracle`'s ``marked_vertices`` parameter.

        hpc : bool, default=True
            Whether or not to use neblina core to
            generate the evolution operator.

        **kwargs : dict, optional
            Additional arguments for constructing the evolution operator.
            Accepts any valid keywords from
            :meth:`shift_operator`
            :meth:`coin_operator`, and
            :meth:`oracle`.

        Returns
        -------
        U : :class:`scipy.sparse.csr_array`
            The evolution operator.

        See Also
        --------
        has_persistent_shift_operator
        shift_operator
        coin_operator
        oracle

        Notes
        -----
        The evolution operator is given by

        .. math::
            U = SCR

        where :math`S` is the shift operator,
        :math:`C` is the coin operator, and
        :math:`R` is the oracle [1]_.

        References
        ----------
        .. [1] Portugal, Renato. "Quantum walks and search algorithms".
            Vol. 19. New York: Springer, 2013.

        """

        S_kwargs = self._filter_valid_kwargs(
            kwargs, self._valid_kwargs['shift'])
        C_kwargs = self._filter_valid_kwargs(
                    kwargs, self._valid_kwargs['coin'])
        R_kwargs = self._filter_valid_kwargs(
                    kwargs, self._valid_kwargs['oracle'])
        R_kwargs['marked_vertices'] = marked_vertices

        if self._shift_operator is None or bool(S_kwargs):
            self.shift_operator(**S_kwargs)
        if self._coin_operator is None or bool(C_kwargs):
            self.coin_operator(**C_kwargs)
        if self._oracle is None or bool(R_kwargs):
            self.oracle(**R_kwargs)
        U = self._evolution_operator_from_SCR(hpc)


        if __DEBUG__:
            if (self._shift_operator is None
                or self._coin_operator is None
                or (self._oracle is None and len(marked_vertices) != 0)
                or self._evolution_operator is None
            ):
                raise AssertionError

        return U

    def _evolution_operator_from_SCR(self, hpc=True):
        r"""
        Create evolution operator from previously set matrices.

        Creates evolution operator by multiplying the
        shift operator, coin operator and oracle.
        If the oracle is not set,
        it is substituted by the identity.

        Parameters
        ----------
        hpc : bool, default=True
            Whether or not the evolution operator should be
            constructed using nelina's high-performance computating.
        """
        if self._shift_operator is None:
            raise AttributeError("Shift operator was not set.")
        if self._coin_operator is None:
            raise AttributeError("Coin operator was not set.")

        U = None
        if hpc and not self._pyneblina_imported():
            try:
                from .. import _pyneblina_interface as nbl
            except ModuleNotFoundError:
                warn(PYNEBLINA_IMPORT_ERROR_MSG)
                hpc = False

        if hpc:

            warn(
                "Sparse matrix multipliation is not supported yet. "
                + "Converting all matrices to dense. "
                + "Then converting back to sparse. "
                + "This uses unnecessary memory and computational time."
            )
            S = self._shift_operator.todense()
            C = self._coin_operator.todense()

            warn("CHECK IF MATRIX IS SPARSE IN PYNELIBNA INTERFACE")
            nbl_S = nbl.send_matrix(S)
            nbl_C = nbl.send_matrix(C)
            nbl_U = nbl.multiply_matrices(nbl_S, nbl_C)

            warn("Check if matrices are deleted "
                          + "from memory and GPU.")
            del S
            del C
            del nbl_S
            del nbl_C

            if self._oracle is not None:
                R = self._oracle.todense()
                nbl_R = nbl.send_matrix(R)
                nbl_U = nbl.multiply_matrices(nbl_U, nbl_R)
                del R

            U = nbl.retrieve_matrix(nbl_U)
            U = scipy.sparse.csr_array(U)

        else:
            U = self._shift_operator @ self._coin_operator
            if self._oracle is not None:
                U = U @ self._oracle

        self._evolution_operator = U
        return U


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
        simulate
        """
        # TODO: test with nonregular graph
        # TODO: test with nonuniform condition
        if __DEBUG__:
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
                Graph._elementwise_probability(
                    states[i][edges_indices[j]:edges_indices[j + 1]]
                ).sum()
                for j in range(len(edges_indices) - 1)
            ] for i in range(len(states)) ])

        if __DEBUG__:
            end = now()
            print("probability_distribution: " + str(end - start) + 's')
        # TODO: benchmark (time and memory usage)
        return prob


    #########################################################
    ######### Auxiliary Methods for state() method ##########
    #########################################################

    def _state_position_coin(self, state, entries):
        raise NotImplementedError

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


    def _state_arc_label(self, state, entries):
        for amplitude, arc in entries:
            state[arc] = amplitude

        return state

    def state(self, entries, type='arc_notation'):
        """
        Generates a valid state.

        The state corresponds to the walker being in a superposition
        of the ``entries``.
        Please refer to the current class documentation for the
        expected direction and arc order --
        for instance, click on :meth:`qwalk.coined.Graph`.

        The final state is normalized in order to be unitary.

        Parameters
        ----------
        entries : list of entry
            Each entry is a tuple (or array).
            An entry can be specified in four different ways:
            ``(amplitude, vertex, dst_vertex)``,
            ``(amplitude, vertex, coin)``,
            ``(amplitude, coin, vertex)``,
            ``(amplitude, arc_label)``.

            amplitude :
                The amplitudes of the given entry.
            vertex :
                The vertex corresponding to the position of the walker
                in the superposition.
            dst_vertex : 
                The vertex which the coin is pointing to.
                In other words, the tuple
                (vertex, dst_vertex) must be a valid arc.
            coin :
                The direction towards which the coin is pointing.
                A value between 0 and degree(``vertices[i]``) - 1
                is expected, respecting the sorted arcs order.
            arc_label :
                The arc number with respect to the sorted arcs order.

        type : {'arc_notation', 'vertex_dir', 'arc_order'}
            The type of each entry sent in the ``entries`` argument.

            * **'arc_notation'** : ``(amplitude, vertex, dst_vertex)``;
            * **'position_coin'**: ``(amplitude, vertex, coin)``;
            * **'coin_position'**: ``(amplitude, coin, vertex)``;
            * **arc_order** : ``(amplitude, arc_label)``.


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

        funcs = {'arc_notation' : self._state_arc_notation,
                 'position_coin' : self._state_vertex_dir,
                 'arc_order' : self._state_arc_order}

        if type not in list(funcs.keys()):
            raise ValueError(
                    'Invalid `type` argument. Expected any from '
                    + str(list(funcs.keys()))
            )

        state = np.zeros(self.hilb_dim, dtype=complex)
        state = funcs[type](state, entries)

        return self._normalize(state)
