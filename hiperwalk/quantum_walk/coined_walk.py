import numpy as np
import scipy
import scipy.sparse
import networkx as nx
from .quantum_walk import QuantumWalk
from warnings import warn
from .._constants import __DEBUG__, PYNEBLINA_IMPORT_ERROR_MSG
from scipy.linalg import hadamard, dft

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

    # static attributes.
    # The class must instantiated once, otherwise are dicts are empty.
    # The class must be instantiated so the interpreter knows
    # the memory location for the function pointers.
    _coin_funcs = dict()
    _valid_kwargs = dict()

    def __init__(self, graph=None, **kwargs):

        if graph is None:
            raise ValueError('graph is None')

        super().__init__(graph)
        self._shift = None
        self._coin = None

        # Expects adjacency matrix with only 0 and 1 as entries
        self.hilb_dim = self._graph.adj_matrix.sum()

        if not bool(CoinedWalk._valid_kwargs):
            # assign static attribute
            CoinedWalk._valid_kwargs = {
                'shift': CoinedWalk._get_valid_kwargs(self.set_shift),
                'coin': CoinedWalk._get_valid_kwargs(self.set_coin),
                'marked': CoinedWalk._get_valid_kwargs(self.set_marked)
            }

        # dict with valid coins as keys and the respective
        # function pointers.
        if not bool(CoinedWalk._coin_funcs):
            # assign static attribute
            CoinedWalk._coin_funcs = {
                'fourier': CoinedWalk._fourier_coin,
                'grover': CoinedWalk._grover_coin,
                'hadamard': CoinedWalk._hadamard_coin,
                'identity': CoinedWalk._identity_coin,
                'minus_fourier': CoinedWalk._minus_fourier_coin,
                'minus_grover': CoinedWalk._minus_grover_coin,
                'minus_hadamard': CoinedWalk._minus_hadamard_coin,
                'minus_identity': CoinedWalk._minus_identity_coin
            }

        self.set_evolution(**kwargs)

        if __DEBUG__:
            methods = list(self._valid_kwargs)
            params = [p for m in methods
                        for p in self._valid_kwargs[m]]
            if len(params) != len(set(params)):
                raise AssertionError


    def _set_flipflop_shift(self):
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

        .. note::
            Check :class:`CoinedWalk` Notes for details
            about the computational basis.


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

            >>> import hiperwalk as hpw
            >>> A = scipy.sparse.csr_array([[0, 1, 0, 0],
            ...                             [1, 0, 1, 1],
            ...                             [0, 1, 0, 1],
            ...                             [0, 1, 1, 0]])
            >>> qw = hpw.CoinedWalk(A)
            >>> qw._flipflop_shift()
            >>> S = qw.get_shift().todense()
            >>> S
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

            >>> (S @ S == np.eye(8)).all() # True by definition
            True
            >>> S @ np.array([1, 0, 0, 0, 0, 0, 0, 0]) # S|0> = |1>
            array([0, 1, 0, 0, 0, 0, 0, 0])
            >>> S @ np.array([0, 1, 0, 0, 0, 0, 0, 0]) # S|1> = |0>
            array([1, 0, 0, 0, 0, 0, 0, 0])
            >>> S @ np.array([0, 0, 1, 0, 0, 0, 0, 0]) # S|2> = |4>
            array([0, 0, 0, 0, 1, 0, 0, 0])
            >>> S @ np.array([0, 0, 0, 0, 1, 0, 0, 0]) # S|4> = |2>
            array([0, 0, 1, 0, 0, 0, 0, 0])
        """

        if __DEBUG__:
            start_time = now()

        num_vert = self._graph.number_of_vertices()
        num_arcs = self._graph.number_of_arcs()

        S_cols = [self._graph.arc_label(j, i)
                  for i in range(num_vert)
                  for j in self._graph.neighbors(i)]

        # Using csr_array((data, indices, indptr), shape)
        # Note that there is only one entry per row and column
        S = scipy.sparse.csr_array(
            ( np.ones(num_arcs, dtype=np.int8),
              S_cols, np.arange(num_arcs+1) ),
            shape=(num_arcs, num_arcs)
        )

        if __DEBUG__:
            print("flipflop_shift Time: " + str(now() - start_time))

        self._shift = S
        self._evolution = None

    def has_persistent_shift(self):
        r"""
        Returns if the persistent shift operator is defined
        for the current graph.

        The persistent shift operator is only defined for specific graphs
        that can be embedded into the plane.
        Hence, a direction can be inferred --
        e.g. left, right, up, down.
        """
        return self._graph.embeddable()

    def _set_persistent_shift(self):
        raise NotImplementedError()


    def set_shift(self, shift='default'):
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
        has_persistent_shift
        """
        valid_keys = ['default', 'flipflop', 'persistent', 'ff', 'p']
        if shift not in valid_keys:
            raise ValueError(
                "Invalid `shift` value. Expected one of "
                + str(valid_keys) + ". But received '"
                + str(shift) + "' instead."
            )

        if shift == 'default':
            shift = 'p' if self.has_persistent_shift() else 'ff'

        if shift == 'ff':
            shift = 'flipflop'
        elif shift == 'p':
            shift = 'persistent'

        
        S = (self._set_flipflop_shift() if shift == 'flipflop'
             else self._set_persistent_shift())

        if __DEBUG__:
            if self._shift is None: raise AssertionError
            if self._evolution is not None: raise AssertionError

    def get_shift(self):
        return self._shift

    def set_coin(self, coin='default'):
        """
        Generate a coin operator based on the graph structure.

        Constructs coin operator depending on the degree of each vertex.
        A single coin type may be applied to all or subset of vertices.
        The coin operator is set to be used during the
        evolution operator generation.

        Parameters
        ----------
        coin
            Coin to be used.
            Several types of arguments are acceptable.

            * str : coin type
                Type of the coin to be used.
                The following are valid entries.

                * 'default', 'd' : default coin,
                * 'fourier', 'F' : Fourier coin,
                * 'grover', 'G' : Grover coin,
                * 'hadamard', 'H' : Hadamard coin,
                * 'identity', 'I' : Identity,
                * 'minus_fourier', '-F' : Fourier coin with negative phase,
                * 'minus_grover', '-G' : Grover coin with negative phase,
                * 'minus_hadamard', '-H' : Hadamard coin with negative phase,
                * 'minus_identity', '-I' : Identity with negative phase.

            * list of str
                List of the coin types to be used.
                Expects list with 'number of vertices' entries.

            * dict
                A dictionary with structure
                ``{coin_type : list_of_vertices}``.
                That is, with any valid coin type as key and
                the list of vertices to be applied as values.
                If ``list_of_vertices = []``,
                the respective ``coin_type`` is applied to all vertices
                that were not explicitly listed.

            * :class:`scipy.sparse.csr_array`
                The explicit coin operator.

        Notes
        -----
        Due to the chosen computational basis
        (see :class:`CoinedWalk` Notes),
        the resulting operator is a block diagonal where
        each block is the :math:`\deg(v)`-dimensional ``coin``.
        Consequently, there are :math:`|V|` blocks.

        """
        try:
            if len(coin.shape) != 2:
                raise TypeError('Explicit coin is not a matrix.')

            # explicit coin
            if not scipy.sparse.issparse(coin):
                coin = scipy.sparse.csr_array(coin)

            warn('TODO: Check if coin is valid')
            self._coin = coin
            self._evolution = None
            return

        except AttributeError:
            pass

        def valid_coin_name(coin):
            s = coin
            if s == 'default' or s == 'd':
                s = self._graph.default_coin()
            
            if len(s) <= 2:
                full_name = 'minus_' if len(coin) == 2 else ''
                abbrv = {'F': 'fourier', 'G' : 'grover',
                         'H': 'hadamard', 'I': 'identity'}
                s = full_name + abbrv[s[-1]]

            if s not in self._coin_funcs.keys():
                raise ValueError(
                    'Invalid coin. Expected any of '
                    + str(list(self._coin_funcs.keys())) + ', '
                    + "but received '" + str(coin) + "'."
                )

            return s

        num_vert = self._graph.number_of_vertices()
        coin_list = []

        if isinstance(coin, str):
            coin_list = [valid_coin_name(coin)] * num_vert

        elif isinstance(coin, dict):
            coin_list = [''] * num_vert
            for key in coin:
                coin_name = valid_coin_name(key)
                value = coin[key]
                if value != []:
                    if not hasattr(value, '__iter__'):
                        value = [value]
                    for vertex in value:
                        coin_list[vertex] = coin_name
                else:
                    coin_list = [coin_name if coin_list[i] == ''
                                 else coin_list[i]
                                 for i in range(num_vert)]

            if '' in coin_list:
                raise ValueError('Coin was not specified for all vertices.')

        else:
            #list of coins
            coin_list = list(map(valid_coin_name, coin))


        self._coin = coin_list
        self._evolution = None

    def _coin_to_list(self, coin):
        r"""
        Convert str, list of str or dict to valid coin list.

        See Also
        ------
        set_coin
        """
        return coin_list

    @staticmethod
    def _fourier_coin(dim):
        return dft(dim, scale='sqrtn')

    @staticmethod
    def _grover_coin(dim):
        return np.array(2/dim * np.ones((dim, dim)) - np.identity(dim))

    @staticmethod
    def _hadamard_coin(dim):
        return hadamard(dim) / np.sqrt(dim)

    @staticmethod
    def _identity_coin(dim):
        return np.identity(dim)

    @staticmethod
    def _minus_fourier_coin(dim):
        return -CoinedWalk._fourier_coin(dim)

    @staticmethod
    def _minus_grover_coin(dim):
        return -CoinedWalk._grover_coin(dim)

    @staticmethod
    def _minus_hadamard_coin(dim):
        return -CoinedWalk._hadamard_coin(dim)

    @staticmethod
    def _minus_identity_coin(dim):
        return -np.identity(dim)

    def get_coin(self):
        if not scipy.sparse.issparse(self._coin):
            num_vert = self._graph.number_of_vertices()
            coin_list = self._coin
            degree = self._graph.degree
            blocks = [self._coin_funcs[coin_list[v]](degree(v))
                      for v in range(num_vert)]
            C = scipy.sparse.block_diag(blocks, format='csr')
            return C
        return self._coin


    def set_marked(self, vertices=[], change_coin='minus_identity'):
        self._marked = vertices

    def set_evolution(self, **kwargs):
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
        has_persistent_shift
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

        S_kwargs = CoinedWalk._filter_valid_kwargs(
                              kwargs,
                              CoinedWalk._valid_kwargs['shift'])
        C_kwargs = CoinedWalk._filter_valid_kwargs(
                              kwargs,
                              CoinedWalk._valid_kwargs['coin'])
        R_kwargs = CoinedWalk._filter_valid_kwargs(
                              kwargs,
                              CoinedWalk._valid_kwargs['marked'])

        self.set_shift(**S_kwargs)
        self.set_coin(**C_kwargs)
        self.set_marked(**R_kwargs)

    def get_evolution(self, hpc=True):
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
        if self._shift is None:
            raise AttributeError("Shift operator was not set.")
        if self._coin is None:
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
            S = self._shift.todense()
            C = self._coin.todense()

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
            U = self._shift @ self._coin
            if self._oracle is not None:
                U = U @ self._oracle

        self._evolution = U
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
