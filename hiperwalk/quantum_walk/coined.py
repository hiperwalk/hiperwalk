import numpy as np
import scipy
import scipy.sparse
import networkx as nx
from .quantum_walk import QuantumWalk
from .._constants import __DEBUG__, PYNEBLINA_IMPORT_ERROR_MSG
from scipy.linalg import hadamard, dft
try:
    from . import _pyneblina_interface as nbl
except:
    pass

if __DEBUG__:
    from time import time as now

class Coined(QuantumWalk):
    r"""
    Manage an instance of a coined quantum walk on
    any simple graph.

    This class provides methods for handling, simulating, and generating
    operators within the coined quantum walk model for
    various types of graphs.
    
    For additional details about coined quantum walks,
    refer to the Notes Section.

    Parameters
    ----------
    graph
        Graph on which the quantum walk takes place.
        It can be the graph itself (:class:`hiperwalk.graph.Graph`) or
        its adjacency matrix (:class:`scipy.sparse.csr_array`).

    **kwargs : optional
        Optional arguments for setting the non-default evolution operator.
        See :meth:`set_evolution`.

    Raises
    ------
    TypeError
        if ``adj_matrix`` is not an instance of
        :class:`scipy.sparse.csr_array`.

    See Also
    --------
    set_evolution

    Notes
    -----

    The computational basis is spanned by the set of arcs.
    The cardinality is :math:`2|E|`, where :math:`E`
    represents the edge set of the graph.
    For further information regarding the order of the arcs,
    consult the respective graph descriptions.
    The Hilbert space of a coined quantum walk is denoted by
    :math:`\mathcal{H}^{2|E|}`.

    For a more detailed understanding of coined quantum walks,
    refer to Section 7.2: Coined Walks on Arbitrary Graphs,
    found in the book  'Quantum Walks and Search Algorithms' [1]_.

    References
    ----------
    .. [1] R. Portugal. "Quantum walks and search algorithms", 2nd edition,
        Springer, 2018.
    """

    # static attributes.
    # The class must be instantiated once, otherwise are dicts are empty.
    # The class must be instantiated so the interpreter knows
    # the memory location for the function pointers.
    _coin_funcs = dict()
    _valid_kwargs = dict()

    def __init__(self, graph=None, **kwargs):

        self._shift = None
        self._coin = None
        self._oracle_coin = []
        super().__init__(graph=graph)

        # Expects adjacency matrix with only 0 and 1 as entries
        self.hilb_dim = self._graph.number_of_arcs()

        if not bool(Coined._valid_kwargs):
            # assign static attribute
            Coined._valid_kwargs = {
                'shift': Coined._get_valid_kwargs(self._update_shift),
                'coin': Coined._get_valid_kwargs(self._update_coin),
                'marked': Coined._get_valid_kwargs(self._update_marked),
                'evolution': Coined._get_valid_kwargs(
                    self._update_evolution)
            }

        # dict with valid coins as keys and the respective
        # function pointers.
        if not bool(Coined._coin_funcs):
            # assign static attribute
            Coined._coin_funcs = {
                'fourier': Coined._fourier_coin,
                'grover': Coined._grover_coin,
                'hadamard': Coined._hadamard_coin,
                'identity': Coined._identity_coin,
                'minus_fourier': Coined._minus_fourier_coin,
                'minus_grover': Coined._minus_grover_coin,
                'minus_hadamard': Coined._minus_hadamard_coin,
                'minus_identity': Coined._minus_identity_coin
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
        the ``_graph`` attribute.

        The operator is configured for future use. If an evolution
        operator was set earlier, it will be unset to maintain coherence.
        """

        num_vert = self._graph.number_of_vertices()
        num_arcs = self._graph.number_of_arcs()

        S_cols = [self._graph.arc_number((j, i))
                  for i in range(num_vert)
                  for j in self._graph.neighbors(i)]

        # Using csr_array((data, indices, indptr), shape)
        # Note that there is only one entry per row and column
        S = scipy.sparse.csr_array(
            ( np.ones(num_arcs, dtype=np.int8),
              S_cols, np.arange(num_arcs+1) ),
            shape=(num_arcs, num_arcs)
        )

        self._shift = S

    def has_persistent_shift(self):
        r"""
        Checks whether the persistent shift operator is defined
        for the current graph.

        Returns
        -------
        bool

        Notes
        -----
        The persistent shift is sometimes called *moving shift*.

        The persistent shift operator can only be defined in a
        meaningful way for certain specific graphs. For instance,
        for graphs that can be embedded onto a plane,
        directions such as left, right, up, and down
        can be referred to naturally.
        """
        return 'previous_arc' in dir(self._graph)

    def _set_persistent_shift(self):
        r"""
        Create the persistent shift operator (:math:`S`) based on
        the ``_graph`` attribute.

        The operator is set for future usage.
        If an evolution operator was set previously,
        it is unset for coherence.
        """
        num_arcs = self._graph.number_of_arcs()

        S_cols = [self._graph.previous_arc(i) for i in range(num_arcs)]

        # Using csr_array((data, indices, indptr), shape)
        # Note that there is only one entry per row and column
        S = scipy.sparse.csr_array(
            ( np.ones(num_arcs, dtype=np.int8),
              S_cols, np.arange(num_arcs+1) ),
            shape=(num_arcs, num_arcs)
        )

        self._shift = S

    def _update_shift(self, shift='default'):
        valid_vals = ['default', 'flipflop', 'persistent', 'ff', 'p']
        if shift not in valid_vals:
            raise ValueError(
                "Invalid `shift` value. Expected one of "
                + str(valid_vals) + ". But received '"
                + str(shift) + "' instead."
            )

        if shift == 'default':
            shift = 'p' if self.has_persistent_shift() else 'ff'

        if shift == 'ff':
            shift = 'flipflop'
        elif shift == 'p':
            shift = 'persistent'

        
        if shift == 'flipflop':
            self._set_flipflop_shift()
        else:
            self._set_persistent_shift()

        if __DEBUG__:
            if self._shift is None: raise AssertionError

    def set_shift(self, shift='default', **kwargs):
        r"""
        Set the shift operator.

        Sets either the flipflop or the persistent shift operator.
        Afterwards, the evolution operator is updated accordingly.

        Parameters
        ----------
        shift: {'default', 'flipflop', 'persistent', 'ff', 'p'}
            Whether to create the flip flop or the persistent shift.
            By default, creates the persistent shift if it is defined;
            otherwise creates the flip flop shift.
            Argument ``'ff'`` is an alias for ``'flipflop'``.
            Argument ``'p'`` is an alias for ``'persistent'``.

        **kwargs:
            Additional arguments.
            Used for determining the procedure for
            updating the evolution operator.
            See :meth:`hiperwalk.Coined.set_evolution` for valid options.

        Raises
        ------
        AttributeError
            If ``shift='persistent'`` and
            the persistent shift operator cannot be defined.

        See Also
        --------
        has_persistent_shift
        set_evolution

        Notes
        -----
        .. note::
            Check :class:`Coined` Notes for details
            about the computational basis.

        The persistent shift is sometimes called *moving shift*.

        The flip-flop shift operator :math:`S` is defined such that

        .. math::
            \begin{align*}
                S \ket{(v, u)} &= \ket{(u, v)} \\
                \implies S\ket i &= \ket j
            \end{align*}

        where :math:`i` is the label of the edge :math:`(v, u)` and
        :math:`j` is the label of the edge :math:`(u, v)`.


        For a more comprehensive understanding of the flipflop
        shift operator, refer to Section 7.2: Coined Walks on Arbitrary
        Graphs in the book "Quantum Walks and Search Algorithms" [1]_.
        

        References
        ----------
        .. [1] R. Portugal. "Quantum walks and search algorithms",
            2nd edition, Springer, 2018.

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
            >>> qw = hpw.Coined(A)
            >>> qw.set_shift(shift='flipflop')
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

        .. todo::
            
            Add persistent example.
        """
        self._update_shift(shift=shift)
        self._update_evolution(**kwargs)

    def get_shift(self):
        r"""
        Return the shift operator.

        Shift operator used for constructing the evolution operator.

        Returns
        -------
        scipy.sparse.csr_array
        """
        return self._shift

    def _update_coin(self, coin='default'):
        try:
            if len(coin.shape) != 2:
                raise TypeError('Explicit coin is not a matrix.')

            # explicit coin
            if not scipy.sparse.issparse(coin):
                coin = scipy.sparse.csr_array(coin)

            self._coin = coin
            return

        except AttributeError:
            pass

        coin_list, undefined_coin = self._coin_to_list(coin)
        if undefined_coin:
            raise ValueError('Coin was not specified for all vertices.')

        self._coin = coin_list

        if __DEBUG__:
            if self._coin is None: raise AssertionError

    def set_coin(self, coin='default', **kwargs):
        """
        Generate a coin operator based on the graph structure.

        Constructs a coin operator based on the degree of each vertex.
        A single type of coin may be applied to
        all vertices or a subset thereof.
        Once the coin operator is set,
        the evolution operator is updated accordingly.

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

        **kwargs:
            Additional arguments.
            Used for determining the procedure for
            updating the evolution operator.
            See :meth:`hiperwalk.Coined.set_evolution` for valid options.

        See Also
        --------
        set_evolution

        Notes
        -----
        Owing to the selected computational basis (refer to the
        Notes in the :class:`Coined`),
        the outcome is a block diagonal operator.
        Each block is a :math:`\deg(v)`-dimensional ``coin``.
        As a result, there are :math:`|V|` blocks in total.

        .. todo::

            Check if explicit coin is valid.

        """
        for key in kwargs:
            if key not in Coined._valid_kwargs['evolution']:
                kwargs.pop(key)

        self._update_coin(coin=coin)
        self._update_evolution(**kwargs)

    def default_coin(self):
        r"""
        Returns the default coin name.

        The default coin name depends on the graph on
        which the quantum walk occurs.
        In general, the default coin is ``'grover'``.
        For the :class:`hiperwalk.Line` and :class:`hiperwalk.Cycle`,
        the default coin is ``'hadamard'``.

        Returns
        -------
        str

        Notes
        -----
        Although the default coin depends on the graph,
        the coin is not an attribute of the graph.
        That is, it is part of graph theory.
        In addition, it does not make sense to have a
        default coin in continuous-time quantum walks.
        Hence, the method is implemented in the Coined class
        instead of each graph class.

        Following duck typing,
        we do not verify if the graph is an instance of the line or cycle.
        Instead, we verify if the maximum degree on the graph is at most 2.
        """
        degs = list(map(self._graph.degree,
                        np.arange(self._graph.number_of_vertices())))
        if max(degs) <= 2:
            return 'hadamard'
        return 'grover'

    def _coin_to_valid_name(self, coin):
        r"""
        Convert a string to its respective valid coin name.
        """
        s = coin
        if s == 'default' or s == 'd':
            s = self.default_coin()
        
        if len(s) <= 2:
            prefix = 'minus_' if len(coin) == 2 else ''
            abbrv = {'F': 'fourier', 'G' : 'grover',
                     'H': 'hadamard', 'I': 'identity'}
            s = prefix + abbrv[s[-1]]

        if s not in Coined._coin_funcs.keys():
            raise ValueError(
                'Invalid coin. Expected any of '
                + str(list(Coined._coin_funcs.keys())) + ', '
                + "but received '" + str(coin) + "'."
            )

        return s

    def _coin_to_list(self, coin):
        r"""
        Convert str, list of str or dict to valid coin list.

        See Also
        --------
        set_coin
        """
        num_vert = self._graph.number_of_vertices()
        coin_list = []
        undefined_coin = False

        if isinstance(coin, str):
            coin_list = [self._coin_to_valid_name(coin)] * num_vert

        elif isinstance(coin, dict):
            coin_list = [''] * num_vert
            for key in coin:
                coin_name = self._coin_to_valid_name(key)
                value = coin[key]
                if value != []:
                    if not hasattr(value, '__iter__'):
                        raise TypeError("Expected a list of vertices. "
                                + "Received " + str(type(value)) + " "
                                + "with value " + str(value) + " instead.")
                    for v in value:
                        coin_list[self._graph.vertex_number(v)] = coin_name
                else:
                    coin_list = [coin_name if coin_list[i] == ''
                                 else coin_list[i]
                                 for i in range(num_vert)]

            undefined_coin = '' in coin_list
        else:
            #list of coins
            if len(coin) != num_vert:
                raise ValueError('There were ' + str(len(coin))
                                 + ' coins specified. Expected '
                                 + str(num_vert) + 'coins instead.')

            coin_list = list(map(self._coin_to_valid_name, coin))

        return coin_list, undefined_coin

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
        return scipy.sparse.eye(dim)

    @staticmethod
    def _minus_fourier_coin(dim):
        return -Coined._fourier_coin(dim)

    @staticmethod
    def _minus_grover_coin(dim):
        return -Coined._grover_coin(dim)

    @staticmethod
    def _minus_hadamard_coin(dim):
        return -Coined._hadamard_coin(dim)

    @staticmethod
    def _minus_identity_coin(dim):
        return -np.identity(dim)

    def _update_marked(self, marked=[]):
        coin_list = []

        if isinstance(marked, dict):
            coin_list, _ = self._coin_to_list(marked)

            dict_values = marked.values()
            vertices = [vlist if hasattr(vlist, '__iter__') else [vlist]
                        for vlist in dict_values]
            vertices = [v for vlist in vertices for v in vlist ]
            marked = vertices

        super()._update_marked(marked=marked)
        self._oracle_coin = coin_list

    def set_marked(self, marked=[], **kwargs):
        r"""
        Set marked vertices.

        If a list of vertices is provided, those vertices are
        deemed as marked.
        The evolution operator is updated accordingly.

        If a dictionary is passed,
        the coin of those vertices are substituted
        only for generating the evolution operator.
        This can only be done if the set coin operator is
        not a explicit matrix.

        Parameters
        ----------
        marked : list of vertices or dict
            list of vertices to be marked and
            how they are going to be marked.
            
            * list of vertices
                Given vertices are set as marked.
                The coin for those vertices is '-I'.

            * dict
                A dictionary with structure
                ``{coin_type : list_of_vertices}``.
                Analogous to the one accepted by :meth:`set_coin`.

        **kwargs:
            Additional arguments.
            Used for determining the procedure for
            updating the evolution operator.
            See :meth:`hiperwalk.Coined.set_evolution` for valid options.

        See Also
        --------
        set_coin
        set_evolution
        """
        super().set_marked(marked=marked, **kwargs)

    def _coin_list_to_explicit_coin(self, coin_list):
        num_vert = self._graph.number_of_vertices()
        degree = self._graph.degree
        blocks = [self._coin_funcs[coin_list[v]](degree(v))
                  for v in range(num_vert)]
        C = scipy.sparse.block_diag(blocks, format='csr')
        return scipy.sparse.csr_array(C)

    def get_coin(self):
        r"""
        Return coin used for creating the evolution operator.

        Returns
        -------
        :class:`scipy.sparse.csr_array`

        See Also
        --------
        set_coin
        
        Notes
        -----
        The final coin :math:`C'` is obtained by multiplying the
        coin operator :math:`C` and the oracle :math:`R`.
        That is,

        .. math::
            
            C' = CR .

        The oracle is not explicitly saved.
        Instead, the oracle coins are saved -- i.e.
        which coin is going to be applied to each marked vertices.
        To generate :math:`C'` we simply substitute the original coin
        by the oracle coin in all marked vertices.

        Examples
        --------
        .. todo::
            examples
        """
        if scipy.sparse.issparse(self._coin):
            if not bool(self._oracle_coin):
                return self._coin

            # if coin was explicitly set,
            # and there are different coins for the marked vertices,
            # change them.
            def get_block(vertex):
                g = self._graph
                neighbors = g.neighbors(vertex)
                start = g.arc_number((vertex, neighbors[0]))
                end = g.arc_number((vertex, neighbors[-1])) + 1

                return scipy.sparse.csr_array(self._coin[start:end,
                                                         start:end])

            num_vert = self.number_of_vertices()
            degree = self.degree
            oracle_coin = self._oracle_coin
            coin_funcs = Coined._coin_funcs
            blocks = [coin_funcs[oracle_coin[v]](degree(v))
                      if oracle_coin[v] != ''
                      else get_block(v)
                      for v in range(num_vert)]
            C = scipy.sparse.block_diag(blocks, format='csr')

            return scipy.sparse.csr_array(C)

        oracle_coin = self._oracle_coin
        if bool(oracle_coin):
            coin = self._coin
            coin_list = [oracle_coin[i] if oracle_coin[i] != ''
                         else coin[i]
                         for i in range(len(coin))]
        else:
            coin_list = self._coin

        return self._coin_list_to_explicit_coin(coin_list)

    def _update_evolution(self, hpc=True):
        U = None
        if hpc and not self._pyneblina_imported():
            hpc = False

        S = self.get_shift()
        C = self.get_coin()

        if hpc:

            S = S.todense()
            C = C.todense()

            nbl_S = nbl.send_matrix(S)
            del S
            nbl_C = nbl.send_matrix(C)
            del C
            nbl_C = nbl.multiply_matrices(nbl_S, nbl_C)

            del nbl_S

            U = nbl.retrieve_matrix(nbl_C)
            del nbl_C
            U = scipy.sparse.csr_array(U)

        else:
            U = S @ C

        self._evolution = U
        return U

    def set_evolution(self, hpc=True, **kwargs):
        """
        Create evolution operator.

        Sets shift, coin and marked vertices.
        They are set using the appropriate ``**kwargs``.
        If ``**kwargs`` is empty, the default arguments are used.

        Then, creates evolution operator by multiplying the
        shift operator and coin operator.
        If the coin operator is not an explicit matrix,
        and the coin for marked vertices was specified,
        the coin of each marked vertex is substituted.

        Parameters
        ----------
        **kwargs : dict, optional
            Arguments for setting the evolution operator.
            Accepts any valid keywords from
            :meth:`set_shift` :meth:`set_coin`, and :meth:`set_marked`.

        hpc : bool, default=True
            Whether or not the evolution operator should be
            constructed using nelina's high-performance computing.

        See Also
        --------
        set_shift
        set_coin
        set_marked
        get_evolution

        Notes
        -----
        The evolution operator is given by

        .. math::
           U = SC'

        where :math`S` is the shift operator, and
        :math:`C'` is the coin operator (probably) modified by
        the marked vertices [1]_.

        If the coin operator was set as an explicit matrix,
        the marked vertices to not alter it.
        If the coin operator was not set as an explicit matrix
        (e.g. as a list of coins),
        the coin of each marked vertex is substituted as specified by
        the last :meth:`set_marked` call.

        .. todo::
            * Sparse matrix multipliation is not supported yet.
              Converting all matrices to dense.
              Then converting back to sparse.
              This uses unnecessary memory and computational time.
            * Check if matrix is sparse in pynelibna interface
            * Check if matrices are deleted from memory and GPU.


        References
        ----------
        .. [1] R. Portugal. "Quantum walks and search algorithms",
            2nd edition, Springer, 2018.

        Examples
        --------

        .. todo::
            Valid examples to clear behavior.
        """

        S_kwargs = Coined._filter_valid_kwargs(
                              kwargs,
                              Coined._valid_kwargs['shift'])
        C_kwargs = Coined._filter_valid_kwargs(
                              kwargs,
                              Coined._valid_kwargs['coin'])
        R_kwargs = Coined._filter_valid_kwargs(
                              kwargs,
                              Coined._valid_kwargs['marked'])

        self._update_shift(**S_kwargs)
        self._update_coin(**C_kwargs)
        self._update_marked(**R_kwargs)
        self._update_evolution(hpc=hpc)

    def probability_distribution(self, states):
        r"""
        Compute the probability distribution of given state(s).

        The probability of the walker being found on each vertex
        for the given state(s).

        Parameters
        ----------
        states : :class:`numpy.ndarray`
            The state(s) used to compute the probabilities.
            It may be a single state or a list of states.

        Returns
        -------
        probabilities : :class:`numpy.ndarray`
            If ``states`` is a single state,
            ``probabilities[v]`` is the probability of the
            walker being found on vertex ``v``.

            If ``states`` is a list of states,
            ``probabilities[i][v]`` is the probability of the
            walker beign found at vertex ``v`` in ``states[i]``.

        See Also
        --------
        simulate

        Notes
        -----
        The probability for a given vertex :math:`u` is the sum of the
        absolute square of the amplitudes of the arcs with tail :math:`u`.
        That is, for an arbitrary superposition

        .. math::
            \sum_{(u, v) \in A(\vec G)} \alpha_{u,v} \ket{u,v},

        -- where :math:`\vec G` is the graph :math:`G` with each
        edge substituted by two arcs (one for each direction) --
        the probability associated with vertex :math:`u` is

        .. math::
            \sum_{v \in N(u)}|\alpha_{u, v}|^2,

        where :math:`N(u)` is the set of neighbors of :math:`u`.
        """
        if __DEBUG__:
            start = now()

        if len(states.shape) == 1:
            states = [states]

        graph = self._graph
        num_vert = graph.number_of_vertices()
        prob = np.array([[Coined._elementwise_probability(
                              states[i, graph.arcs_with_tail(v)]).sum()
                          for v in range(num_vert)]
                         for i in range(len(states))])

        return prob

    def state(self, *args):
        """
        Generates a valid state.

        The state corresponds to the walker being in a superposition
        of the ``entries``.
        For instance, click on :meth:`qwalk.coined.Graph`.
        The final state is normalized in order to be a unit vector.

        Parameters
        ----------
        *args
            Each entry is a tuple (or array).
            An entry can be specified in three different ways:
            ``(amplitude, (tail, head))``,
            ``(amplitude, tail, head)``, and
            ``(amplitude, arc_number)``.

            amplitude
                The amplitude of the given entry.
            tail
                The vertex corresponding to the position of the walker
                in the superposition.
                In other words, the tail of the arc.
            head
                The vertex to which the coin is pointing.
                That is, the tuple
                ``(tail, head)`` must be a valid arc.
            arc_number
                The numerical arc label with respect to the arc ordering
                given by the computational basis.

        Notes
        -----
        If there are repeated arcs,
        the amplitude of the last entry is used.

        Examples
        --------
        The following commands generate the same state on a
        ``(dim, dim)``-dimensional grid.

        .. testsetup::

            from sys import path
            path.append('../..')
            import hiperwalk as hpw
            import numpy as np
            dim = 10
            g = hpw.Grid((dim, dim))
            qw = hpw.Coined(graph=g)

        >>> psi = qw.state((1, (0, 1)), [1, 1], (1, 2))
        >>> psi1 = qw.state((1, ([0, 0], [1, 0])),
        ...                 [[1, (0, dim - 1)],
        ...                  (1, [(0, 0), [0, 1]])])
        >>> psi2 = qw.state([(1, [0, 0], [1, 0]),
        ...                  [1, 0, dim - 1]],
        ...                 (1, (0, 0), [0, 1]))
        >>> np.all(psi == psi1)
        True
        >>> np.all(psi1 == psi2)
        True
        """
        if len(args) == 0:
            raise TypeError("Entries were not specified.")

        state = [0] * self.hilb_dim

        def add_entry(entry):
            ampl = entry[0]
            arc = entry[1:]
            if len(arc) == 1:
                arc = arc[0]
            state[self._graph.arc_number(arc)] = ampl

        for arg in args:
            if hasattr(arg[0],'__iter__'):
                for entry in arg:
                    add_entry(entry)
            else:
                add_entry(arg)

        state = np.array(state)
        return self._normalize(state)

    def ket(self, *args):
        r"""
        Create a computational basis state.

        Parameters
        ----------
        *args
            The ket label.
            There are three different labels acceptable.

            (tail, head)
                The arc notation.
            tail, head
                The arc notation with ``tail`` and ``head`` as
                separate arguments.
            arc_number
                The label of the arc.
                Its number according to the computational basis order.

        Examples
        --------
        .. todo::
            valid examples
        """
        ket = np.zeros(self.hilb_dim, dtype=float)
        ket[self._graph.arc_number(*args)] = 1

        return ket

    def _prepare_engine(self, initial_state, hpc):
        if hpc:
            S = nbl.send_matrix(self.get_shift())
            C = nbl.send_matrix(self.get_coin())
            self._simul_mat = (C, S)
            self._simul_vec = nbl.send_vector(initial_state)

            dtype = (complex if (S.is_complex or C.is_complex
                                 or np.iscomplex(initial_state.dtype))
                     else np.double)

            return dtype

        else:
            return super()._prepare_engine(initial_state, hpc)


    def _simulate_step(self, step, hpc):
        if hpc:
            for i in range(step):
                self._simul_vec = nbl.multiply_matrix_vector(
                    self._simul_mat[0], self._simul_vec)

                self._simul_vec = nbl.multiply_matrix_vector(
                    self._simul_mat[1], self._simul_vec)
        else:
            super()._simulate_step(step, hpc)
