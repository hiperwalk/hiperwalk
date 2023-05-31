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

    adjacency : :class:`scipy.sparse.csr_array`, optional
        .. deprecated:: 2.0
            It will be removed in version 2.1.
            Use ``graph`` instead.

        Use ``graph`` instead.
        Adjacency matrix of the graph.

    **kwargs : optional
        Optional arguments for setting the non-default evolution operator.
        See :meth:`set_evolution`.

    Warns
    -----
    If ``adjacency`` is set. It is deprecated. Use ``graph`` instead.

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

    def __init__(self, graph=None, adjacency=None, **kwargs):

        self._shift = None
        self._coin = None
        self._oracle_coin = []
        super().__init__(graph=graph, adjacency=adjacency)

        # Expects adjacency matrix with only 0 and 1 as entries
        self.hilb_dim = self._graph.number_of_arcs()

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
        the ``_graph`` attribute.

        The operator is set for future usage.
        If an evolution operator was set previously,
        it is unset for coherence.
        """

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
        self._evolution = None

    def set_shift(self, shift='default'):
        r"""
        Set the shift operator.

        Set either the flipflop or the persistent shift operator.

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

        See Also
        --------
        has_persistent_shift

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

        self._evolution = None

        if __DEBUG__:
            if self._shift is None: raise AssertionError

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

        .. todo::

            Check if explicit coin is valid.

        """
        try:
            if len(coin.shape) != 2:
                raise TypeError('Explicit coin is not a matrix.')

            # explicit coin
            if not scipy.sparse.issparse(coin):
                coin = scipy.sparse.csr_array(coin)

            self._coin = coin
            self._evolution = None
            return

        except AttributeError:
            pass

        coin_list, undefined_coin = self._coin_to_list(coin)
        if undefined_coin:
            raise ValueError('Coin was not specified for all vertices.')

        self._coin = coin_list
        self._evolution = None

        if __DEBUG__:
            if self._coin is None: raise AssertionError
            if self._evolution is not None: raise AssertionError

    def _coin_to_valid_name(self, coin):
        r"""
        Convert a string to its respective valid coin name.
        """
        s = coin
        if s == 'default' or s == 'd':
            s = self._graph.default_coin()
        
        if len(s) <= 2:
            prefix = 'minus_' if len(coin) == 2 else ''
            abbrv = {'F': 'fourier', 'G' : 'grover',
                     'H': 'hadamard', 'I': 'identity'}
            s = prefix + abbrv[s[-1]]

        if s not in CoinedWalk._coin_funcs.keys():
            raise ValueError(
                'Invalid coin. Expected any of '
                + str(list(CoinedWalk._coin_funcs.keys())) + ', '
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
                        value = [value]
                    for vertex in value:
                        coin_list[vertex] = coin_name
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

    def _coin_list_to_explicit_coin(self, coin_list):
        num_vert = self._graph.number_of_vertices()
        degree = self._graph.degree
        blocks = [self._coin_funcs[coin_list[v]](degree(v))
                  for v in range(num_vert)]
        C = scipy.sparse.block_diag(blocks, format='csr')
        return scipy.sparse.csr_array(C)

    #def get_coin(self):
    #    r"""
    #    Returns the coin operator in matricial form.
    #
    #    Returns
    #    -------
    #    :class:`scipy.sparse.csr_array`
    #    """
    #    if not scipy.sparse.issparse(self._coin):
    #        coin_list = self._coin
    #        C = self._coin_list_to_explicit_coin(coin_list)
    #
    #        if __DEBUG__:
    #            if not isinstance(C, scipy.sparse.csr_array):
    #                raise AssertionError
    #
    #        return C
    #    return self._coin

    def set_marked(self, marked=[]):
        r"""
        Set marked elements (vertices).

        If a list of vertices is received,
        those vertices are considered marked.
        The evolution operators remains unchanged.

        If a dictionary is passed,
        the coin of those vertices are substituted
        only for generating the evolution operator.
        This can only be done if the set coin operator is
        not a explicit matrix.

        Parameters
        ----------
        marked : list of int of dict
            list of vertices to be marked and
            how they are going to be marked.
            
            * list of int
                Given vertices are set as marked but
                the evolution operator remains unchanged.

            * dict
                A dictionary with structure
                ``{coin_type : list_of_vertices}``.
                Analogous to the one accepted by :meth:`set_coin`.

        See Also
        --------
        set_coin
        """
        coin_list = []

        if isinstance(marked, dict):
            coin_list, _ = self._coin_to_list(marked)

            dict_values = marked.values()
            vertices = [vlist if hasattr(vlist, '__iter__') else [vlist]
                        for vlist in dict_values]
            vertices = [v for vlist in vertices for v in vlist ]
            marked = vertices

        super().set_marked(marked)
        if bool(coin_list) or bool(self._oracle_coin):
            # evolution operator was changed
            self._evolution = None
        self._oracle_coin = coin_list

    #def get_oracle_coin(self):
    #    r"""
    #    The coins to be applied in the marked vertices.
    #
    #    Returns
    #    -------
    #    list of str
    #        If empty, no coin is changed (coin operator not altered).
    #        If an entry is the empty string,
    #        the coin for that vertex is not substituted.
    #    """
    #    return self._oracle_coin

    def get_coin(self):
        r"""
        Return coin to be used for creating the evolution operator.

        Returns
        -------
        :class:`scipy.sparse.csr_array`

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
                start = g.arc_label(vertex, neighbors[0])
                end = g.arc_label(vertex, neighbors[-1]) + 1

                return scipy.sparse.csr_array(self._coin[start:end,
                                                         start:end])

            num_vert = self.number_of_vertices()
            degree = self.degree
            oracle_coin = self._oracle_coin
            coin_funcs = CoinedWalk._coin_funcs
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

    def set_evolution(self, **kwargs):
        """
        Set the evolution operator.

        Shorthand for setting shift, coin and marked vertices.
        They are set using the appropriate ``**kwargs``.
        If ``**kwargs`` is empty, the default arguments are used.

        Parameters
        ----------
        **kwargs : dict, optional
            Arguments for setting the evolution operator.
            Accepts any valid keywords from
            :meth:`set_shift` :meth:`set_coin`, and :meth:`set_marked`.

        See Also
        --------
        set_shift
        set_coin
        set_marked
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
        self._evolution = None

    def get_evolution(self, hpc=True):
        r"""
        Create evolution operator from previously set attributes.

        Creates evolution operator by multiplying the
        shift operator and coin operator.
        If the coin operator is not an explicit matrix,
        and the coin for marked vertices was specified,
        the coin of each marked vertex is substituted.

        Parameters
        ----------
        hpc : bool, default=True
            Whether or not the evolution operator should be
            constructed using nelina's high-performance computing.

        Returns
        -------
        :class:`scipy.sparse.csr_array`

        Notes
        -----
        The evolution operator is given by

        .. math::
           U = SC'

        where :math`S` is the shift operator, and
        :math:`C'` is the coin operator (probably) altered by
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
        .. [1] Portugal, Renato. "Quantum walks and search algorithms".
            Vol. 19. New York: Springer, 2013.

        Examples
        --------

        .. todo::
            Valid examples to clear behaviour.
        """
        if self._evolution is not None:
            # evolution operator was not changed.
            # No need to create it again
            return self._evolution

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

        Notes
        -----
        .. note::
            
            benchmark performance
        """
        # TODO: test with nonregular graph
        # TODO: test with nonuniform condition
        if __DEBUG__:
            start = now()

        if len(states.shape) == 1:
            states = [states]

        #edges_indices = self.adj_matrix.indptr 
        #
        #prob = np.array([[
        #        Graph._elementwise_probability(
        #            states[i][edges_indices[j]:edges_indices[j + 1]]
        #        ).sum()
        #        for j in range(len(edges_indices) - 1)
        #    ] for i in range(len(states)) ])

        def get_entries(state, indexes):
            return np.array([state[i] for i in indexes])

        num_vert = self._graph.number_of_vertices()
        graph = self._graph
        prob = np.array([[CoinedWalk._elementwise_probability(
                              get_entries(
                                  states[i], 
                                  graph.arcs_with_tail(v)
                              )
                          ).sum()
                          for v in range(num_vert)]
                        for i in range(len(states))])

        # TODO: benchmark (time and memory usage)
        return prob

    def state(self, *args):
        """
        Generates a valid state.

        The state corresponds to the walker being in a superposition
        of the ``entries``.
        for instance, click on :meth:`qwalk.coined.Graph`.
        The final state is normalized in order to be unitary.

        Parameters
        ----------
        *args
            Each entry is a tuple (or array).
            An entry can be specified in three different ways:
            ``(amplitude, (vertex, dst_vertex))``,
            ``(amplitude, arc_label)``,
            ``(amplitude, vertex, coin)``.

            amplitude
                The amplitude of the given entry.
            vertex
                The vertex corresponding to the position of the walker
                in the superposition.
            dst_vertex 
                The vertex which the coin is pointing to.
                In other words, the tuple
                ``(vertex, dst_vertex)`` must be a valid arc.
            arc_label
                The arc label with respect to the arc ordering
                given by the computational basis.
            coin
                The direction towards which the coin is pointing.
                It is dependabble on the Graph coloring.

        Notes
        -----
        If entries are repeated (except by the amplitude),
        they are overwritten by the last one.

        Examples
        --------
        .. todo::
            
            qw.state([1/np.sqrt(2), 0], [1/np.sqrt(2), (1, 0)])
        """

        has_complex = np.any([arg[0].imag != 0 for arg in args])
        state = np.zeros(self.hilb_dim, dtype=complex if has_complex
                                                      else float)

        for entry in args:
            if len(entry) == 3:
                raise NotImplementedError(
                    'position-coin notation not implemented')

            if hasattr(entry[1], '__iter__'):
                # arc notation
                head, tail = entry[1]
                state[self._graph.arc_label(head, tail)] = entry[0]
            else:
                # arc label
                state[entry[1]] = entry[0]

        return self._normalize(state)

    def ket(self, *args):
        r"""
        Create a computational basis state.

        Parameters
        ----------
        *args
            The ket label.
            There are two different labels acceptable.

            tail, head
                The arc notation.
            arc_label
                The label of the arc.
                Its number according to the computational basis order.

        Examples
        --------
        .. todo::
            valid examples
        """
        ket = np.zeros(self.hilb_dim, dtype=float)
        if len(args) == 2:
            ket[self._graph.arc_label(args[0], args[1])] = 1
        else:
            ket[args] = 1

        return ket
