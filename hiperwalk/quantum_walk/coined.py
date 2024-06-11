import numpy as np
import scipy
import scipy.sparse
import networkx as nx
from .quantum_walk import QuantumWalk
from ..graph import SDMultigraph
from scipy.linalg import hadamard, dft
from . import _pyneblina_interface as nbl

class Coined(QuantumWalk):
    r"""
    Manage instances of coined quantum walks on arbitrary graphs.

    The class provides methods to handle and generate operators in the 
    coined quantum walk model. It also facilitates the simulation of 
    coined quantum walks on graphs.
    
    For additional details about coined quantum walks,
    refer to the Notes Section.

    Parameters
    ----------
    graph
        Graph on which the quantum walk takes place.
        Two types of entries are accepted:

        * Simple graph (:class:`hiperwalk.Graph`);
        * Multigraph (:class:`hiperwalk.Multigraph`);

        A symmetric directed multigraph is created based on the input.

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
    set_shift
    set_coin

    Notes
    -----

    The coined quantum walk model is a quantum analog of 
    classical random walks, incorporating an additional 
    quantum coin-toss mechanism. It uses an extra quantum 
    internal degree of freedom, represented by the coin state,
    to determine the direction of the walker's movement 
    on a graph.

    In the coined model, the graph is interpreted
    as a directed graph as follows:
    Each edge in :math:`G(V, E)` connecting two distinct vertices
    translates into a pair of arcs in the directed graph
    :math:`\vec{G}(V, \mathcal{A})`, where

    .. math::
        \begin{align*}
            \mathcal{A} = \bigcup_{v_k v_\ell\, \in E} \{(v_k, v_\ell), (v_\ell, v_k)\}.
        \end{align*}

    .. note::
        The order of arcs depends on the order of neighbors.

    Arcs are represented using either
    the (tail,head) notation or numerical labels.
    In the :obj:`Graph` class, the arc labels are ordered such that for two arcs,
    :math:`(v_i, v_j)` and :math:`(v_k, v_\ell)`, with labels :math:`a_1` and
    :math:`a_2` respectively, :math:`a_1 < a_2` if and only if :math:`i < k` or
    (:math:`i = k` and :math:`j < \ell`).
    Loops are depicted as single arcs,
    affecting the dimension of the associated Hilbert space.
    In coined quantum walks, the weights of arcs do not influence the dynamics.

    The computational basis is composed of the graph's arc set.
    For simple graphs, the cardinality of the computational
    basis is :math:`2|E|`, where :math:`E`
    represents the graph's edge set.
    When a loop is added to the graph, the cardinality of the 
    computational basis increases by one for each loop.
    
    The arcs are arranged within the computational basis 
    to ensure that the coin operator adopts a block-diagonal 
    matrix form.
    For additional information on the arc ordering,
    please consult the respective graph descriptions.

    For a more detailed understanding of coined quantum walks,
    refer to Section 7.2: Coined Walks on Arbitrary Graphs,
    found in the book  'Quantum Walks and Search Algorithms' [1]_.

    For example, the graph :math:`G(V, E)` shown in
    Figure 1 has an adjacency matrix ``adj_matrix``.

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

    .. graphviz:: ../../graphviz/graph-example.dot
        :align: center
        :layout: neato
        :caption: Figure 1

    Suppose that, for this graph,
    the neighbors are given in ascending order.
    Then,
    the arcs of the associated digraph in the arc notation are

    >>> arcs = [(i, j) for i in range(4)
    ...                for j in range(4) if adj_matrix[i,j] == 1]
    >>> arcs
    [(0, 1), (1, 0), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]

    Note that ``arcs`` is already sorted, hence the associated
    numeric labels are

    >>> arcs_labels = {arcs[i]: i for i in range(len(arcs))}
    >>> arcs_labels
    {(0, 1): 0, (1, 0): 1, (1, 2): 2, (1, 3): 3, (2, 1): 4, (2, 3): 5, (3, 1): 6, (3, 2): 7}

    The numeric labels are depicted in Figure 2.

    .. graphviz:: ../../graphviz/graph-arcs.dot
        :align: center
        :layout: neato
        :caption: Figure 2

    If we insert the labels of the arcs into the adjacency matrix,
    we obtain matrix ``adj_labels`` as follows:

    >>> adj_labels = [[arcs_labels[(i,j)] if (i,j) in arcs_labels
    ...                                   else '' for j in range(4)]
    ...               for i in range(4)]
    >>> adj_labels = np.matrix(adj_labels)
    >>> adj_labels
    matrix([['', '0', '', ''],
            ['1', '', '2', '3'],
            ['', '4', '', '5'],
            ['', '6', '7', '']], dtype='<U21')

    Note that, intuitively,
    the arcs are labeled in left-to-right and top-to-bottom fashion.

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

        # create symmetric directed multigraph from input
        sdmg = SDMultigraph(graph)

        super().__init__(graph=sdmg)
        self.hilb_dim = self._graph.number_of_arcs()

        # Specific coined quantum walk attributes
        self._shift = None
        self._coin = None
        self._oracle_coin = []

        # create static dicts
        if not bool(Coined._valid_kwargs):
            Coined._valid_kwargs = {
                'shift': Coined._get_valid_kwargs(self._set_shift),
                'coin': Coined._get_valid_kwargs(self._set_coin),
                'marked': Coined._get_valid_kwargs(self._set_marked),
                'evolution': Coined._get_valid_kwargs(
                    self._set_evolution)
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

    def _set_flipflop_shift(self):
        r"""
        Creates the flipflop shift operator (:math:`S`) based on
        the ``_graph`` attribute.

        The operator is configured for future use. If an evolution
        operator was set earlier, it will be unset to maintain coherence.
        """
        g = self._graph
        num_vert = g.number_of_vertices()
        num_arcs = g.number_of_arcs()

        if g.is_underlying_simple():
            S_cols = [g.arc_number((j, i))
                      for i in range(num_vert)
                      for j in g.neighbors(i)]
        else:
            S_cols = [g.arc_number((j, i, e))
                      for i in range(num_vert)
                      for j in g.neighbors(i)
                      for e in range(g.number_of_edges(j, i))]

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
        Check whether the persistent shift operator is defined
        for the current graph.

        Returns
        -------
        bool

        Notes
        -----
        The persistent shift is sometimes called *moving shift*.

        The persistent shift operator can only be defined in a
        meaningful way for certain specific graphs. For instance,
        for graphs that can be embedded onto a plane so that
        directions such as left, right, up, and down
        can be referred to naturally.
        """
        return self._graph.previous_arc(0) is not None

    def _set_persistent_shift(self):
        r"""
        Creates the persistent shift operator (:math:`S`) based on
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

    def _set_shift(self, shift='default'):
        valid_vals = ['default', 'flipflop', 'persistent', 'ff', 'p']

        # check if string
        try:
            shift = shift.lower()

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

            if str(self._shift) != shift:
                if shift == 'flipflop':
                    self._set_flipflop_shift()
                else:
                    self._set_persistent_shift()
                return True

            return False

        except AttributeError:
            pass

        # check if explict matrix
        try:
            shift[0][0] #if this works, then shift is numpy or list of list
            # convert to sparse
            shift = scipy.sparse.csr_array(shift)
        except NotImplementedError:
            # already sparse
            pass

        if (len(shift.shape) != 2 or shift.shape[0] != shift.shape[1]):
            raise TypeError('Explicit coin is not a square matrix.')

        if (id(self._shift) != id(shift)):
            self._shift = shift
            return True

        return False

    def set_shift(self, shift='default'):
        r"""
        Set the shift operator.

        Defines either the flipflop or the persistent shift operator.
        Following this, the evolution operator updates accordingly.

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

        The flip-flop shift operator :math:`S` is defined as

        .. math::
            S \ket{v, u} = \ket{u, v},

        in the context of arc notation, where :math:`(v, u)` and
        :math:`(u, v)` represent opposite arcs. This can be equivalently 
        expressed as :math:`S\ket{i} = \ket{j}`, where :math:`i` is the 
        label of the arc :math:`(v, u)` and :math:`j` is the label of 
        the arc :math:`(u, v)`. The flip-flop shift satisfies the 
        property :math:`S^2 = I`.

        The persistent shift, also known as the *moving shift*, 
        is defined for graphs with a clear notion of direction or rotation.
        When the shift operator is applied repeatedly, it causes the walker 
        to continue moving persistently in the same direction. Unlike the 
        flip-flop shift, the persistent shift does not satisfy :math:`S^2 = I`.
        
        For a more comprehensive understanding of the 
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
            from sys import path
            path.append('../..')

        .. doctest::

            >>> import hiperwalk as hpw
            >>> A = scipy.sparse.csr_array([[0, 1, 0, 0],
            ...                             [1, 0, 1, 1],
            ...                             [0, 1, 0, 1],
            ...                             [0, 1, 1, 0]])
            >>> qw = hpw.Coined(hpw.Graph(A))
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
        """
        self.set_evolution(shift=shift,
                           coin=self._coin,
                           marked=self._marked)

    def get_shift(self):
        r"""
        Retrieve the shift operator.

        Shift operator used for constructing the evolution operator.

        Returns
        -------
        :class:`scipy.sparse.csr_array`

        See Also
        --------
        set_shift
        """
        return self._shift

    def _set_coin(self, coin='default'):
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

    def set_coin(self, coin='default'):
        """
        Set the coin operator based on the graph's structure.

        Builds a coin operator considering the degree of each vertex.
        The same coin can be applied to all vertices, or multiple 
        coins can be assigned, each to a specific subset of vertices. 
        After setting the coin operator,
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

        See Also
        --------
        set_evolution

            
        Notes
        -----
        
        The output of this method is a block-diagonal 
        operator, which results from the specific ordering of arcs 
        in the computational basis 
        (refer to the Notes in :class:`Coined` for more details).        
        Each block is associated with a :math:`\deg(v)`-dimensional ``coin``.
        As a result, there are :math:`|V|` blocks in total.
        Note that a loop at a vertex :math:`u` is treated
        as the arc :math:`(u,u)`, contributing an additional 
        one to the degree of :math:`u`.
        """
        self.set_evolution(shift=self._shift,
                           coin=coin,
                           marked=self._marked)

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

    def _set_marked(self, marked=[]):
        try:
            marked.get(0) #throws exception if list
        except AttributeError:
            # list
            if len(marked) > 0:
                marked = {'-I': marked}
            else:
                marked = {}

        coin_list, _ = self._coin_to_list(marked)

        dict_values = marked.values()
        vertices = [vlist if hasattr(vlist, '__iter__') else [vlist]
                    for vlist in dict_values]
        vertices = [v for vlist in vertices for v in vlist ]
        marked = vertices

        super()._set_marked(marked=marked)
        self._oracle_coin = coin_list

    def set_marked(self, marked=[]):
        r"""
        Set the marked vertices.

        When a set or list of vertices is provided, they 
        are set as marked.
        The evolution operator is updated accordingly.

        If a dictionary is passed, the coin of those vertices is
        replaced solely for the purpose of generating the evolution 
        operator. This can only be done if the set coin operator is
        not an explicit matrix.

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

        See Also
        --------
        set_coin
        set_evolution
        """
        self.set_evolution(shift=self._shift,
                           coin=self._coin,
                           marked=marked)

    def _coin_list_to_explicit_coin(self, coin_list):
        num_vert = self._graph.number_of_vertices()
        degree = self._graph.degree
        blocks = [self._coin_funcs[coin_list[v]](degree(v))
                  for v in range(num_vert)]
        C = scipy.sparse.block_diag(blocks, format='csr')
        return scipy.sparse.csr_array(C)

    def get_coin(self):
        r"""
        Retrieve the coin used in the creation of the evolution operator.

        Returns
        -------
        :class:`scipy.sparse.csr_array`

        See Also
        --------
        set_coin
        """
        if scipy.sparse.issparse(self._coin):
            if not bool(self._marked):
                return self._coin

            # if coin was explicitly set,
            # and there are different coins for the marked vertices,
            # change them.
            def get_block(vertex):
                g = self._graph
                neighbors = g.neighbors(vertex)
                # TODO: this technique wont work after the behavior of
                # neighbors() change.
                a1 = g.arc_number((vertex, neighbors[0]))
                a2 = g.arc_number((vertex, neighbors[-1]))
                # arc order may change
                start = min(a1, a2)
                end = max(a1, a2) + 1

                return scipy.sparse.csr_array(self._coin[start:end,
                                                         start:end])

            num_vert = self._graph.number_of_vertices()
            degree = self._graph.degree
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

    def _set_evolution(self):
        # TODO: Check if matrix is sparse in pynelibna interface
        # TODO: Check if matrices are deleted from memory and GPU.
        U = None

        S = self.get_shift()
        C = self.get_coin()

        if nbl.get_hpc() is not None:
            from warnings import warn
            warn('HPC sparse matrix multiplication is not implemented. '
                 + 'Using standard scipy multiplication instead.')
            # # TODO: implement sparse matrix multiplication with hpc
            # S = S.todense()
            # C = C.todense()

            # nbl_S = nbl.send_matrix(S)
            # del S
            # nbl_C = nbl.send_matrix(C)
            # del C
            # nbl_C = nbl.multiply_matrices(nbl_S, nbl_C)

            # del nbl_S

            # U = nbl.retrieve_matrix(nbl_C)
            # del nbl_C
            # U = scipy.sparse.csr_array(U)

        U = S @ C

        self._evolution = U
        return U

    def set_evolution(self, **kwargs):
        """
        Set the evolution operator.

        Establishes the shift operator, coin operator,
        and the marked vertices.
        They are set using the appropriate ``**kwargs``.
        If ``**kwargs`` is empty, the default arguments are used.

        Subsequently, the evolution operator is constructed by 
        multiplying the shift and coin operators. 
        If the coin operator is given as an explicit matrix, 
        its definition remains unaltered 
        even in the presence of marked vertices. However, if the coin 
        operator is defined using coin names, any marked vertices will 
        prompt an update to the coin operator. Specifically, the coin names 
        for each marked vertex will default to a replacement with -I, 
        unless an alternative substitution is provided.

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
        get_evolution

        Notes
        -----
        The evolution operator is given by

        .. math::
           U = SC

        where :math:`S` is the shift operator, and
        :math:`C` is the coin operator. If there are
        any marked vertices, the coin operator is
        modified accordingly [1]_.

        When the coin operator is set as an explicit matrix, it remains 
        unaltered by marked vertices. However, if it's not provided in 
        matrix form (e.g., as a list of coins), the coin for each marked 
        vertex will be substituted based on the most recent 
        :meth:`set_marked` invocation.


        .. warning::
            Sparse matrix multipliation is not supported yet.
            Converting all matrices to dense.
            Then converting back to sparse.
            This uses unnecessary memory and computational time.


        References
        ----------
        .. [1] R. Portugal. "Quantum walks and search algorithms",
            2nd edition, Springer, 2018.
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

        self._set_shift(**S_kwargs)
        self._set_coin(**C_kwargs)
        self._set_marked(**R_kwargs)
        self._set_evolution()

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
        probability
        simulate

        Notes
        -----
        The probability for a given vertex :math:`u` is calculated as the sum of the
        absolute squares of the amplitudes of the arcs originating from :math:`u`.
        More precisely, if the state of the walker is
        
        .. math::
            \sum_{(v, w) \in \mathcal{A}(\vec G)} \alpha_{v,w} \ket{v,w},
        
        where :math:`\mathcal{A}(\vec G)` is the set of arcs of
        the symmetric directed graph formed by
        replacing each edge in :math:`G` with two arcs, one for each direction,
        then the probability associated with vertex :math:`u` is given by
        
        .. math::
            \sum_{v \in N(u)}|\alpha_{u, v}|^2,
        
        with :math:`N(u)` being the set of out neighbors of :math:`u`.
        A loop at :math:`u` is the arc :math:`(u,u)`. 
        
        The probability distribution, which is returned by this
        method as a ``numpy.ndarray``, is the collection of these
        probabilities for all vertices.
        """
        try:
            states.shape == 1
        except:
            states = np.array(states, copy=False)

        if len(states.shape) == 1:
            states = np.array([states], copy=False)


        graph = self._graph
        num_vert = graph.number_of_vertices()
        prob = np.array([[Coined._elementwise_probability(
                              states[i, graph.arcs_with_tail(v)]).sum()
                          for v in range(num_vert)]
                         for i in range(len(states))])

        return prob

    def state(self, entries):
        """
        Generates a valid state.

        The state corresponds to the walker being in a superposition
        of the ``entries``.
        The final state is normalized in order to be a unit vector.

        Parameters
        ----------
        entries : list of entry
            Each entry is a tuple (or array).
            An entry can be specified in two different ways:
            ``(amplitude, (tail, head))``, and
            ``(amplitude, arc_number)``.

            amplitude
                The amplitude of the given entry.
            tail
                The vertex corresponding to the position of the walker
                in the superposition.
                In other words, the tail of the arc.
                The tuple ``(tail, head)`` must be a valid arc.
            head
                The vertex to which the coin is pointing.
                The tuple ``(tail, head)`` must be a valid arc.
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

        >>> psi = qw.state([(1, (0, 1)), [1, 1], (1, 2)])
        >>> psi1 = qw.state([(1, ([0, 0], [0, 1])),
        ...                  [1, [(0, 0), (dim - 1, 0)]],
        ...                  (1, (0, 1))])
        >>> psi2 = qw.state([(1, [[0, 0], [0, 1]]),
        ...                  [1, ((0, 0), (dim - 1, 0))],
        ...                  (1, [0, 1])])
        >>> np.all(psi == psi1)
        True
        >>> np.all(psi1 == psi2)
        True
        """
        if len(entries) == 0:
            raise TypeError("Entries were not specified.")

        dtype = np.array([entry[0] for entry in entries]).dtype
        dtype = (complex if np.issubdtype(dtype, np.complexfloating)
                 else float)
        state = np.zeros(self.hilb_dim, dtype=dtype)

        for ampl, arc in entries:
            state[self._graph.arc_number(arc)] = ampl

        return self._normalize(state)

    def ket(self, arc):
        r"""
        Create a computational basis state.

        Parameters
        ----------
        arc: int or tuple of int
            The ket label.
            There are two different labels acceptable.

            (tail, head)
                The arc notation.
            arc_number
                The label of the arc.
                The number according to the computational basis order.
        """
        ket = np.zeros(self.hilb_dim, dtype=float)
        ket[self._graph.arc_number(arc)] = 1

        return ket

    def uniform_state(self, vertices=None, arcs=None):
        r"""
        Create a uniform state.

        The uniform state is a unit vector with entries 
        that have the same real amplitudes.
        If both ``vertices is None`` and ``arcs is None``,
        create the uniform superposition of all arcs.
        Otherwise,
        create the uniform superposition of all arcs in
        ``arcs`` and all the arcs with tail in ``vertices``.

        Parameters
        ----------
        vertices: list of vertices, default=None
            If ``vertices is not None``,
            every arc with tail in ``vertices`` is added to
            the uniform superposition.

        arcs: list of arcs, default=None
            If ``arcs is not None``,
            every arc in ``arcs`` is added to
            the uniform superposition.

        Returns
        -------
        :obj:`numpy.ndarray`

        Notes
        -----
        An example of the uniform state is

        .. math::
            \ket{d} = \frac{1}{\sqrt{N}} \sum_{i = 0}^{N - 1} \ket{i}

        where :math:`N` represents the dimension of the Hilbert space, and 
        :math:`i` is a label within the graph. 
        In the continuous-time quantum walk model, 
        :math:`i` corresponds to the label of a vertex, 
        while in the coined quantum walk model, 
        :math:`i` is the label of an arc.
        """
        if vertices is None and arcs is None:
            return (np.ones(self.hilb_dim, dtype=float)
                    / np.sqrt(self.hilb_dim))

        # uniform superposition of the given arcs
        state = np.zeros(self.hilb_dim)
        if arcs is not None:
            state[[self._graph.arc_number(a) for a in arcs]] = 1

        # uniform superposition of the given vertices
        if vertices is not None:
            for v in vertices:
                state[self._graph.arcs_with_tail(v)] = 1

        return state / np.sqrt(np.sum(state))

    def probability(self, states, vertices):
        r"""
        Computes the sum of probabilities for the specified vertices.
        
        Computes the probability of the walker being located on a
        vertex within the set of provided vertices, given that the walk 
        is on specified states.
        
        Parameters
        ----------
        states : :class:`numpy.ndarray`
            The state(s) used to compute the probability.
            ``states`` can be a single state or a list of states.
        
        vertices: list of int
           The subset of vertices. 
        
        Returns
        -------
        probabilities : float or :class:`numpy.ndarray`
            float:
                If ``states`` is a single state.
            :class:`numpy.ndarray`:
                If ``states`` is a list of states,
                ``probabilities[i]`` is the probability
                corresponding to the ``i``-th state.
        
        See Also
        --------
        simulate
        
        Notes
        -----
        
        The probability of finding the walker on vertex 
        :math:`v`, given the state of the walk
        :math:`|\psi \rangle`, is calculated as

        .. math::
            \sum_{\substack{a\in{\mathcal{A}}\\ \operatorname{tail}(a)=v}} \, 
            \left|{\langle a} | {\psi \rangle}\right|^2,

        where :math:`\mathcal{A}` denotes the set of arcs.        
        """
        return super().probability(states, vertices)
