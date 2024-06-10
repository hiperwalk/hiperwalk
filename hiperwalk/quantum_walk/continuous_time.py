import numpy as np
import scipy.sparse
import scipy.linalg
from .quantum_walk import QuantumWalk
from . import _pyneblina_interface as nbl

class ContinuousTime(QuantumWalk):
    r"""
    Manage instances of continuous-time quantum walks on graphs.
    
    For further implementation details and theoretical background, 
    refer to the Notes Section.

    Parameters
    ----------
    graph :
        Graph on which the quantum walk takes place.
        There are two acceptable inputs:

        * Simple graph (:class:`hiperwalk.Graph`);
        * Weighted graph (:class:`hiperwalk.WeightedGraph`);

    **kwargs : optional
        Additional arguments to set the Hamiltonian and evolution operator.

    See Also
    --------
    set_hamiltonian
    set_evolution

    Notes
    -----
    The continuous-time quantum walk model represents quantum particles 
    evolving on a graph in continuous time, as directed by the Schrödinger 
    equation. The Hamiltonian is usually chosen as the adjacency matrix 
    or the Laplacian of the graph. A positive parameter gamma acts 
    as a weighting factor for the Hamiltonian, adjusting the walk's 
    spreading rate. When marked vertices are present, 
    the Hamiltonian is suitably modified.
    
    The computational basis associated with a graph 
    :math:`G(V, E)` comprising :math:`n` vertices
    :math:`v_0, \ldots, v_{n-1}` is spanned by the states 
    :math:`\ket{i}` for
    :math:`0 \leq i < n`, where
    :math:`\ket{i}` describes the walker's position
    as vertex :math:`v_i`.
    
    The adjacency matrix of :math:`G(V, E)` is the 
    :math:`n`-dimensional matrix :math:`A` such that
    
    .. math::
        A_{i,j} = \begin{cases}
            1, \text{ if } v_i \text{ is adjacent to } v_j,\\
            0, \text{ otherwise.}
        \end{cases}
        
    Similarly, the Laplacian matrix is defined as
    
    .. math::
        L_{i,j} = \begin{cases}
            \text{degree}(v_i), \text{ if } i=j,\\
            -1, \text{ if } i\neq j \text{ and }
                v_i \text{ is adjacent to } v_j,\\
            0, \text{ otherwise.}
        \end{cases}

    The Hamiltonian's formulation is detailed in 
    :meth:`hiperwalk.ContinuousTime.set_hamiltonian`, 
    depending on the choice between the adjacency 
    or Laplacian matrix, along with the positioning 
    of the marked vertices.

    The :class:`hiperwalk.ContinuousTime` class enables the simulation of
    real Hamiltonians.
    A particular Hamiltonian :math:`H`,
    can be simulated by creating a :class:`hiperwalk.WeightedGraph`
    with adjacency matrix :math:`C` such that :math:`H = -\gamma C`.
    Additionally, the Laplacian matrix is computed 
    as :math:`D - A`, with :math:`D` being 
    the degree matrix. 
    See :meth:`hiperwalk.Graph.adjacency_matrix`
    and :meth:`hiperwalk.Graph.laplacian_matrix`.
    
    For a comprehensive understanding of continuous-time quantum 
    walks, consult reference [1]_. 
    To examine the differences between utilizing 
    the adjacency matrix and the Laplacian matrix, 
    refer to reference [2]_.
    
    References
    ----------
    .. [1] E. Farhi and S. Gutmann. "Quantum computation and decision trees". 
        Physical Review A, 58(2):915–928, 1998. ArXiv:quant-ph/9706062.
        
    .. [2] T. G. Wong, L. Tarrataca, and N. Nahimov. Laplacian versus adjacency 
    	matrix in quantum walk search. Quantum Inf Process 15, 4029-4048, 2016.
    """

    _valid_kwargs = dict()

    def __init__(self, graph=None, **kwargs):

        super().__init__(graph=graph)

        # create attributes
        self.hilb_dim = self._graph.number_of_vertices()

        self._gamma = None
        self._hamil_type = None
        self._terms = None
        self._hamiltonian = None

        self._time = None

        # import inspect

        if not bool(ContinuousTime._valid_kwargs):
            # assign static attribute
            ContinuousTime._valid_kwargs = {
                'gamma': ContinuousTime._get_valid_kwargs(
                    self._set_gamma),
                'marked': ContinuousTime._get_valid_kwargs(
                    self._set_marked),
                'evolution': ContinuousTime._get_valid_kwargs(
                    self._set_evolution),
                'time': ContinuousTime._get_valid_kwargs(
                    self._set_time),
                }

        self.set_evolution(**kwargs)

    def _set_gamma(self, gamma=0.1):
        if gamma is None or gamma.imag != 0:
            raise TypeError("Value of 'gamma' is not float.")

        if self._gamma != gamma:
            self._gamma = gamma
            return True
        return False

    def set_gamma(self, gamma=0.1):
        r"""
        Set gamma.
        
        Parameter gamma is used in the definition of the
        Hamiltonian to determine the hopping probability per unit 
        of time. Upon setting gamma, both the Hamiltonian and evolution 
        operators are updated.

        Parameters
        ----------
        gamma : float, default=0.1
            The value of gamma.

        Raises
        ------
        TypeError
            If ``gamma`` is ``None`` or complex.
        ValueError
            If ``gamma < 0``.
        """
        self.set_hamiltonian(gamma=gamma,
                             type=self._hamil_type,
                             marked=self._marked)

    def get_gamma(self):
        r"""
        Retrieve the value of gamma used in
        the definition of the Hamiltonian.

        Returns
        -------
        float
        
        See Also
        --------
        set_gamma
        """
        return self._gamma

    def set_marked(self, marked=[]):
        self.set_hamiltonian(gamma=self._gamma,
                             type=self._hamil_type,
                             marked=marked)

    def _set_hamiltonian(self, gamma=0.1, type='adjacency', marked=[]):
        update = self._set_gamma(gamma)
        update = self._set_hamiltonian_type(type) or update
        update = self._set_marked(marked) or update

        if update:
            if self._hamil_type == 'adjacency':
                H = -self._gamma * self._graph.adjacency_matrix()
            else:
                H = -self._gamma * self._graph.laplacian_matrix()

            # creating oracle
            if len(self._marked) > 0:
                data = np.ones(len(self._marked), dtype=np.int8)
                oracle = scipy.sparse.csr_array(
                        (data, (self._marked, self._marked)),
                        shape=(self.hilb_dim, self.hilb_dim))

                H -= oracle

            self._hamiltonian = H

        return update

    def set_hamiltonian(self, gamma=0.1, type="adjacency", marked=[]):
        r"""
        Set the Hamiltonian.

        The Hamiltonian takes the form of ``-gamma*C`` wheres
        ``C`` is either the adjacency matrix or the Laplacian matrix.
        If marked vertices are specified, the Hamiltonian 
        is modified as described in the Notes section. 
        Once the Hamiltonian has been established, 
        the evolution operator is updated accordingly. 

        Parameters
        ----------
        gamma : float, default=0.1
            The value of gamma.

        type: {'adjacency', 'laplacian'}
            Two types of Hamiltonian are used:
            ``'A'`` is shorthand for ``'adjacency'`` (default).
            ``'L'`` is shorthand for ``'laplacian'``.

        marked : list of vertices, default=[]
            List of vertices to be marked.
            If empty list, no vertex is marked.

        Raises
        ------
        TypeError
            If ``gamma`` is ``None`` or complex.
        ValueError
            If ``gamma < 0``.

        See Also
        --------
        set_gamma
        set_hamiltonian_type
        set_marked
        set_evolution

        Notes
        -----
        The Hamiltonian is given by [1]_ [2]_

        .. math::
            H = -\gamma C  - \sum_{m \in M} \ket m \bra m,

        where :math:`C` is either the adjacency matrix or
        the Laplacian matrix.
        The set :math:`M` specifies the marked vertices via
        the argument ``marked=M``. For instance, ``marked={0}``
        specifies that :math:`v_0` is the marked vertex.
        The default is :math:`M=\emptyset`.
        
        References
        ----------
        .. [1] E. Farhi and S. Gutmann.
            "Quantum computation and decision trees". 
            Physical Review A, 58(2):915–928, 1998.
            ArXiv:quant-ph/9706062.
        
        .. [2] A. M. Childs and J. Goldstone.
            "Spatial search by quantum walk",
            Phys. Rev. A 70, 022314, 2004.
        """
        self.set_evolution(time=self._time,
                           terms=self._terms,
                           gamma=gamma,
                           type=type,
                           marked=marked)

    def get_hamiltonian(self):
        r"""
        Retrieve the Hamiltonian.

        Returns
        -------
        :class:`scipy.sparse.csr_array`

        See Also
        --------
        set_hamiltonian
        """
        return self._hamiltonian

    def _set_hamiltonian_type(self, type='adjacency'):
        if type.upper() == 'A':
            type = 'adjacency'
        elif type.upper() == 'L':
            type = 'laplacian'

        if type != self._hamil_type:
            self._hamil_type = type
            return True

        return False

    def set_hamiltonian_type(self, type='adjacency'):
        r"""
        Set the type of the Hamiltonian.
        
        Parameters
        ----------
        type: {'adjacency', 'laplacian'}
            Two types of Hamiltonian are used:
            ``'A'`` is shorthand for ``'adjacency'``.
            ``'L'`` is shorthand for ``'laplacian'``.
        """
        self.set_hamiltonian(gamma=self._gamma,
                             type=type,
                             marked=self._marked)

    def get_hamiltonian_type():
        r"""
        Retrieve the type of the Hamiltonian.
        
        See Also
        --------
        set_hamiltonian_type
        
        Returns
        -------
        type: {'adjacency', 'laplacian'}
        """
        return self._hamil_type


    def _set_time(self, time=1):
        if time is None or time < 0:
            raise ValueError(
                "Expected non-negative `time` value."
            )

        if time != self._time:
            self._time = time
            return True
        return False

    def get_time(self):
        r"""
        Return the time used to construct the evolution operator.

        Returns
        -------
        float
        """
        return self._time

    def set_time(self, time=1):
        r"""
        Set a time instant.

        Defines a time t and calculates the evolution operator U(t) at
        the specified time (see Notes).

        Parameters
        ----------
        time : float, default=1

        Raises
        ------
        ValueError
            If ``time < 0``.

        See Also
        --------
        set_evolution

        Notes
        -----
        The evolution operator is given by

        .. math::
            U(t) = 	\text{e}^{-\text{i}tH},

        where :math:`H` is the Hamiltonian, and
        :math:`t` is the time.
        """
        self.set_evolution(time=time,
                           terms=self._terms,
                           gamma=self._gamma,
                           type=self._hamil_type,
                           marked=self._marked)

    def _set_terms(self, terms=21):
        if self._terms != terms:
            self._terms = terms
            return True
        return False

    def set_terms(self, terms=21):
        r"""
        Set the number of terms used to calculate the
        evolution operator as a power series.

        Parameters
        ----------
        terms : int, default=21
            Number of terms in the truncated Taylor series expansion.

        See Also
        --------
        set_evolution
        """
        self.set_evolution(time=self._time,
                           terms=terms,
                           gamma=self._gamma,
                           type=self._hamil_type,
                           marked=self._marked)

    def get_terms(self):
        r"""
        Retrieve the number of terms in the power series used to
        calculate the evolution operator.

        Returns
        -------
        int

        See Also
        --------
        set_terms
        set_evolution
        """
        return self._terms

    def _set_evolution(self, terms=21):
        r"""
        If this method is invoked,
        the evolution is recalculated
        """
        time = self._time

        if time == 0:
            self._evolution = np.eye(self.hilb_dim)
            return

        n = terms - 1
        H = self.get_hamiltonian()

        hpc = nbl.get_hpc()

        #TODO: when scipy issue 18086 is solved,
        # invoke scipy.linalg.expm to calculate power series
        def numpy_matrix_power_series(A, n):
            """
            I + A + A^2/2 + A^3/3! + ... + A^n/n!
            """
            U = np.eye(A.shape[0], dtype=A.dtype)
            curr_term = U.copy()
            for i in range(1, n + 1):
                curr_term = curr_term @ A / i
                U += curr_term

            return U

        if hpc is not None:
            nbl_U = nbl.matrix_power_series(-1j*time*H, n)
            U = nbl.retrieve_matrix(nbl_U)
        else:
            U = numpy_matrix_power_series(-1j*time*H.todense(), n)

        self._evolution = U

    def set_evolution(self, **kwargs):
        r"""
        Set the evolution operator.

        This method defines the evolution operator for a specified 
        ``time``.
        It first determines the 
        Hamiltonian and subsequently derives the evolution operator 
        via a truncated Taylor series. The default number of terms 
        in this series is set to ``terms=21``, which is adequate 
        when the Hamiltonian is derived from the adjacency matrix 
        and gamma is less than 1.

        Parameters
        ----------
        **kwargs :
            Additional arguments for setting Hamiltonian and time.
            If omitted, the default arguments are used.
            See :meth:`hiperwalk.ContinuousTime.set_hamiltonian`,
            :meth:`hiperwalk.ContinuousTime.set_time`, and
            :meth:`hiperwalk.ContinuousTime.set_terms`.

        See Also
        --------
        set_hamiltonian
        set_time
        set_terms

        Notes
        -----
        The evolution operator is given by

        .. math::
            U(t) = 	\text{e}^{-\text{i}tH},

        where :math:`H` is the Hamiltonian, and
        :math:`t` is the time.

        The :math:`n\text{th}` partial sum of
        the Taylor series expansion is given by
        
        .. math::
            \text{e}^{-\text{i}tH} \approx
            \sum_{j = 0}^{n-1} (-\text{i}tH)^j / j!

        where ``terms``:math:`=n`.
        This choice reflects default Python loops over integers,
        such as ``range`` and ``np.arange``.

        .. warning::
            For non-integer time (floating number),
            the result is approximate. It is recommended 
            to select a small time interval and perform 
            multiple matrix multiplications to minimize 
            rounding errors.
        """
        # TODO: Use ``scipy.linalg.expm`` when ``hpc=None`` once the
        #       `scipy issue 18086
        #       <https://github.com/scipy/scipy/issues/18086>`_
        #       is solved.

        def filter_and_call(method, update):
            valid = self._get_valid_kwargs(method)
            filtered = self._filter_valid_kwargs(kwargs, valid)
            return method(**filtered) or update

        update = filter_and_call(self._set_time, False)
        update = filter_and_call(self._set_hamiltonian, update)
        update = filter_and_call(self._set_terms, update)
        if (update):
            filter_and_call(self._set_evolution, update)
