from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse

class BaseWalk(ABC):
    """
    Base class for Quantum Walks.

    Base methods and attributes used for implementing
    specific Quantum Walk models.

    .. todo::
        The following methods must be overwritten.

    Parameters
    ----------
    adj_matrix : :class:`scipy.sparse.csr_array`
        Adjacency matrix of the graph on
        which the quantum walk occurs.

    Attributes
    ----------
    hilb_dim : int, default=0
        Hilbert Space dimension.
        It must be updated by the subclass' ``__init__``.

    Raises
    ------
    TypeError
        if ``adj_matrix`` is not an instance of
        :class:`scipy.sparse.csr_array`.
    """

    @abstractmethod
    def __init__(self, adj_matrix):
        self._states = None
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

        self.hilb_dim = 0

    def uniform_state(self):
        r"""
        Generate the uniform state.

        The state is constructed based on the ``hilb_dim`` attribute.
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
        """
        return (np.ones(self.hilb_dim, dtype=float)
                / np.sqrt(self.hilb_dim))

    def uniform_initial_condition(self):
        """
        Alias for :obj:`uniform_state`.
        """
        return self.uniform_state()

    @abstractmethod
    def oracle(self, vertices):
        r"""
        Create the oracle that marks the given vertices.

        Parameters
        ----------
        vertices : array_like
            ID(s) of the vertex (vertices) to be marked.

        Returns
        -------
        :class:`scipy.sparse.csr_array`
        """
        return None

    @abstractmethod
    def evolution_operator(self, hpc=False, **kwargs):
        """
        Create the standard evolution operator.

        Parameters
        ----------
        hpc : bool, default=False
            Whether or not evolution operator should be
            constructed using nelina's high-performance computating.

        **kwargs : dict, optional
            Additional arguments for constructing the evolution operator

        Returns
        -------
        U_w : :class:`scipy.sparse.csr_array`
            The evolution operator.
        """
        return None

    @abstractmethod
    def search_evolution_operator(self, vertices, hpc=False,
                                  **kwargs):
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

        **kwargs : dict, optional
            Additional arguments for constructing the evolution operator

        Returns
        -------
        U : :class:`scipy.sparse.csr_array`
            The search evolution operator.

        See Also
        --------
        evolution_operator
        oracle
        """
        return None

    @staticmethod
    def _elementwise_probability(elem):
        # This is more efficient than:
        # (np.conj(elem) * elem).real
        # elem.real**2 + elem.imag**2
        return elem.real*elem.real + elem.imag*elem.imag

    @abstractmethod
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
        return None

    def prepare_walk(self, evolution_operator, initial_condition,
                     num_steps, save_interval=0):
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
            Must be a positive integer.

        save_interval : int, default=0
            Number of applications of the evolution operation
            before saving an intermediate state.
            If ``save_interval=0``,
            only saves the initial and final states.
            Otherwise, saves the initial state, the intermediate
            states and the final state.

        Raises
        ------
        ValueError
            If ``num_steps`` is not a positive integer.
            If ``evolution_operator`` or ``initial_condition``'s
            dimensions are inconsistent with the Hilbert Space
            (``self.hilb_dim``).

        See Also
        --------
        simulate_walk

        Notes
        -----
        .. todo::
            Implement assertion of arguments.
            For example: check if evolution operator is unitary.

        Examples
        --------
        If ``num_steps=10`` and ``save_interval=3``,
        the returned saved states are:
        the initial state, the intermediate states (3, 6, and 9),
        and the final state (10).

        >>> qw.prepare_walk(U, psi0, 10, 3)
        """

        if num_steps <= 0 :
            raise ValueError('Expected a positive integer for num_steps.')

        if evolution_operator.shape != (self.hilb_dim, self.hilb_dim):
            raise ValueError(
                'Evolution operator has wrong dimension.'
                + 'It has shape ' + str(evolution_operator.shape)
                + ', but shape ' + str((self.hilb_dim, self.hilb_dim))
                + ' was expected.'
            )

        if initial_condition.shape != (self.hilb_dim,):
            raise ValueError(
                'Initial condition has wrong dimension.'
                + 'It has shape ' + str(initial_condition.shape)
                + ', but shape ' + str((self.hilb_dim,))
                + ' was expected.'
            )

        self._evolution_operator = evolution_operator

        # number of states to save
        if save_interval <= 0:
            save_interval = num_steps
        num_states = int(np.ceil(num_steps / save_interval))
        num_states += 1 # saves initial state

        # create saved states matrix
        dtype = (initial_condition.dtype if
            initial_condition.dtype == evolution_operator.dtype
            else complex
        )
        self._states = np.zeros(
            (num_states, self.hilb_dim), dtype=dtype
        )
        self._states[0] = initial_condition

        self._num_steps = num_steps
        self._save_interval = save_interval

    def simulate_walk(self, hpc=False):
        r"""
        Simulates quantum walk.
        
        It is necessary to call :obj:`prepare_walk` beforehand.

        Parameters
        ----------
        hpc : bool, default=False
            Whether or not to use neblina's high-performance computing
            to perform matrix multiplications.
            If ``hpc=False`` uses python.

        Returns
        -------
        Returns array with saved states.

        See Also
        --------
        prepare_walk
        """

        if hpc:
            from . import _pyneblina_interface as nbl

        def _prepare_engine(self):
            if hpc:
                self._simul_mat = nbl.send_sparse_matrix(
                    self._evolution_operator)
                self._simul_vec = nbl.send_vector(
                    self._states[0])

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
                    self._simul_vec, self.hilb_dim, delete_vector=True
                )
            return self._simul_vec

        _prepare_engine(self)

        # Index of the state to be saved.
        # Initial condition is always saved in the 0-th index.
        state_index = 1 

        # simulate walk / apply evolution operator
        for i in range(int(self._num_steps / self._save_interval)):
            _simulate_steps(self, self._save_interval)
            self._states[state_index] = _save_simul_vec(self)
            state_index += 1

        if self._num_steps % self._save_interval > 0:
            _simulate_steps(self, self._num_steps % self._save_interval)
            self._states[state_index] = _save_simul_vec(self)

        # TODO: free vector from neblina core
        self._simul_mat = None
        self._simul_vec = None

        return self._states

    def plot_probability(self, probabilities, **kwargs):
        """
        Plot probabilities of saved states.

        It is strongly recommend that subclasses overwrite this method.

        Parameters
        ----------
        probabilities : :class:`numpy.ndarray`
            The probabilities of the states.

        **kwargs : dict, optional
            Optional keywords arguments used for customizing the plot.

        Notes
        -----
        Currently calls :obj:`plot.plot_probability_distribution`,
        respecting its ``kwargs``.

        By default, ``self.adj_matrix`` is sent as a ``kwargs`` entry.

        See Also
        --------
        :obj:`plot.plot_probability_distribution`
        """
        import plot as hpw_plot

        if 'adj_matrix' not in kwargs:
            kwargs['adj_matrix'] = self.adj_matrix
        hpw_plot.plot_probability_distribution(probabilities, **kwargs)

    # TODO:
    def plot_states(self, states):
        """
        Plot amplitudes  of saved states.

        It is strongly recommend that subclasses overwrite this method.

        Parameters
        ----------
        states : :class:`numpy.ndarray` 
            The states to be plotted.

        Warnings
        --------
        .. todo::
            Not implemented.
        """
        # Function to plot each state and the direction it is pointing to.
        raise NotImplementedError("Implement general function")
