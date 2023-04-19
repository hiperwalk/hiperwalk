from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse
from sys import modules as sys_modules

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

    adj_matrix : :class:`scipy.sparse.csr_array`
        Adjacency matrix of the graph on
        which the quantum walk occurs.

    Raises
    ------
    TypeError
        if ``adj_matrix`` is not an instance of
        :class:`scipy.sparse.csr_array`.
    """

    @abstractmethod
    def __init__(self, adj_matrix):
        self._initial_condition = None
        self._evolution_operator = None
        self._oracle = None
        self._time = None # Accepts range-like values

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

        The state is *NOT* saved to be used for the simulation.

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
        Creates and sets the uniform state as the initial condition.

        The uniform condition is set to be used as simulation input.

        Returns
        -------
        :obj:`numpy.ndarray`

        See Also
        --------
        uniform_state
        """
        self._initial_condition = self.uniform_state()
        return self._initial_condition

    @abstractmethod
    def oracle(self, vertices=[0]):
        r"""
        Create the oracle that marks the given vertices.

        Parameters
        ----------
        vertices : array_like, default=[0]
            ID(s) of the vertex (vertices) to be marked.

        Returns
        -------
        :class:`scipy.sparse.csr_array`
        """
        return None

    def set_oracle(self, R):
        r"""
        Sets the oracle to be used for constructing
        the evolution operator.
        """
        self._oracle = R
        # indicates that the evolution operator must be reconstructed
        self._evolution_operator = None

    def get_oracle():
        r"""
        Returns the oracle used to construct the evolution operator.

        If ``None`` is returned,
        no oracle was used to construct the evolution operator
        or the oracle used is unknown.

        See Also
        --------
        evolution_operator
        """
        return self._oracle

    @abstractmethod
    def evolution_operator(self, hpc=True, vertices=[], **kwargs):
        """
        Create the standard evolution operator.

        The evolution operator is saved to be used during the simulation.

        Parameters
        ----------
        hpc : bool, default=True
            Whether or not evolution operator should be
            constructed using nelina's high-performance computating.

        vertices : array_like, default=[]
            The marked vertices IDs.
            See :obj:`oracle`'s ``vertices`` parameter.

        **kwargs : dict, optional
            Additional arguments for constructing the evolution operator

        Returns
        -------
        U : :class:`scipy.sparse.csr_array`
            The evolution operator.

        See Also
        --------
        oracle
        simulate
        """
        return None

    def set_evolution_operator(self, U):
        r"""
        Sets ``U`` as the evolution operator.
        It is used during the simulation.

        Parameters
        ----------
        U : :class:`scipy.sparse.csr_array`
            Evolution Operator.

        Raises
        ------
        ValueError
            If ``U`` has invalid dimensions.

        Notes
        -----
        .. todo::
            Check if ``U`` is unitary.
        """
        if U.shape != (self.hilb_dim, self.hilb_dim):
            raise ValueError(
                "Matrix `U` has invalid dimensions."
                + " Expected " + str((self.hilb_dim, self.hilb_dim))
                + " but received " + str(U.shape) + " instead."
            )

        self._evolution_operator = U
        # it is not known whether the oracle was used to
        # construct U
        self._oracle = None

    def get_evolution_operator():
        r"""
        Returns the evolution operator.

        Returns the evolution operator set to be used in
        the quantum walk simulation.
        If ``None`` is returned,
        no evolution operator was set.
        """
        return self._evolution_operator

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

    def _clean_time(self, time_range):
        r"""
        Clean and format ``time_range`` to ``(start, end, step)`` format.

        See :meth:`simulate_walk` for valid input format options.

        Raises
        ------
        ValueError
            If ``time_range`` is in an invalid input format.

        See Also
        --------
        time
        """

        if not hasattr(time_range, '__iter__'):
            time_range = [time_range]

        start = time_range[0]
        end = time_range[1] if len(time_range) >= 2 else start
        step = time_range[2] if len(time_range) >= 3 else 1

        time_range = (start, end, step)

        if start < 0 or end < 0 or step <= 0:
            raise ValueError(
                "Invalid 'time_range' value."
                + "'start' and 'end' must be non-negative"
                + " and 'step' must be positive."
            )

        if (end - start)%step != 0:
            raise ValueError(
                "Invalid 'time_range' value."
                + "'start' and 'end' are not a multiple of "
                + "'step' time_range apart"
            )

        return time_range


    def time(self, time_range):
        r"""
        Configures the quantum walk simulation time.

        The simulation will save the states at the
        time instants given by ``time_range``.

        Parameters
        ----------

        time_range : float, 2-tuple or 3-tuple
            Describes at which time instants the state must be saved.
            It can be specified in three different ways.
            
            * ``end``
                The evolution operator is applied
                ``end`` times. Only the final state is saved.

            * ``(start, end)``
                Saves each state from the
                ``start``-th to the ``end``-th application
                of the evolution operator.
                That is, ``[start, start + 1, ..., end - 1, end]``.

            * ``(start, end, step)``
                Saves every state from the
                ``start``-th to the ``end``-th application
                of the evolution operator separated by
                ``step`` applications.
                That is, ``[start, start + step, ..., end - step, end]``.

        Raises
        ------
        ValueError
            If ``time_range=(start, end, step)`` and
            ``end`` cannot be reached from ``start`` after a
            multiple ``step`` applications.
            In other words, if
            ``end - start`` is not a multiple of ``step``.
            
            It is also raised if any of the following occurs.
            
            * ``start < 0``,
            * ``end < 0``,
            * ``step <= 0``.

        See Also
        --------
        simulate

        Examples
        --------
        If ``time_range=(0, 12, 3)``, the saved states will be:
        the initial state (0), the intermediate states (3, 6, and 9),
        and the final state (12).

        >>> qw.time((0, 12, 3))
        """
        self._time = self._clean_time(time_range)

    def get_time():
        r"""
        Returns the configured time.

        Notes
        -----
        If ``None`` is returned, time was not configured.
        """
        return self._time


    def _normalize(self, state, error=1e-16):
        norm = np.linalg.norm(state)
        if 1 - error <= norm and norm <= 1 + error:
            return state
        return state / norm

    @abstractmethod
    def state(self, entries, **kwargs):
        """
        Generates a valid state.

        The state corresponds to the walker being in a superposition
        of the ``entries`` with the given amplitudes.

        The final state is normalized in order to be unitary.

        The state is not saved to be used for the simulation.

        Parameters
        ----------
        entries :
            Entries of the state to be generated.
            Valid entries vary according to the quantum walk model.

        **kwargs : dict, optionaly
            Additional arguments for generating a valid state.
        """

        return None


    def initial_condition(self, entries, **kwargs):
        r"""
        Generates a valid initial condition.

        The generated state is saved to be used for the simulation.

        See Also
        --------
        state
        """
        self._initial_condition = self.state(entries, **kwargs)
        return self._initial_condition

    def set_initial_condition(self, state):
        r"""
        Sets the initial conditions.

        Saves the ``state`` to be used as
        the simulation initial condition.

        Parameters
        ----------
        state : 
            State to be set as initial condition.

        Raises
        ------
        ValueError
            If ``state`` has not the right dimension.
        """
        if state.shape != (self.hilb_dim, ) :
            raise ValueError(
                "`state` has invalid shape. "
                + "Expected (" + str(self.hilb_dim) + ",)."
            )

        self._initial_condition = state


    def get_initial_condition(self):
        r"""
        Returns the current initial condition.

        If ``None``, no initial condition was set.
        """
        return self._initial_condition

    def _pyneblina_imported(self):
        return 'qwalk._pyneblina_interface' in sys_modules


    def simulate(self, hpc=True):
        r"""
        Simulates the quantum walk using the
        evolution operator, initial condition and
        time previously set.

        Parameters
        ----------
        hpc : bool, default=True
            Whether or not to use neblina's high-performance computing
            to perform matrix multiplications.
            If ``hpc=False`` uses standalone python.

        Returns
        -------
        states : :class:`numpy.ndarray`.
            States saved during simulation where
            ``states[i]`` corresponds to the ``i``-th saved state.

        Raises
        ------
        ValueError
            If the time, evolution operator, or initial condition
            were not set previously.

        See Also
        --------
        time
        evolution_operator
        initial_condition

        Notes
        -----
        The walk is simulated by applying the
        evolution operator to the initial condition multiple times.
        The maximum and intermediate applications
        are describred by ``time``.
        """
        ############################################
        ### Check if simulation was set properly ###
        ############################################
        if self._time is None:
            raise ValueError(
                "Time was not set."
            )
        if self._evolution_operator is None:
            raise ValueError(
                "Evolution Operator was not set."
            )
        if self._initial_condition is None:
            raise ValueError(
                "Initial condition was not set."
            )

        ###########################
        ### Auxiliary functions ###
        ###########################

        def __prepare_engine(self):
            if __debug__:
                print("Preparing engine")

            if hpc:
                self._simul_mat = nbl.send_sparse_matrix(
                    self._evolution_operator)
                self._simul_vec = nbl.send_vector(
                    self._initial_condition)

            else:
                self._simul_mat = self._evolution_operator
                self._simul_vec = self._initial_condition

            if __debug__:
                print("Done\n")

        def __simulate_step(self, step):
            """
            Apply the simulation evolution operator ``step`` times
            to the simulation vector.
            Simulation vector is then updated.
            """
            if __debug__:
                print("Simulating steps")

            if hpc:
                # TODO: request multiple multiplications at once
                #       to neblina-core
                # TODO: check if intermediate states are being freed
                for i in range(step):
                    self._simul_vec = nbl.multiply_sparse_matrix_vector(
                        self._simul_mat, self._simul_vec)
            else:
                for i in range(step):
                    self._simul_vec = self._simul_mat @ self._simul_vec

                # TODO: compare with numpy.linalg.matrix_power

            if __debug__:
                print("Done\n")

        def __save_simul_vec(self):
            if __debug__:
                print("Saving simulated vec")

            ret = None

            if hpc:
                # TODO: check if vector must be deleted or
                #       if it can be reused via neblina-core commands.
                ret = nbl.retrieve_vector(
                    self._simul_vec, self._initial_condition.shape[0]
                )
            else:
                ret = self._simul_vec

            if __debug__:
                print("Done\n")

            return ret


        ####################################
        ### simulate_walk implemantation ###
        ####################################

        start, end, step = self._time
        
        if hpc and not self._pyneblina_imported():
            if __debug__:
                print("IMPORTING PYNEBLINA")
            from . import _pyneblina_interface as nbl

        __prepare_engine(self)

        # number of states to save
        num_states = int((end - start)/step) + 1

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
        if start == 0:
            saved_states[0] = self._initial_condition.copy()
            state_index += 1
            num_states -= 1

        # simulate walk / apply evolution operator
        if start > 0:
            __simulate_step(self, start - step)

        for i in range(num_states):
            __simulate_step(self, step)
            saved_states[state_index] = __save_simul_vec(self)
            state_index += 1

        # TODO: free vector from neblina core
        self._simul_mat = None
        self._simul_vec = None

        return saved_states
