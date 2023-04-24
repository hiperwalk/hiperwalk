from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse
import inspect
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
        self._oracle = None
        self._evolution_operator = None

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
        if not isinstance(adj_matrix, scipy.sparse.csr_array):
            raise TypeError(
                "Invalid `adj_matrix` type."
                + " Expected 'scipy.sparse.csr_array',"
                + " but received " + str(type(adj_matrix)) + '.'
            )
        if (len(adj_matrix.shape) != 2
            or adj_matrix.shape[0] != adj_matrix.shape[1]
        ):
            raise ValueError(
                "`adj_matrix` is not a square matrix."
            )

        self.adj_matrix = adj_matrix
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

    @abstractmethod
    def oracle(self, vertices=0):
        r"""
        Create the oracle that marks the given vertices.

        The oracle is set to be used for constructing the
        evolution operator.
        If ``vertices=[]`` no oracle is created and
        ``None`` is returned.
        For coherence, the previous evolution operator is unset.

        Parameters
        ----------
        vertices : int, array_like, default=0
            ID(s) of the vertex (vertices) to be marked.

        Returns
        -------
        :class:`scipy.sparse.csr_array`
        """
        raise NotImplementedError()

    @abstractmethod
    def evolution_operator(self, vertices=[], **kwargs):
        """
        Create the standard evolution operator.

        The evolution operator is saved to be used during the simulation.

        Parameters
        ----------
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
        raise NotImplementedError()

    @staticmethod
    def _elementwise_probability(elem):
        # This is more efficient than:
        # (np.conj(elem) * elem).real
        # elem.real**2 + elem.imag**2
        return elem.real*elem.real + elem.imag*elem.imag

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
            ``probabilities[i]`` is the probability of the
            walker beign found at vertex ``i``.

        See Also
        --------
        simulate
        """
        if len(states.shape) == 1:
            states = [states]

        prob = list(map(BaseWalk._elementwise_probability, states))
        prob = np.array(prob)

        return prob

    def _time_range_to_tuple(self, time_range):
        r"""
        Clean and format ``time_range`` to ``(start, end, step)`` format.

        See :meth:`simulate` for valid input format options.

        Raises
        ------
        ValueError
            If ``time_range`` is in an invalid input format.
        """

        if not hasattr(time_range, '__iter__'):
            time_range = [time_range]

        if len(time_range) == 1:
            start = end = step = time_range[0]
        elif len(time_range) == 2:
            start = 0
            end = time_range[0]
            step = time_range[1]
        else:
            start = time_range[0]
            end = time_range[1]
            step = time_range[2]
        
        time_range = [start, end, step]

        if start < 0 or end < 0 or step <= 0:
            raise ValueError(
                "Invalid 'time_range' value."
                + "'start' and 'end' must be non-negative"
                + " and 'step' must be positive."
            )
        if start > end:
            raise ValueError(
                "Invalid `time_range` value."
                + "`start` cannot be larger than `end`."
            )

        return time_range

    def _normalize(self, state, error=1e-16):
        norm = np.linalg.norm(state)
        if 1 - error <= norm and norm <= 1 + error:
            return state
        return state / norm

    def state(self, entries):
        """
        Generates a valid state.

        The state corresponds to the walker being in a superposition
        of the ``entries`` with the given amplitudes.

        The state is normalized in order to be unitary.

        Parameters
        ----------
        entries : list of 2-tuples
            Each entry is a 2-tuple with format ``(amplitude, vertex)``.

        Returns
        -------
        :class:`numpy.array`
        """
        # TODO benchmark with list comprehension

        # checking if there is a complex entry
        has_complex = np.any([ampl.imag != 0 for ampl, _ in entries])
        state = np.zeros(self.hilb_dim,
                         dtype=complex if has_complex else float)
        for ampl, v in entries:
            state[v] = ampl

        return self._normalize(state)

    def _pyneblina_imported(self):
        """
        Expects pyneblina interface to be imported as nbl
        """
        return ('qwalk._pyneblina_interface' in sys_modules
                and 'nbl' in locals())


    def simulate(self, time_range=None, initial_condition=None,
                 evolution_operator=None, hpc=True):
        r"""
        Simulates the quantum walk.

        Simulates the quantum walk applying the evolution operator
        multiple times to the initial condition.
        If ``evolution_operator=None``,
        uses the previously set evolution operator.
        Otherwise, uses the one sent as argument.

        The given (or set) evolution operator is interpreted as
        being a single step.

        Parameters
        ----------
        time_range : int, tuple of int, default=None
            Describes at which time instants the state must be saved.
            It can be specified in three different ways.
            
            * end
                Save the state at time ``end``.
                Only the final state is saved.

            * (end, step)
                Saves each state from time 0 to time ``end`` (inclusive)
                that is multiple of ``step``.

            * (start, end, step)
                Saves every state from time ``start`` (inclusive)
                to time ``end`` (inclusive)
                that is multiple of ``step``.

        initial_condition : :class:`numpy.array`, default=None
            The initial condition which the evolution operator
            is going to be applied to.

        evolution_operator : :class:`scipy.sparse.csr_array`, default=None
            The evolution operator.
            If ``None``, the previously set evolution operator is used.

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
            If any of the following occurs
            * ``time_range=None``.
            * ``initial_condition=None``.
            * ``evolution_operator=None`` and it was no set previously.

        See Also
        --------
        evolution_operator
        state

        Notes
        -----
        The walk is simulated by applying the
        evolution operator to the initial condition multiple times.
        The maximum and intermediate applications
        are describred by ``time``.

        Examples
        --------
        If ``time_range=(0, 13, 3)``, the saved states will be:
        the initial state (0), the intermediate states (3, 6, and 9),
        and the final state (12).
        """
        ############################################
        ### Check if simulation was set properly ###
        ############################################
        if time_range is None:
            raise ValueError(
                "``time_range` not specified`. "
                + "Must be an int or tuple of int."
            )

        for e in time_range:
            if not isinstance(e, int):
                raise ValueError("`time_range` has non-int entry.")

        if initial_condition is None:
            raise ValueError(
                "``initial_condition`` not specified. "
                + "Expected a np.array."
            )

        if len(initial_condition) != self.hilb_dim:
            raise ValueError(
                "Initial condition has invalid dimension. "
                + "Expected an np.array with length " + str(self.hilb_dim)
            )

        if self._evolution_operator is None and evolution_operator is None:
            raise ValueError(
                "Evolution Operator was not set. "
                + "Did you forget to call the evolution_operator() method"
                + " or to pass it as argument?"
            )

        if evolution_operator is not None:
            if evolution_operator.shape != (self.hilb_dim, self.hilb_dim):
                raise ValueError(
                    "Evolution operator has incorret dimensions."
                    + " Expected shape is "
                    + str((self.hilb_dim, self.hilb_dim))
                )
            prev_U = self._evolution_operator
            self._evolution_operator = evolution_operator

        ###########################
        ### Auxiliary functions ###
        ###########################

        def __prepare_engine(self):
            if __debug__:
                print("Preparing engine")

            if hpc:
                self._simul_mat = nbl.send_matrix(
                    self._evolution_operator)
                self._simul_vec = nbl.send_vector(
                    self._initial_condition)

            else:
                self._simul_mat = self._evolution_operator
                self._simul_vec = initial_condition

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
                    self._simul_vec = nbl.multiply_matrix_vector(
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
                ret = nbl.retrieve_vector(self._simul_vec)
            else:
                ret = self._simul_vec

            if __debug__:
                print("Done\n")

            return ret


        ###############################
        ### simulate implemantation ###
        ###############################

        start, end, step = self._time_range_to_tuple(time_range)
        
        if hpc and not self._pyneblina_imported():
            if __debug__:
                print("IMPORTING PYNEBLINA")
            from . import _pyneblina_interface as nbl

        __prepare_engine(self)

        # number of states to save
        num_states = int(end/step) + 1
        num_states -= (int((start - 1)/step) + 1) if start > 0 else 0

        # create saved states matrix
        # TODO: error: if initial condition is int and
        # evolution operator is float, dtype is complex
        dtype = (initial_condition.dtype if
            initial_condition.dtype == self._evolution_operator.dtype
            else complex
        )
        saved_states = np.zeros(
            (num_states, initial_condition.shape[0]), dtype=dtype
        )
        state_index = 0 # index of the state to be saved

        # if save_initial_state:
        if start == 0:
            saved_states[0] = initial_condition.copy()
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

        if evolution_operator is not None:
            self._evolution_operator = prev_U

        return saved_states

    def _get_valid_kwargs(self, method):
        return inspect.getargspec(method)[0][1:]

    def _filter_valid_kwargs(self, kwargs, valid_kwargs):
        return {k : kwargs.get(k) for k in valid_kwargs if k in kwargs}
                #if kwargs.get(k) is not None}
