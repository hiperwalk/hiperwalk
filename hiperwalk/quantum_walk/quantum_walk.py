from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse
import inspect
from sys import modules as sys_modules
from .._constants import __DEBUG__, PYNEBLINA_IMPORT_ERROR_MSG
from warnings import warn
from ..graph import Graph
try:
    from . import _pyneblina_interface as nbl
except ModuleNotFoundError:
    warn(PYNEBLINA_IMPORT_ERROR_MSG)

class QuantumWalk(ABC):
    """
    Basic (abstract) class for Quantum Walks.

    Basic methods and attributes used for implementing
    specific Quantum Walk models.

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

        The adjacency matrix.

    Attributes
    ----------
    hilb_dim : int, default=0
        Hilbert Space dimension.
        It must be updated by the subclass' ``__init__``.

    Warns
    -----
    If ``adjacency`` is set. It is deprecated. Use ``graph`` instead.

    Raises
    ------
    TypeError
        if ``adj_matrix`` is not an instance of
        :class:`scipy.sparse.csr_array`.

    Notes
    -----

    .. todo::
        * List the that methods must be overwritten.
        * Accept other types as ``graph`` such as numpy array

    """

    @abstractmethod
    def __init__(self, graph=None, adjacency=None, **kwargs):
        if adjacency is not None:
            if graph is None:
                graph = adjacency
            warn("'adjacency' parameter is deprecated. "
                 + "It will be removed in future versions.")

        if graph is None:
            raise TypeError('graph is None')

        self._marked = (self.set_marked(kwargs['marked'])
                        if 'marked' in kwargs
                        else self.set_marked([]))
        self._evolution = None

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
        if isinstance(graph, Graph):
            # DO STUFF
            self._graph = graph

        elif isinstance(graph, scipy.sparse.csr_array):
            if (len(graph.shape) != 2
                or graph.shape[0] != graph.shape[1]
            ):
                raise ValueError(
                    "Adjacency matrix is not a square matrix."
                )

            self._graph = Graph(graph)
        else:
            raise TypeError(
                "Invalid `graph` type."
                + " Expected 'hiperwalk.Graph' or "
                + "'scipy.sparse.csr_array', but received "
                + str(type(adj_matrix)) + ', instead.'
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

    def set_marked(self, marked=[]):
        r"""
        Sets marked vertices.

        Parameters
        ----------
        marked : list of int or int
            List of vertices to be marked.
            If empty list, no vertex is marked.
        """
        if not hasattr(marked, '__iter__'):
            marked = [marked]
        self._marked = set(marked)
        self._evolution = None

    def get_marked(self):
        r"""
        Gets marked vertices.

        Returns
        -------
        List of int
            List of marked vertices.
            If no vertex is marked, returns the empty list.
        """
        return list(self._marked)

    @abstractmethod
    def set_evolution(self, **kwargs):
        """
        Create the standard evolution operator.

        The evolution operator is saved to be used during the simulation.

        Parameters
        ----------
        **kwargs : dict, optional
            Additional arguments for constructing the evolution operator

        See Also
        --------
        simulate
        """
        raise NotImplementedError()

    @abstractmethod
    def get_evolution():
        r"""
        Returns the evolution operator in matricial form.
        """
        raise NotImplementedError()

    @staticmethod
    def _elementwise_probability(elem):
        # This is more efficient than:
        # (np.conj(elem) * elem).real
        # elem.real**2 + elem.imag**2
        return elem.real*elem.real + elem.imag*elem.imag

    def probability(self, states):
        r"""
        Compute the probability states.

        The probability of each entry of the state.

        Parameters
        ----------
        states : :class:`numpy.ndarray`
            The states used to compute the probabilities.

        Returns
        -------
        probabilities : :class:`numpy.ndarray`
            ``probabilities[i]`` is the probability of the ``i``-entry.

        See Also
        --------
        simulate
        """
        if len(states.shape) == 1:
            states = [states]

        prob = list(map(QuantumWalk._elementwise_probability, states))
        prob = np.array(prob)

        return prob

    def probability_distribution(self, states):
        r"""
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
        return self.probability(states)

    def _time_to_tuple(self, time):
        r"""
        Clean and format ``time`` to ``(start, end, step)`` format.

        See :meth:`simulate` for valid input format options.

        Raises
        ------
        ValueError
            If ``time`` is in an invalid input format.
        """

        if not hasattr(time, '__iter__'):
            time = [time]

        if len(time) == 1:
            start = end = step = time[0]
        elif len(time) == 2:
            start = 0
            end = time[0]
            step = time[1]
        else:
            start = time[0]
            end = time[1]
            step = time[2]
        
        time = [start, end, step]

        if start < 0 or end < 0 or step <= 0:
            raise ValueError(
                "Invalid 'time' value."
                + "'start' and 'end' must be non-negative"
                + " and 'step' must be positive."
            )
        if start > end:
            raise ValueError(
                "Invalid `time` value."
                + "`start` cannot be larger than `end`."
            )

        return time

    def _normalize(self, state, error=1e-16):
        norm = np.linalg.norm(state)
        if 1 - error <= norm and norm <= 1 + error:
            return state
        return state / norm

    def state(self, *args):
        """
        Generates a valid state.

        The state corresponds to the walker being in a superposition
        of the given labels with the given amplitudes.
        The state is normalized in order to be unitary.

        Parameters
        ----------
        *args
            Each entry is a 2-tuple or array with format
            ``(amplitude, vertex)``.

        Returns
        -------
        :class:`numpy.array`

        Examples
        --------
        .. todo::

            valid example
        """
        # TODO benchmark with list comprehension

        # checking if there is a complex entry
        has_complex = np.any([ampl.imag != 0 for ampl, _ in args])
        state = np.zeros(self.hilb_dim,
                         dtype=complex if has_complex else float)
        for ampl, v in args:
            state[v] = ampl

        return self._normalize(state)

    def ket(self, label):
        r"""
        Create a computational basis state.

        Parameters
        ----------
        label : int
            The ket label.

        Examples
        --------
        .. todo::
            valid examples
        """
        ket = np.zeros(self.hilb_dim, dtype=float)
        ket[label] = 1
        return ket

    def _pyneblina_imported(self):
        """
        Expects pyneblina interface to be imported as nbl
        """
        return ('hiperwalk.quantum_walk._pyneblina_interface'
                in sys_modules)

    def simulate(self, time=None, initial_condition=None, hpc=True):
        r"""
        Simulates the quantum walk.

        Simulates the quantum walk applying the evolution operator
        multiple times to the initial condition.

        Parameters
        ----------
        time : int, tuple of int, default=None
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
            * ``time=None``.
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
        If ``time=(0, 13, 3)``, the saved states will be:
        the initial state (0), the intermediate states (3, 6, and 9),
        and the final state (12).
        """
        ############################################
        ### Check if simulation was set properly ###
        ############################################
        if time is None:
            raise ValueError(
                "``time` not specified`. "
                + "Must be an int or tuple of int."
            )

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

        ###########################
        ### Auxiliary functions ###
        ###########################

        def __prepare_engine(self):
            if hpc:
                self._simul_mat = nbl.send_matrix(self._evolution)
                self._simul_vec = nbl.send_vector(initial_condition)

            else:
                self._simul_mat = self._evolution
                self._simul_vec = initial_condition

        def __simulate_step(self, step):
            """
            Apply the simulation evolution operator ``step`` times
            to the simulation vector.
            Simulation vector is then updated.
            """
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

        def __save_simul_vec(self):
            ret = None

            if hpc:
                # TODO: check if vector must be deleted or
                #       if it can be reused via neblina-core commands.
                ret = nbl.retrieve_vector(self._simul_vec)
            else:
                ret = self._simul_vec

            return ret


        ###############################
        ### simulate implemantation ###
        ###############################

        time = np.array(self._time_to_tuple(time))
        if self._evolution is None:
            self._evolution = self.get_evolution(hpc=hpc)

        if not np.all([e.is_integer() for e in time]):
            raise ValueError("`time` has non-int entry.")

        start, end, step = time

        
        if hpc and not self._pyneblina_imported():
            hpc = False

        __prepare_engine(self)

        # number of states to save
        num_states = int(end/step) + 1
        num_states -= (int((start - 1)/step) + 1) if start > 0 else 0

        # create saved states matrix
        # TODO: error: if initial condition is int and
        # evolution operator is float, dtype is complex
        dtype = (initial_condition.dtype if
            initial_condition.dtype == self._evolution.dtype
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

        return saved_states

    @staticmethod
    def _get_valid_kwargs(method):
        return inspect.getfullargspec(method)[0][1:]

    @staticmethod
    def _filter_valid_kwargs(kwargs, valid_kwargs):
        return {k : kwargs.get(k) for k in valid_kwargs if k in kwargs}
                #if kwargs.get(k) is not None}
