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
    Abstract class for Quantum Walks.

    Basic methods and attributes used for implementing
    specific quantum walk models.

    Parameters
    ----------
    graph
        Graph on which the quantum walk takes place.
        It can be the graph itself (:class:`hiperwalk.graph.Graph`) or
        its adjacency matrix (:class:`scipy.sparse.csr_array`).

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
    def __init__(self, graph=None, **kwargs):
        if graph is None:
            raise TypeError('graph is None')

        self._marked = []
        if 'marked' in kwargs:
            self._update_marked(kwargs['marked'])

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
        Creates a uniform state.

        The state is constructed based on the ``hilb_dim`` attribute.
        The uniform state is a unit vector with entries 
        that have the same real amplitudes.

        Returns
        -------
        :obj:`numpy.ndarray`

        Notes
        -----
        An example of the uniform state is

        .. math::

            \ket{d} = \frac{1}{\sqrt{N}} \sum_{i = 0}^{N - 1} \ket{i}

        where :math:`N` is the dimension of the Hilbert space.
        """
        return (np.ones(self.hilb_dim, dtype=float)
                / np.sqrt(self.hilb_dim))

    def _update_marked(self, marked=[]):
        self._marked = set(map(self._graph.vertex_number, marked))
        self._marked = np.sort(list(self._marked))

    def set_marked(self, marked=[], **kwargs):
        r"""
        Sets marked vertices.

        After the marked elements are changed,
        the evolution operator is updated accordingly.

        Parameters
        ----------
        marked : list of vertices
            List of vertices to be marked.
            If empty list, no vertex is marked.

        ** kwargs :
            Additional arguments for updating the evolution operator.
            For example, whether to use neblina HPC or not.
            See :meth:`set_evolution` for more options.

        See Also
        --------
        set_evolution
        """
        self._update_marked(marked=marked)
        self._update_evolution(**kwargs)

    def get_marked(self):
        r"""
        Gets marked vertices.

        Returns
        -------
        List of int
            List of marked vertices.
            If no vertex is marked, returns the empty list.
        """
        return self._marked

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

    def get_evolution(self):
        r"""
        Returns the evolution operator.

        Returns
        -------
        :class:`numpy.ndarray`.

        See Also
        --------
        set_evolution
        """
        return self._evolution

    @staticmethod
    def _elementwise_probability(elem):
        # This is more efficient than:
        # (np.conj(elem) * elem).real
        # elem.real**2 + elem.imag**2
        return elem.real*elem.real + elem.imag*elem.imag

    def success_probability(self, states):
        r"""
        Computes the success probability for the given state(s).

        The success probability is the probability of the
        walker being found in any of the marked vertices.
        If no vertex is marked,
        the success probability is 0.

        Parameters
        ----------
        states : :class:`numpy.ndarray`
            The state(s) used to compute the probability.
            ``states`` can be a single state or a list of states.

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
        probability
        """
        if len(self._marked) > 0:
            return self.probability(states, self._marked)

        if len(states.shape) > 1:
            return np.zeros(states.shape[0])
        return 0

    def probability(self, states, vertices):
        r"""
        Computes the sum of probabilities of the given vertices.

        Computes the probability of the walker being found on a
        subset of the vertices in the given state(s).

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
        """
        single_state = len(states.shape) == 1
        if single_state:
            states = np.array([states])

        probs = self.probability_distribution(states)
        probs = np.array([
                    np.sum(probs[i, vertices])
                    for i in range(len(states))
                    ])

        return probs[0] if single_state else probs

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
        The probability for a given vertex is the absolute square of
        its amplitude.
        That is, for an arbitrary superposition

        .. math::
            \sum_{v \in V} \alpha_v \ket{v},

        the probability associated with vertex :math:`v` is
        :math:`|\alpha_v|^2`.
        """
        single_state = len(states.shape) == 1
        if single_state:
            states = [states]

        prob = list(map(QuantumWalk._elementwise_probability, states))
        prob = np.array(prob)

        return prob[0] if single_state else prob

    @staticmethod
    def _time_to_tuple(time):
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
            An entry may be an array of such tuples.

        Returns
        -------
        :class:`numpy.array`

        Notes
        -----
        If there are repeated vertices,
        the amplitude of the last entry is used.

        Examples
        --------
        The following commands generate the same state.

        >>> psi = qw.state([1, 0], (1, 1), (1, 2)) #doctest: +SKIP
        >>> psi1 = qw.state([1, 0], [(1, 1), (1, 2)]) #doctest: +SKIP
        >>> psi2 = qw.state(([1, 0], (1, 1)), (1, 2)) #doctest: +SKIP
        >>> psi3 = qw.state([[1, 0], (1, 1), (1, 2)]) #doctest: +SKIP
        >>> np.all(psi == ps1) #doctest: +SKIP
        True
        >>> np.all(psi1 == ps2) #doctest: +SKIP
        True
        >>> np.all(psi2 == ps3) #doctest: +SKIP
        True
        """
        if len(args) == 0:
            raise TypeError("Entries were not specified.")

        state = [0] * self.hilb_dim

        for arg in args:
            if hasattr(arg[0],'__iter__'):
                for ampl, v in arg:
                    state[self._graph.vertex_number(v)] = ampl
            else:
                ampl, v = arg
                state[self._graph.vertex_number(v)] = ampl

        state = np.array(state)
        return self._normalize(state)

    def ket(self, label):
        r"""
        Creates a state of the computational basis.

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


    ######################################
    ### Auxiliary Simulation functions ###
    ######################################

    def _prepare_engine(self, initial_state, hpc):
        if self._evolution is None:
            self._evolution = self.get_evolution(hpc=hpc)

        if hpc:
            self._simul_mat = nbl.send_matrix(self._evolution)
            self._simul_vec = nbl.send_vector(initial_state)

        else:
            self._simul_mat = self._evolution
            self._simul_vec = initial_state

        dtype = (np.complex128 if (np.iscomplexobj(self._evolution)
                             or np.iscomplexobj(initial_state))
                 else np.double)

        return dtype

    def _simulate_step(self, step, hpc):
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

    def _save_simul_vec(self, hpc):
        ret = None

        if hpc:
            # TODO: check if vector must be deleted or
            #       if it can be reused via neblina-core commands.
            ret = nbl.retrieve_vector(self._simul_vec)
        else:
            ret = self._simul_vec

        return ret


    def simulate(self, time=None, initial_state=None, hpc=True):
        r"""
        Simulates the quantum walk.

        Simulates the quantum walk applying the evolution operator
        multiple times to the initial state.

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

        initial_state : :class:`numpy.array`, default=None
            The initial state which the evolution operator
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
            * ``initial_state=None``.
            * ``evolution_operator=None`` and it was no set previously.

        See Also
        --------
        evolution_operator
        state

        Notes
        -----
        The walk is simulated by applying the
        evolution operator to the initial state multiple times.
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

        if initial_state is None:
            raise ValueError(
                "``initial_state`` not specified. "
                + "Expected a np.array."
            )

        if len(initial_state) != self.hilb_dim:
            raise ValueError(
                "Initial condition has invalid dimension. "
                + "Expected an np.array with length " + str(self.hilb_dim)
            )

        ###############################
        ### simulate implemantation ###
        ###############################

        time = np.array(QuantumWalk._time_to_tuple(time))

        if not np.all([e.is_integer() for e in time]):
            raise ValueError("`time` has non-int entry.")

        start, end, step = time

        
        if hpc and not self._pyneblina_imported():
            hpc = False

        dtype = self._prepare_engine(initial_state, hpc)

        # number of states to save
        num_states = int(end/step) + 1
        num_states -= (int((start - 1)/step) + 1) if start > 0 else 0

        saved_states = np.zeros(
            (num_states, initial_state.shape[0]), dtype=dtype
        )
        state_index = 0 # index of the state to be saved

        # if save_initial_state:
        if start == 0:
            saved_states[0] = initial_state.copy()
            state_index += 1
            num_states -= 1

        # simulate walk / apply evolution operator
        if start > 0:
            self._simulate_step(start - step, hpc)

        for i in range(num_states):
            self._simulate_step(step, hpc)
            saved_states[state_index] = self._save_simul_vec(hpc)
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

    @staticmethod
    def _pop_valid_kwargs(kwargs, valid_kwargs):
        return {k : kwargs.pop(k) for k in valid_kwargs if k in kwargs}

    def hilbert_space_dimension(self):
        """
        Returns dimension of the Hilbert space.
        """
        return self.hilb_dim
