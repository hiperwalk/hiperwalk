from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse
import inspect
from sys import modules as sys_modules
from .._constants import __DEBUG__
from warnings import warn
from ..graph import Graph
import scipy.optimize
from . import _pyneblina_interface as nbl

class QuantumWalk(ABC):
    """
    Abstract class for Quantum Walks.

    Basic methods and attributes used for implementing
    specific quantum walk models.

    Parameters
    ----------
    graph: :class:`hiperwalk.graph.Graph`
        Graph on which the quantum walk takes place.

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

        self._graph = graph
        self.hilb_dim = 0


    def uniform_state(self):
        r"""
        Create a uniform state.

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

        where :math:`N` represents the dimension of the Hilbert space, and 
        :math:`i` is a label within the graph. 
        In the continuous-time quantum walk model, 
        :math:`i` corresponds to the label of a vertex, 
        while in the coined quantum walk model, 
        :math:`i` is the label of an arc.
        """
        return (np.ones(self.hilb_dim, dtype=float)
                / np.sqrt(self.hilb_dim))

    def _set_marked(self, marked=[]):
        if (id(marked) != id(self._marked)):
            self._marked = set(map(self._graph.vertex_number, marked))
            self._marked = np.sort(list(self._marked))
            return True
        return False

    def set_marked(self, marked=[], **kwargs):
        r"""
        Set the marked vertices.

        When the marked vertices are updated using this method,
        the evolution operator adjusts accordingly.

        Parameters
        ----------
        marked : list of vertices
            List of vertices to be marked.
            If empty list, no vertex is marked.

        ** kwargs :
            Additional arguments for updating the evolution operator.
            For example, whether to use HPC or not.
            See :meth:`set_evolution` for more options.

        See Also
        --------
        set_evolution
        """
        self.set_evolution(marked=marked, **kwargs)

    def get_marked(self):
        r"""
        Retrieve the marked vertices.

        Returns
        -------
        List of int
            List of marked vertices.
            If no vertex is marked, returns the empty list.

        See Also
        --------
        set_marked
        """
        return self._marked

    @abstractmethod
    def set_evolution(self, **kwargs):
        """
        Set the standard evolution operator.

        This evolution operator is stored for use in subsequent simulations.

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
        Retrieve the evolution operator.

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
        states : :class:`numpy.ndarray` or list of :class:`numpy.ndarray`
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
        if isinstance(states, set):
            raise TypeError("Type 'set' is not supported.")

        if len(self._marked) > 0:
            return self.probability(states, self._marked)

        try:
            states.shape == 1
        except TypeError:
            states = np.array(states, copy=False)

        if states.shape == 1:
            return np.zeros(size)
        return 0

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
        """
        if isinstance(states, set):
            raise TypeError("Type 'set' is not supported.")

        single_state = False
        try:
            len(states[0])
        except TypeError:
            single_state = True
            states = np.array([states], copy=False)

        probs = self.probability_distribution(states)
        probs = np.array([
                    np.sum(probs[i, vertices])
                    for i in range(len(states))
                    ])

        return probs[0] if single_state else probs

    def probability_distribution(self, states):
        r"""
        Compute the probability distribution of the given state(s).

        The probability distribution is determined by the
        state of the walk. It describes the likelihood of the
        walker being located at each vertex for the specified state(s).

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
        If the Hilbert space is spanned by the set of vertices,
        the probability of finding the walker on a given vertex 
        is the absolute square of its amplitude.
        That is, for an arbitrary superposition

        .. math::
            \sum_{v \in V} \alpha_v \ket{v},

        the probability associated with vertex :math:`v` is
        :math:`|\alpha_v|^2`. The calculation of the probability
        depends on the specifics of the quantum walk model when
        the Hilbert space is not spanned by the set of vertices.
        """
        single_state = False
        try:
            len(states[0])
        except TypeError:
            single_state = True
            states = np.array([states])

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

    def state(self, entries):
        """
        Generates a state in the Hilbert space.

        The state corresponds to the walker being in a superposition
        of the given labels with the given amplitudes.
        The state is normalized in order to have a unitary norm.

        Parameters
        ----------
        entries : list of entry
            Each entry is a 2-tuple or array with format
            ``(amplitude, label)``.
            That is, an amplitude and the corresponding label of
            the computational basis.

        Returns
        -------
        :class:`numpy.array`

        Notes
        -----
        If there are repeated vertices,
        the amplitude of the last entry is used.

        Examples
        --------
        .. TODO:
            Consider the Graph...

        The following commands generate the same state.

        >>> psi = qw.state([[1, 0], [1, 1], [1, 2]])
        >>> psi1 = qw.state([[1, 0], (1, 1), (1, 2)])
        >>> psi2 = qw.state(([1, 0], (1, 1), (1, 2)))
        >>> psi3 = qw.state(((1, 0), (1, 1), (1, 2)))
        >>> np.all(psi == ps1)
        True
        >>> np.all(psi1 == ps2)
        True
        >>> np.all(psi2 == ps3)
        True
        """
        if len(entries) == 0:
            raise TypeError("Entries were not specified.")

        state = np.zeros(self.hilb_dim)

        for ampl, vertex in entries:
            state[self._graph.vertex_number(vertex)] = ampl

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

    ######################################
    ### Auxiliary Simulation functions ###
    ######################################

    def _prepare_engine(self, state, hpc):
        if self._evolution is None:
            self._evolution = self.get_evolution()

        if hpc is not None:
            self._simul_mat = nbl.send_matrix(self._evolution)
            self._simul_vec = nbl.send_vector(state)

        else:
            self._simul_mat = self._evolution
            self._simul_vec = state

        dtype = (np.complex128 if (np.iscomplexobj(self._evolution)
                             or np.iscomplexobj(state))
                 else np.double)

        return dtype

    def _simulate_step(self, step, hpc):
        """
        Apply the simulation evolution operator ``step`` times
        to the simulation vector.
        Simulation vector is then updated.
        """
        if hpc is not None:
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

        if hpc is not None:
            # TODO: check if vector must be deleted or
            #       if it can be reused via neblina-core commands.
            ret = nbl.retrieve_vector(self._simul_vec)
        else:
            ret = self._simul_vec

        return ret



    def simulate(self, time=None, state=None, initial_state=None):
        r"""
        Simulates the quantum walk.

        The simulation progresses by iteratively applying 
        the evolution operator to the outcome of its prior 
        application, initiating from the specified initial state.

        Parameters
        ----------
        time : int, tuple of int, default=None
            Specifies the time instances when the state should be saved.
            It can be defined in three distinct ways:
            
            * end
                Saves the state at the ``end`` time.
                Only the final state is retained.

            * (end, step)
                Retains every state from time 0 to ``end`` (inclusive)
                that corresponds to a multiple of ``step``.

            * (start, end, step)
                Stores every state from time ``start`` (inclusive)
                to ``end`` (inclusive)
                that is a multiple of ``step``.

        state : :class:`numpy.array`, default=None
            The starting state onto which the evolution operator
            will be applied.

        initial_state :
            .. deprecated: 2.0
                ``initial_state`` will be removed in version 2.1,
                it is replaced by ``state`` because
                the latter is more concise.

        Returns
        -------
        states : :class:`numpy.ndarray`.
            States retained during the simulation where
            ``states[i]`` is the ``i``-th saved state.

        Raises
        ------
        ValueError
            Triggered if:
            * ``time=None``.
            * ``state=None``.
            * ``evolution_operator=None`` and hasn't been set before.

        See Also
        --------
        evolution_operator
        state

        Notes
        -----
        The states computed and saved during the simulation 
        are determined by the parameter ``time=(start,end,step)``.        
        
        The simulation of the walk is based on the expression
        :math:`|\psi(t)\rangle=U^t|\psi(\text{start})\rangle`, where
        :math:`|\psi(\text{start})\rangle` denotes the initial state.        
        The values for :math:`t` progress as 
        :math:`t=0,\,\text{step},\,2\cdot\text{step},...`, 
        until reaching the highest multiple of step that is less than or equal to 
        :math:`\text{end}-\text{start}`.
        
        Specifically, the simulation begins from the state 
        :math:`|\psi(\text{start})\rangle` and 
        sequentially calculates and saves states in the form of 
        :math:`|\psi(\text{start}+j\cdot\text{step})\rangle`,
        where :math:`j=0,1,...` and the maximum value of :math:`j`
        ensures that 
        :math:`\text{start}+j\cdot\text{step}\le \text{end}`.

        Examples
        --------
        Given ``time=(0, 13, 3)``, the saved states would include:
        the initial state (t=0), intermediate states (t=3, 6, and 9),
        and the concluding state (t=12).
        """
        ############################################
        ### Check if simulation was set properly ###
        ############################################
        if time is None:
            raise ValueError(
                "``time` not specified`. "
                + "Must be an int or tuple of int."
            )

        if initial_state is not None:
            warn("``initial_state`` will be removed in version 2.1,"
                 + "it is replaced by ``state`` because"
                 + "the latter is more concise.")
            if state is None:
                state = initial_state

        if state is None:
            raise ValueError(
                "``state`` not specified. "
                + "Expected a np.array."
            )

        if len(state) != self.hilb_dim:
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
        hpc = nbl.get_hpc()
        dtype = self._prepare_engine(state, hpc)

        # number of states to save
        num_states = int(end/step) + 1
        num_states -= (int((start - 1)/step) + 1) if start > 0 else 0

        saved_states = np.zeros(
            (num_states, state.shape[0]), dtype=dtype
        )
        state_index = 0 # index of the state to be saved

        # if save_state:
        if start == 0:
            saved_states[0] = state.copy()
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


    @staticmethod
    def fit_sin_squared(x, y):
        r"""
        Fit data to the squared sine function.

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            The domain values.
            It is assumed that the entries are evenly spaced.
            That is, ``x[i + 1] - x[i] == x[j + 1] - x[j]`` for
            any valid ``i`` and ``j``.

        y : :class:`numpy.ndarray`
            The image values evaluated at each ``x[i]``.
            It is required that ``y[i]`` corresponds to
            the evaluation at ``x[i]``.

        Returns
        -------
        d : dict
            It contains information about the best fit found.
            The dictionary keys are

            ``"fit function"`` :
                a pointer to the obtained sine squared
                function. Issuing ``d["fit function"](theta)`` evaluates
                the function at point ``theta``.

            ``"amplitude"``:
                amplitude of the obtained fit function.

            ``"angular frequency"`` :
                angular frequency of the obtained fit function.

            ``"phase shift"`` :
                phase shift of the obtained fit function.

            ``"vertical offset"`` :
                vertical offset of the obtained fit function.

            ``"frequency"`` :
                frequency of the obtained fit function.

            ``"period"`` :
                period of the obtained fit function.

        Notes
        -----
        The returned ``d["fit function"]`` is a pointer to

        .. code-block:: python

            def fit_func(theta):
                return d["amplitude"]*np.sin(
                           d["angular frequency"]*theta +
                           d["phase shift"]
                       )**2 + d["vertical offset"]


        The code was adapted from
        `https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy
        <https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy>`_
        """
        # uniform spacing is assumed
        fft_freq = np.fft.fftfreq(len(x), (x[1] - x[0]))
        abs_fft = abs(np.fft.fft(y))

        # excluding the zero frequency "peak", which is related to offset
        guess_freq = abs(fft_freq[np.argmax(abs_fft[1:]) + 1])
        guess_amp = 2*np.std(y) * np.sqrt(2)
        guess_offset = np.mean(y)
        guess = np.array([guess_amp, np.pi*guess_freq, 0, guess_offset])

        def sin_square(t, ampl, ang_freq, shift, vert_offset):
            return ampl*np.sin(ang_freq*t + shift)**2 + vert_offset
                
        opt_res, _ = scipy.optimize.curve_fit(sin_square, x, y, p0=guess)

        ampl, ang_freq, shift, vert_offset = opt_res
        freq = ang_freq/np.pi
        fitfunc = lambda x: ampl*np.sin(ang_freq*x + shift)**2 + vert_offset
        return {"amplitude": ampl,
                "angular frequency": ang_freq,
                "phase shift": shift,
                "vertical offset": vert_offset,
                "frequency": freq,
                "period": 1/freq,
                "fit function": fitfunc}

    def _number_to_valid_time(self, number):
        raise NotImplementedError()

    def _optimal_runtime(self, state, delta_time):
        r"""
        .. todo::
            Returns all arguments.
            It is used by optinal_runtime and max_p_succ to avoid
            redundant computation.
        """
        if state is None:
            state = self.uniform_state()

        N = self._graph.number_of_vertices()
        # if search algorithm takes O(N),
        # it is better to use classical computing.
        final_time = self._number_to_valid_time(N/2)
        states = self.simulate(time=(final_time, delta_time),
                               state=state)
        p_succ = self.success_probability(states)
        del states

        d = QuantumWalk.fit_sin_squared(
                np.arange(0, final_time + delta_time, delta_time),
                p_succ
            )
        t_opt = (np.pi/2 - d['phase shift']) / d['angular frequency']
        return self._number_to_valid_time(t_opt), p_succ


    def optimal_runtime(self, state=None, delta_time=1):
        r"""
        Find the optimal running time of a quantum-walk-based search.

        This method simulates the use of the ``set_evolution`` operator,
        taking the ``state`` as an input for the simulation. It then
        calculates the success probability for each intermediate state and fits
        these probabilities to a sine-squared function. The optimal running time
        corresponds to the point in the domain where the sine-squared function
        reaches its first peak.

        Parameters
        ----------
        state : :class:`numpy.ndarray`, default=None
            The state initial state for the simulation.
            If ``None``, uses the uniform state.

        delta_time :
            Time difference between two consecutive states
            to be saved by the simulation.
            See ``time`` argument in :meth:`simulate` for details.

        Returns
        -------
        int or float
            The optimal runtime that was found after
            fitting the sine-squared function.
            The returned type depends on the quantum walk model.


        See Also
        --------
        simulate
        uniform_state
        success_probability
        fit_sin_squared
        """
        t_opt, _ = self._optimal_runtime(state, delta_time)
        return t_opt

    def max_success_probability(self, state=None, delta_time=1):
        r"""
        Find the maximum success probability.
        
        This method returns the success probability that corresponds 
        to the optimal running time.

        Parameters
        ----------
        state : :class:`numpy.ndarray`, default=None
            The state initial state for the simulation.
            If ``None``, uses the uniform state.

        delta_time :
            Time difference between two consecutive states
            to be saved by the simulation.
            See ``time`` argument in :meth:`simulate` for details.

        Returns
        -------
        float
            The maximum success probability.

        See Also
        --------
        optimal_runtime
        simulate
        uniform_state

        Notes
        -----

        .. todo::
            If ``t_opt / delta_time`` is not close to an integer,
            the max success probability was not obtained in the simulation.
            The simulation must be rerun or interpolated.
        """
        t_opt, p_succ = self._optimal_runtime(state, delta_time)
        opt_index = int(t_opt / delta_time)
        # TODO: if t_opt / delta_time is not close to an integer,
        # max_sucess_probability is not in p_succ.
        # simulation must be rerun
        return p_succ[opt_index]
