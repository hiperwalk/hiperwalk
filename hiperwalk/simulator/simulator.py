import numpy as np
import inspect
from sys import modules as sys_modules
from .._constants import PYNEBLINA_IMPORT_ERROR_MSG
try:
    from sys import path
    path.append('..')
    import _pyneblina_interface as nbl
except ModuleNotFoundError:
    from warnings import warn
    warn(PYNEBLINA_IMPORT_ERROR_MSG)

def pyneblina_imported():
    """
    Expects pyneblina interface to be imported as nbl
    """
    return ('hiperwalk.quantum_walk._pyneblina_interface'
            in sys_modules)

class Simulator():
    r"""
    TODO: docs
    """

    def __init__(self, evolution):
        self.set_evolution(evolution)

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

    def set_evolution(self, evolution=None, copy=True):
        """
        Create the standard evolution operator.

        The evolution operator is saved to be used during the simulation.

        Parameters
        ----------
        evolution: :class:`numpy.ndarray`
            The evolution operator.

        copy: bool, default=True
            If ``True``, a hard copy of the matrix is made.
            Otherwise, a pointer to the matrix is saved.

        Raises
        ------
        ValueError
            If ``evolution`` is not a matrix.

        See Also
        --------
        simulate
        """
        try:
            evolution.shape
        except AttributeError:
            evolution = np.array(evolution)

        if len(evolution.shape) != 2:
            raise ValueError("Expected a matrix.")

        self._evolution = np.copy(evolution) if copy else evolution

    def get_evolution(self, copy=True):
        r"""
        Returns the evolution operator.

        Parameters
        ----------
        copy: bool, default=True
            If ``True`` returns a hard copy.
            If ``False`` returns matrix pointer.

        Returns
        -------
        :class:`numpy.ndarray`.

        See Also
        --------
        set_evolution
        """
        return np.copy(self._evolution) if copy else self._evolution

    @staticmethod
    def time_to_tuple(time):
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

    ######################################
    ### Auxiliary Simulation functions ###
    ######################################

    def _prepare_engine(self, vector, hpc):
        if self._evolution is None:
            raise ValueError("Matrix not set.")

        if hpc:
            self._simul_mat = nbl.send_matrix(self._evolution)
            self._simul_vec = nbl.send_vector(vector)

        else:
            self._simul_mat = self._evolution
            self._simul_vec = vector

        dtype = (np.complex128 if (np.iscomplexobj(self._evolution)
                             or np.iscomplexobj(vector))
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
            # TODO: check if intermediate vectors are being freed
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

    def simulate(self, time=None, state=None, hpc=True,
                 initial_state=None):
        r"""
        Simulates the dynamics described by the evolution operator.

        The dynamics is simulated by applying the
        evolution operator to the initial ``state`` multiple times.
        The first, intermediate and last applications
        are describred by ``time``.

        .. deprecated: 2.0
            ``initial_state`` will be removed in version 2.1,
            it is replaced by ``state`` because the latter is more concise.

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

        state : :class:`numpy.array`, default=None
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

        Examples
        --------
        If ``time=(0, 13, 3)``, the saved states will be:
        the initial state (0), the intermediate states (3, 6, and 9),
        and the final state (12).
        """
        if initial_state is not None:
            from warnings import warn
            warn("Deprecation warning. `initial_state` is deprecated. "
                 + "Use `state` instead.")
            if state is None:
                state = initial_state
        ############################################
        ### Check if simulation was set properly ###
        ############################################
        if time is None:
            raise ValueError(
                "``time` not specified`. "
                + "Must be an int or tuple of int."
            )

        if state is None:
            raise ValueError(
                "``state`` not specified. "
                + "Expected a np.array."
            )

        if len(state) != self._evolution.shape[1]:
            raise ValueError(
                "Vector has invalid dimension. "
                + "Expected an np.array with length "
                + str(self._evolution.shape[1])
            )

        ###############################
        ### simulate implemantation ###
        ###############################

        time = np.array(Simulator.time_to_tuple(time))

        if not np.all([e.is_integer() for e in time]):
            raise ValueError("`time` has non-int entry.")

        start, end, step = time

        if hpc and not pyneblina_imported():
            hpc = False

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

        # TODO: check if state is freed from neblina core
        if hpc:
            del self._simul_mat
            del self._simul_vec
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
