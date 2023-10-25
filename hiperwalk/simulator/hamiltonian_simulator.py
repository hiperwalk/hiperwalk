import numpy as np
from scipy import sparse
from .simulator import *

class HamiltonianSimulator(Simulator):
    r"""
    Class for simulating the dynamics of a Hamiltonian.

    Parameters
    ----------
    hamiltonian : matrix
        A skew-Hermitian matrix.
        The Hamiltonian describes the dynamics of a system.

    time : float
        Time value used to construct the evolution operator.

    terms : int, default=21
        Number of terms in power series expansion.

    hpc : bool, default=True
        Determines whether or not to use neblina HPC 
        functions to generate the evolution operator.

    Notes
    -----
    The dynamics of the Hamiltonian is simulating by
    an unitary operator called the evolution operator,

    .. math::
        U = e^{-iHt},

    where :math:`U` is the evolution operator,
    :math:`i` is the imaginary unit,
    :math:`H` is the Hamiltonian, and
    :math:`t` is a timestamp.
    The evolution operator is unitary as long as
    the Hamiltonian is skew-Hermitian.
    """

    def __init__(self, **kwargs):
        self._time = None
        self._hamiltonian = None
        self._terms = None
        self.set_evolution(**kwargs)

    def set_time(self, time=None, hpc=True):
        r"""
        Alias for :meth:`set_evolution`,
        but only the ``time`` parameter is changed.

        Parameters
        ----------
        time : float
            Time value used to construct the evolution operator.

        ** kwargs :
            Additional arguments for updating the evolution operator.
            For example, whether to use neblina HPC or not.
            See :meth:`set_evolution` for more options.

        Raises
        ------
        ValueError
            If ``time < 0``.

        See Also
        --------
        set_evolution
        """
        self.set_evolution(time=time, 
                           hamiltonian=self._hamiltonian,
                           terms=self._terms,
                           hpc=hpc)

    def get_time(self):
        r"""
        Return the ``time`` value used to calculate the evolution operator.

        Returns
        -------
        float
        """
        return self._time

    def set_hamiltonian(self, hamiltonian=None, hpc=True):
        r"""
        Set the Hamiltonian and recalculate the evolution operator.

        Parameters
        ----------
        hamiltonian : matrix
            A skew-Hermitian matrix.
            The Hamiltonian describes the dynamics of a system.

        hpc : bool, default=True
            Determines whether or not to use neblina HPC 
            functions to generate the evolution operator.

        See Also
        --------
        set_evolution
        """
        self.set_evolution(time=self._time,
                           hamiltonian=hamiltonian,
                           terms=self._terms,
                           hpc=hpc)

    def get_hamiltonian(self, copy=True):
        r"""
        Return the ``hamiltonian`` matrix used to
        calculate the evolution operator.

        Parameters
        ----------
        copy : bool, default=True
            If ``True``, a copy of the matrix is returned.
            Otherwise, a pointer to the matrix is returned.

        Returns
        -------
        :class:`numpy.ndarray` or :class:`scipy.csr_matrix`.
        """
        if not copy:
            return self._hamiltonian
        if sparse.issparse(self._hamiltonian):
            return sparse.csr_matrix.copy(self._hamiltonian)
        return np.copy(self._hamiltonian)

    def set_terms(self, terms=21, hpc=True):
        r"""
        Set the number of terms used to calculate the
        evolution operator as a power series.

        Parameters
        ----------
        terms : int, default=21
            Number of terms in power series expansion.

        hpc : bool, default = True
            Determines whether or not to use neblina HPC 
            functions to generate the evolution operator.

        See Also
        --------
        set_evolution
        """
        self.set_evolution(time=self._time,
                           hamiltonian=self._hamiltonian,
                           terms=terms,
                           hpc=hpc)

    def get_terms(self):
        r"""
        Number of terms in the power series used to
        calculate the evolution operator.

        Returns
        -------
        int

        See Also
        --------
        get_evolution
        """
        return self._terms

    def set_evolution(self, **kwargs):
        r"""
        Set the evolution operator.

        The evolution operator is completely described by
        the Hamiltonian and time.
        The evolution operator is constructed by
        calculating a power series.

        Parameters
        ----------
        **kwargs :
            Key arguments for setting Hamiltonian, time, and
            number of terms in the power series.
            See :meth:`hiperwalk.ContinuousTime.set_hamiltonian`,
            :meth:`hiperwalk.ContinuousTime.set_time`, and
            :meth:`hiperwalk.ContinuousTime.set_terms`.
            If omitted, the default arguments are used.

        See Also
        --------
        set_hamiltonian
        set_time
        set_terms

        Notes
        -----
        The evolution operator is given by

        .. math::
            U(t) = e^{-\text{i}tH},

        where :math:`H` is the Hamiltonian, and
        :math:`t` is the time.

        The Taylor series expansion is given by

        .. math::
            e^{-\text{i}tH} &= \sum_{j = 0}^{n} (\text{i}tH)^j / j!

        where :math:`n` is the number of terms minus 1
        (i.e. ``terms - 1``).

        .. warning::
            For non-integer time (floating number),
            the result is approximate. It is recommended 
            to select a small time interval and perform 
            multiple matrix multiplications to minimize 
            rounding errors.

        .. todo::
            Use ``scipy.linalg.expm`` when ``hpc=False`` once the
            `scipy issue 18086
            <https://github.com/scipy/scipy/issues/18086>`_
            is solved.
        """

        def filter_and_call(method, update):
            valid = self._get_valid_kwargs(method)
            filtered = self._filter_valid_kwargs(kwargs, valid)
            return method(**filtered) or update

        update = filter_and_call(self._set_time, False)
        update = filter_and_call(self._set_hamiltonian, update)
        update = filter_and_call(self._set_terms, update)
        if (update):
            filter_and_call(self._set_evolution, update)


    ############################################
    ### Auxiliary methods for _set_evolution ###
    ############################################

    def _set_time(self, time=None):
        if time is None or time < 0:
            raise ValueError(
                "Expected non-negative `time` value."
            )

        if time != self._time:
            self._time = time
            return True
        return False

    def _set_hamiltonian(self, hamiltonian=None):
        if id(self._hamiltonian) != id(hamiltonian):
            self._hamiltonian = hamiltonian
            return True
        return False

    def _set_terms(self, terms=21):
        if self._terms != terms:
            self._terms = terms
            return True
        return False

    def _set_evolution(self, hpc=True):
        r"""
        If this method is invoked,
        the evolution is recalculated
        """
        time = self._time

        if time == 0:
            try:
                self._evolution = np.eye(self._hamiltonian.shape[0])
            except:
                self._evolution = np.eye(len(self._hamiltonian))
            self._matrix = self._evolution
            return

        n = self._terms - 1
        H = self.get_hamiltonian()

        if hpc and not pyneblina_imported():
            hpc = False

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

        # determining the number of terms in power series
        max_val = np.max(np.abs(H))
        if max_val*time <= 1:
            if hpc:
                nbl_U = nbl.matrix_power_series(-1j*time*H, n)
            else:
                U = numpy_matrix_power_series(-1j*time*H.todense(), n)

        else:
            # if the order of magnitude is very large,
            # float point errors may occur
            if ((isinstance(time, int) or time.is_integer())
                and max_val <= 1
            ):
                new_time = 1
                num_mult = time - 1
            else:
                # TODO: assert precision
                new_time = max_val*time
                order = np.ceil(np.math.log(new_time, n))
                new_time /= 10**order
                num_mult = int(np.round(time/new_time)) - 1

            if hpc:
                new_nbl_U = nbl.matrix_power_series(
                        -1j*new_time*H, n)
                nbl_U = nbl.multiply_matrices(new_nbl_U, new_nbl_U)
                for i in range(num_mult - 1):
                    nbl_U = nbl.multiply_matrices(nbl_U, new_nbl_U)
            else:
                if sparse.issparse(H):
                    H = H.todense()
                U = numpy_matrix_power_series(-1j*new_time*H, n)
                U = np.linalg.matrix_power(U, num_mult + 1)

        if hpc:
            U = nbl.retrieve_matrix(nbl_U)

        self._evolution = U
        self._matrix = self._evolution

    ############################################
    ############################################
    ############################################

    def simulate(self, time=None, state=None, hpc=True,
                 initial_state=None):
        r"""
        Analogous to :meth:`hiperwalk.Simulator.simulate`,
        but accepts float entries for the ``time`` parameter.

        Parameters
        ----------
        time : float or tuple of floats
            This parameter is analogous to those in
            :meth:`hiperwalk.Simulator.simulate`,
            with the distinction that it accepts float inputs.

            If the ``step`` parameter specified, it is used to
            recalculate the evolution operator if needed.
            Otherwise, ``step`` is set to the value of
            :meth:`hiperwalk.HamiltonianSimulator.get_time`.

            The saved states are within the interval
            **[** ``start``, ``end`` **]**
            such that the timestamps are multiples of ``step``.

        Other Parameters
        ----------------
        See `hiperwalk.Simulator.simulate`.

        See Also
        --------
        set_time
        hiperwalk.Simulator.siulate
        """
        if time is None:
            raise ValueError(
                "Invalid `time_range`. Expected a float, 2-tuple, "
                + "or 3-tuple of float."
            )

        time = np.array(Simulator.time_to_tuple(time))

        self.set_time(time=time[2], hpc=hpc)

        # converting time_range to int
        tol = 1e-5
        time = [int(val/time[2])
                if int(val/time[2])
                   <= np.ceil(val/time[2]) - tol
                else int(np.ceil(val/time[2]))
                for val in time]

        saved_states = super().simulate(time=time, state=state, hpc=hpc,
                                        initial_state=initial_state)
        return saved_states

    def _number_to_valid_time(self, number):
        return number
