import numpy as np
from scipy.sparse import issparse
from .simulator import *

class HamiltonianSimulator(Simulator):
    r"""
    TODO: docs
    """

    def __init__(self, **kwargs):
        self._time = None
        self._hamiltonian = None
        self._terms = None
        self.set_evolution(**kwargs)

    def set_matrix(self, **kwargs):
        r"""
        Alias for :meth:`set_evolution`.
        """
        self.set_evolution(**kwargs)

    def set_time(self, time=None, hpc=True):
        r"""
        Alias for :meth:`set_evolution`,
        only changing the ``time`` parameter.

        Parameters
        ----------
        time : float

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

        Notes
        -----
        The evolution operator is given by

        .. math::
            U(t) = e^{-\text{i}tH},

        where :math:`H` is the Hamiltonian, and
        :math:`t` is the time.
        """
        self.set_evolution(time=time, 
                           hamiltonian=self._hamiltonian,
                           terms=self._terms,
                           hpc=hpc)

    def get_time(self):
        return self._time

    def set_hamiltonian(self, hamiltonian=None, hpc=True):
        r"""
        Generates the evolution operator with the same
        previous parameters but changes the Hamiltonian.
        Hamiltonian is expected to be is kew-Hermitian.
        """
        self.set_evolution(time=self._time,
                           hamiltonian=hamiltonian,
                           terms=self._terms,
                           hpc=hpc)

    def get_hamiltonian(self, copy=True):
        r"""
        TODO
        """
        return np.copy(self._hamiltonian) if copy else self._hamiltonian

    def set_terms(self, terms=21, hpc=True):
        self.set_evolution(time=self._time,
                           hamiltonian=self._hamiltonian,
                           terms=terms,
                           hpc=hpc)

    def get_terms(self):
        return self._terms

    def set_evolution(self, time=None, hamiltonian=None, terms=21,
                      hpc=True):
        r"""
        Sets the evolution operator.

        Sets Hamiltonian and time.
        They are set using the appropriate ``**kwargs``.
        If ``**kwargs`` is empty, the default arguments are used.
        Then, the evolution operator is constructed using
        a Taylor series expansion.

        Parameters
        ----------
        hpc : bool, default = True
            Determines whether or not to use neblina HPC 
            functions to generate the evolution operator.

        terms : int
            Number of terms in Taylor series expansion.

        **kwargs :
            Additional arguments for setting Hamiltonian and time.
            See :meth:`hiperwalk.ContinuousTime.set_hamiltonian`, and
            :meth:`hiperwalk.ContinuousTime.set_time`.
            If omitted, the default arguments are used.

        See Also
        --------
        set_hamiltonian
        set_time

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

        update = self._set_time(time)
        update = self._set_hamiltonian(hamiltonian) or update
        update = self._set_terms(terms) or update
        if (update):
            self._set_evolution(hpc=hpc)


    ############################################
    ### Auxiliary methods for _set_evolution ###
    ############################################

    def _set_time(self, time):
        if time is None or time < 0:
            raise ValueError(
                "Expected non-negative `time` value."
            )

        if time != self._time:
            self._time = time
            return True
        return False

    def _set_hamiltonian(self, hamiltonian):
        if id(self._hamiltonian) != id(hamiltonian):
            self._hamiltonian = hamiltonian
            return True
        return False

    def _set_terms(self, terms):
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
                if issparse(H):
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

    def get_evolution(self, copy=True):
        return np.copy(self._evolution) if copy else self._evolution

    def simulate(self, time=None, vector=None, hpc=True):
        r"""
        Analogous to :meth:`hiperwalk.QuantumWalk.simulate`,
        which accepts float entries for the ``time`` parameter.

        Parameters
        ----------
        time : float or tuple of floats
            This parameter is analogous to those in
            :meth:`hiperwalk.QuantumWalk.simulate`,
            with the distinction that it accepts float inputs.
            The ``step`` parameter is used to
            construct the evolution operator.
            The states within the interval
            **[** ``start/step``, ``end/step`` **]** are stored.
            The values describing this interval are
            rounded up if the decimal part exceeds ``1 - 1e-5``,
            and rounded down otherwise.

        Other Parameters
        ----------------
        See `hiperwalk.QuantumWalk.simulate`.

        See Also
        --------
        set_evolution
        get_evolution
        """
        if time is None:
            raise ValueError(
                "Invalid `time_range`. Expected a float, 2-tuple, "
                + "or 3-tuple of float."
            )

        time = np.array(Simulator.exponent_to_tuple(time))

        self.set_time(time=time[2], hpc=hpc)

        # converting time_range to int
        tol = 1e-5
        time = [int(val/time[2])
                if int(val/time[2])
                   <= np.ceil(val/time[2]) - tol
                else int(np.ceil(val/time[2]))
                for val in time]

        saved_vectors = super().simulate(time, vector, hpc)
        return saved_vectors

    def _number_to_valid_time(self, number):
        return number
