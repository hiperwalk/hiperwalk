import numpy as np
import scipy.sparse
import scipy.linalg
from .quantum_walk import QuantumWalk
from .._constants import PYNEBLINA_IMPORT_ERROR_MSG
try:
    from . import _pyneblina_interface as nbl
except:
    pass

class ContinuousTime(QuantumWalk):
    r"""
    Manage an instance of a continuous-time quantum walk
    on any simple graph.

    For further implementation details, refer to the Notes Section.

    Parameters
    ----------
    graph :
        Graph on which the quantum walk takes place.
        There are two acceptable inputs:

        :class:`hiperwalk.graph.Graph` :
            The graph itself.

        :class:`class:scipy.sparse.csr_array`:
            The adjacency matrix of the graph.

    gamma : float
        Gamma value for setting Hamiltonian.

    **kwargs : optional
        Arguments to set the Hamiltonian and evolution operator.

    See Also
    --------
    set_hamiltonian
    set_evolution

    Notes
    -----
    The adjacency matrix of a graph :math:`G(V, E)` is
    the :math:`|V| \times |V|`-dimensional matrix :math:`A` such that
    
    .. math::
        A_{i,j} = \begin{cases}
            1, \text{ if } (i,j) \in E(G),\\
            0, \text{ otherwise.}
        \end{cases}

    The Hamiltonian, which depends on the adjacency matrix and the location of 
    the marked vertices, is described in the
    :meth:`hiperwalk.ContinuousTime.set_hamiltonian` method.

    The states of the computational basis are :math:`\ket{i}` for
    :math:`0 \leq i < |V|`, where
    :math:`\ket i` is associated with the :math:`i`-th vertex.

    This class can also facilitate the simulation of any Hamiltonian
    evolution. To do this, simply pass the desired Hamiltonian in place
    of the adjacency matrix.
    """

    _valid_kwargs = dict()

    def __init__(self, graph=None, gamma=None, **kwargs):

        super().__init__(graph=graph)

        # create attributes
        self.hilb_dim = self._graph.number_of_vertices()
        self._gamma = None
        self._hamiltonian = None
        self._evolution = None
        self._evolution_time = None

        # import inspect

        if not bool(ContinuousTime._valid_kwargs):
            # assign static attribute
            ContinuousTime._valid_kwargs = {
                'gamma': ContinuousTime._get_valid_kwargs(
                    self._update_gamma),
                'marked': ContinuousTime._get_valid_kwargs(
                    self._update_marked),
                'evolution': ContinuousTime._get_valid_kwargs(
                    self._update_evolution),
                'time': ContinuousTime._get_valid_kwargs(
                    self._update_time),
                }

        if not 'time' in kwargs:
            kwargs['time'] = 0

        self.set_evolution(gamma=gamma, **kwargs)

    def _update_gamma(self, gamma=None):
        if gamma is None or gamma.imag != 0:
            raise TypeError("Value of 'gamma' is not float.")

        if self._gamma != gamma:
            self._gamma = gamma
            return True
        return False

    def set_gamma(self, gamma=None, **kwargs):
        r"""
        Sets the gamma parameter.
        
        The gamma parameter is used in the definition of the Hamiltonian.
        By setting gamma,
        both the Hamiltonian and evolution operator are updated.

        Parameters
        ----------
        gamma : float
            Gamma value.

        ** kwargs :
            Additional arguments for updating the evolution operator.
            For example, whether to use neblina HPC or not.
            See :meth:`set_evolution` for more options.

        Raises
        ------
        TypeError
            If ``gamma`` is ``None`` or complex.
        ValueError
            If ``gamma < 0``.
        """
        if self._update_gamma(gamma=gamma):
            self._update_hamiltonian()
            self._update_evolution(**kwargs)

    def get_gamma(self):
        r"""
        Retrieves the gamma value used in
        the definition of the Hamiltonian.

        Returns
        -------
        float
        """
        return self._gamma

    def set_marked(self, marked=[], **kwargs):
        self._update_marked(marked=marked)
        self._update_hamiltonian()
        self._update_evolution(**kwargs)

    def _update_hamiltonian(self):
        r"""
        If this method is invoked,
        the hamiltonian is recalculated
        """
        self._hamiltonian = -self._gamma * self._graph.adjacency_matrix()

        # creating oracle
        if len(self._marked) > 0:
            data = np.ones(len(self._marked), dtype=np.int8)
            oracle = scipy.sparse.csr_array(
                    (data, (self._marked, self._marked)),
                    shape=(self.hilb_dim, self.hilb_dim))

            self._hamiltonian -= oracle

    def set_hamiltonian(self, gamma=None, **kwargs):
        r"""
        Creates the Hamiltonian.

        If no marked vertices is specified,
        the default value is used.
        After the Hamiltonian is created,
        the evolution operator is updated accordingly.

        Parameters
        ----------
        gamma : float
            Gamma value.

        **kwargs :
            Used for determining the marked vertices, and
            the procedure for updating the evolution operator.
            See :meth:`hiperwalk.ContinuousTime.set_marked`, and
            See :meth:`hiperwalk.ContinuousTime.set_evolution`.

        Raises
        ------
        TypeError
            If ``gamma`` is ``None`` or complex.
        ValueError
            If ``gamma < 0``.

        See Also
        --------
        set_gamma
        set_marked
        set_evolution

        Notes
        -----
        The Hamiltonian is given by

        .. math::
            H = -\gamma A  - \sum_{m \in M} \ket m \bra m,

        where :math:`A` is the adjacency matrix, and
        :math:`M` is the set of marked vertices.
        """
        self._update_gamma(gamma=gamma)

        marked_kwargs = ContinuousTime._pop_valid_kwargs(kwargs,
                ContinuousTime._valid_kwargs['marked'])
        self._update_marked(**marked_kwargs)

        self._update_hamiltonian()
        self._update_evolution(**kwargs)

    def get_hamiltonian(self):
        r"""
        Returns the Hamiltonian.

        Returns
        -------
        :class:`scipy.sparse.csr_array`
        """

        return self._hamiltonian

    def _update_time(self, time=None):
        if time is None or time < 0:
            raise ValueError(
                "Expected non-negative `time` value."
            )

        if time != self._evolution_time:
            self._evolution_time = time
            return True
        return False

    def set_time(self, time=None, **kwargs):
        r"""
        Sets time.

        Sets time and
        generates the evolution operator corresponding to
        the specified time.

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
        if self._update_time(time=time):
            self._update_evolution(**kwargs)

    def _update_evolution(self, hpc=True, terms=21):
        r"""
        If this method is invoked,
        the evolution is recalculated
        """
        time = self._evolution_time

        if time == 0:
            self._evolution = np.eye(self.hilb_dim)
            return

        n = terms - 1
        H = self.get_hamiltonian()

        if hpc and not self._pyneblina_imported():
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
                U = numpy_matrix_power_series(
                        -1j*new_time*H.todense(), n)
                U = np.linalg.matrix_power(U, num_mult + 1)

        if hpc:
            U = nbl.retrieve_matrix(nbl_U)

        self._evolution = U

    def set_evolution(self, hpc=True, terms=21, **kwargs):
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
        self._update_time(**ContinuousTime._pop_valid_kwargs(
            kwargs,
            ContinuousTime._valid_kwargs['time']))
        self.set_hamiltonian(**kwargs)

    def simulate(self, time=None, initial_state=None, hpc=True):
        r"""
        Analogous to :meth:`hiperwalk.QuantumWalk.simulate`,
        which accepts float entries for the ``time`` parameter.

        Parameters
        ----------
        time : float or tuple of floats
            This parameter is analogous to those in
            :meth:`hiperwalk.QuantumWalk.simulate`,
            with the distinction that it accepts float inputs.
            The ``step`` parameter is used to construct the evolution operator.
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

        time = np.array(QuantumWalk._time_to_tuple(time))

        self.set_time(time=time[2])

        # cleaning time_range to int
        tol = 1e-5
        time = [int(val/time[2])
                if int(val/time[2])
                   <= np.ceil(val/time[2]) - tol
                else int(np.ceil(val/time[2]))
                for val in time]

        states = super().simulate(time, initial_state, hpc)
        return states
