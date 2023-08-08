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

    **kwargs : optional
        Arguments to set the Hamiltonian.

    See Also
    --------
    set_hamiltonian

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

    _hamiltonian_kwargs = dict()

    def __init__(self, graph=None, **kwargs):

        self.__update_hamiltonian = False
        super().__init__(graph=graph)

        self.hilb_dim = self._graph.number_of_vertices()
        self._hamiltonian = None

        import inspect

        if not bool(ContinuousTime._hamiltonian_kwargs):
            # assign static attribute
            ContinuousTime._hamiltonian_kwargs = {
                'gamma': ContinuousTime._get_valid_kwargs(self.set_gamma),
                'marked': ContinuousTime._get_valid_kwargs(self.set_marked)
            }

        self.set_hamiltonian(**kwargs)

    def set_gamma(self, gamma=None):
        r"""
        Sets the gamma parameter.
        
        The gamma parameter is used in the definition of the Hamiltonian.

        Parameters
        ----------
        gamma : float, default = 1
            Gamma value.
        """
        if gamma is None or gamma.imag != 0:
            raise TypeError("Value of 'gamma' is not float.")

        self._gamma = gamma
        self._update_hamiltonian()
        self._evolution = None

    def set_marked(self, marked=[]):
        super().set_marked(marked)
        self._update_hamiltonian()

    def get_gamma(self):
        r"""
        Retrieves the gamma value used in
        the definition of the Hamiltonian.

        Returns
        -------
        float
        """
        return self._gamma

    def _update_hamiltonian(self):
        if self.__update_hamiltonian:
            self._hamiltonian = -self._gamma * self._graph.adj_matrix

            # creating oracle
            if len(self._marked) > 0:
                data = np.ones(len(self._marked), dtype=np.int8)
                oracle = scipy.sparse.csr_array(
                        (data, (self._marked, self._marked)),
                        shape=(self.hilb_dim, self.hilb_dim))

                self._hamiltonian -= oracle

    def set_hamiltonian(self, **kwargs):
        r"""
        Creates the Hamiltonian.


        Parameters
        ----------
        **kwargs :
            Additional arguments.
            Used for determining the gamma value and marked vertices.
            See :meth:`hiperwalk.ContinuousTime.set_gamma` and
            :meth:`hiperwalk.ContinuousTime.set_marked`.

        Notes
        -----
        The Hamiltonian is given by

        .. math::
            H = -\gamma A  - \sum_{m \in M} \ket m \bra m,

        where :math:`A` is the adjacency matrix, and
        :math:`M` is the set of marked vertices.

        See Also
        --------
        set_gamma
        set_marked
        """

        self.__update_hamiltonian = False
        gamma_kwargs = ContinuousTime._filter_valid_kwargs(
                              kwargs,
                              ContinuousTime._hamiltonian_kwargs['gamma'])
        marked_kwargs = ContinuousTime._filter_valid_kwargs(
                              kwargs,
                              ContinuousTime._hamiltonian_kwargs['marked'])

        self.set_gamma(**gamma_kwargs)
        self.set_marked(**marked_kwargs)

        self.__update_hamiltonian = True
        self._update_hamiltonian()

    def get_hamiltonian(self):
        r"""
        Returns the Hamiltonian.

        Returns
        -------
        :class:`scipy.sparse.csr_array`
        """

        return self._hamiltonian

    def set_evolution(self, **kwargs):
        r"""
        Alias for :meth:`hiperwalk.ContinuousTime.set_hamiltonian`.
        """
        self.set_hamiltonian(**kwargs)

    def get_evolution(self, time=None, hpc=True):
        r"""
        Returns the evolution operator.

        Constructs the evolution operator based on the previously
        set Hamiltonian.

        Parameters
        ----------
        time : float
            Generates the evolution operator corresponding to the specified time.

        hpc : bool, default = True
            Determines whether or not to use neblina HPC 
            functions to generate the evolution operator.

        Returns
        -------
        :class:`numpy.ndarray`.

        Raises
        ------
        ValueError
            If ``time < 0``.

        See Also
        --------
        set_hamiltonian

        Notes
        -----
        The evolution operator is given by

        .. math::
            U = e^{-\text{i}tH},

        where :math:`H` is the Hamiltonian, and
        :math:`t` is the time.

        The evolution operator is constructed using
        a Taylor series expansion.

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
        if time is None or time < 0:
            raise ValueError(
                "Expected non-negative `time` value."
            )

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
                nbl_U = nbl.matrix_power_series(
                        -1j*time*H, 30)
            else:
                U = numpy_matrix_power_series(-1j*time*H.todense(), 30)

        else:
            # if the order of magnitude is very large,
            # float point errors may occur
            if ((isinstance(time, int) or time.is_integer())
                and max_val <= 1
            ):
                new_time = 1
                num_mult = time - 1
            else:
                new_time = max_val*time
                order = np.ceil(np.math.log(new_time, 20))
                new_time /= 10**order
                num_mult = int(np.round(time/new_time)) - 1

            if hpc:
                new_nbl_U = nbl.matrix_power_series(
                        -1j*new_time*H, 20)
                nbl_U = nbl.multiply_matrices(new_nbl_U, new_nbl_U)
                for i in range(num_mult - 1):
                    nbl_U = nbl.multiply_matrices(nbl_U, new_nbl_U)
            else:
                U = numpy_matrix_power_series(
                        -1j*new_time*H.todense(), 20)
                U = np.linalg.matrix_power(U, num_mult + 1)

        if hpc:
            U = nbl.retrieve_matrix(nbl_U)

        self._evolution = U
        return U

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

        time = np.array(self._time_to_tuple(time))

        self.get_evolution(time=time[2], hpc=hpc)

        # cleaning time_range to int
        tol = 1e-5
        time = [int(val/time[2])
                if int(val/time[2])
                   <= np.ceil(val/time[2]) - tol
                else int(np.ceil(val/time[2]))
                for val in time]

        states = super().simulate(time, initial_state, hpc)
        return states
