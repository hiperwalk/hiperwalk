import numpy as np
import scipy.sparse
import scipy.linalg
from .quantum_walk import QuantumWalk
from .._constants import PYNEBLINA_IMPORT_ERROR_MSG
try:
    from . import _pyneblina_interface as nbl
except:
    pass

class ContinuousWalk(QuantumWalk):
    r"""
    Manage instance of the continuous time quantum walk model
    on unweighted graphs.

    For implemantation details see Notes Section.

    Parameters
    ----------
    graph :
        Graph on which the quantum walk occurs.
        There are three two types acceptable.

        :class:`hiperwalk.graph.Graph` :
            The graph itself.

        :class:`class:scipy.sparse.csr_array`:
            The graph adjacency matrix.

    adjacency : :class:`scipy.sparse.csr_array`, optional
        .. deprecated:: 2.0
            It will be removed in version 2.1.
            Use ``graph`` instead.

        Adjacency matrix of the graph on which the quantum occurs
        is going to occur.

    **kwargs : optional
        Arguments for setting Hamiltonian.

    See Also
    --------
    set_hamiltonian

    Notes
    -----
    Let :math:`A` be the adjacency matrix of the graph :math:`G(V, E)`.
    :math:`A` is a :math:`|V| \times |V|`-dimensional matrix such that

    .. math::
        A_{i,j} = \begin{cases}
            1, \text{ if } (i,j) \in E(G),\\
            0, \text{ otherwise}
        \end{cases}

    The states of the computational basis are :math:`\ket{i}` for
    :math:`0 \leq i < |V|` where
    :math:`\ket i` is associated with the :math:`i`-th vertex.

    This class can also be used to simulate the evolution of any
    any Hamiltonian.
    Simply pass the disered Hamiltonian instead of the adjacency matrix.
    """

    _hamiltonian_kwargs = dict()

    def __init__(self, graph=None, adjacency=None, **kwargs):

        super().__init__(graph=graph, adjacency=adjacency)

        self.hilb_dim = self._graph.number_of_vertices()
        self._hamiltonian = None

        import inspect

        if not bool(ContinuousWalk._hamiltonian_kwargs):
            # assign static attribute
            ContinuousWalk._hamiltonian_kwargs = {
                'gamma': ContinuousWalk._get_valid_kwargs(self.set_gamma),
                'marked': ContinuousWalk._get_valid_kwargs(self.set_marked)
            }

        self.set_hamiltonian(**kwargs)

    def set_gamma(self, gamma=None):
        r"""
        Set gamma used for the Hamiltonian.

        Parameters
        ----------
        gamma : float, default = 1
            Gamma value.
        """
        if gamma is None or gamma.imag != 0:
            raise TypeError("Value of 'gamma' is not float.")

        self._gamma = gamma
        self._evolution = None

    def get_gamma(self):
        r"""
        Get the gamma used for the Hamiltonian.

        Returns
        -------
        float
        """
        return self._gamma

    def set_hamiltonian(self, **kwargs):
        r"""
        Creates the Hamiltonian.


        Parameters
        ----------
        **kwargs : 
            Additional arguments.
            Used for determining the gamma value and marked vertices.
            See :meth:`set_gamma` and :meth:`set_marked`.

        Returns
        -------
        :class:`scipy.sparse.csr_array`
            
        Notes
        -----
        The Hamiltonian is given by

        .. math::
            H = -\gamma A  - \sum_{m \in M} \ket m \bra m

        where :math:`A` is the adjacency matrix, and
        :math:`M` is the set of marked vertices.

        See Also
        --------
        set_gamma
        set_marked
        """

        # if laplacian:
        #     degrees = self.adj_matrix.sum(axis=1)
        #     H = scipy.sparse.diags(degrees, format="csr")
        #     del degrees
        #     H -= self.adj_matrix
        #     H *= -gamma

        gamma_kwargs = ContinuousWalk._filter_valid_kwargs(
                              kwargs,
                              ContinuousWalk._hamiltonian_kwargs['gamma'])
        marked_kwargs = ContinuousWalk._filter_valid_kwargs(
                              kwargs,
                              ContinuousWalk._hamiltonian_kwargs['marked'])

        self.set_gamma(**gamma_kwargs)
        self.set_marked(**marked_kwargs)
        H = -self._gamma * self._graph.adj_matrix

        # creating oracle
        if len(self._marked) > 0:
            data = np.ones(len(self._marked), dtype=np.int8)
            oracle = scipy.sparse.csr_array(
                    (data, (self._marked, self._marked)),
                    shape=(self.hilb_dim, self.hilb_dim))

            H -= self._oracle

        self._hamiltonian = H
        # since the hamiltonian was changed,
        # the previous evolution operator may not be coherent.
        self._evolution_operator = None
        return H

    def get_hamiltonian(self):
        r"""
        Returns Hamiltonian.

        Returns
        -------
        :class:`scipy.sparse.csr_array`
        """
        return self._hamiltonian

    def set_evolution(self, **kwargs):
        r"""
        Alias for :meth:`set_hamiltonian`.
        """
        self.set_hamiltonian(**kwargs)

    def get_evolution(self, time=None, hpc=True):
        r"""
        Return the evolution operator.

        Constructs the evolution operator based on the previously
        set Hamiltonian.

        Parameters
        ----------
        time : float
            Gerate the evolution operator of the given time.

        hpc : bool, default = True
            Whether or not to use neblina hpc functions to
            generate the evolution operator.

        Returns
        -------
        :class:`numpy.ndarray`.

        Raises
        ------
        ValueError
            If `time < 0`.

        See Also
        --------
        set_hamiltonian

        Notes
        -----
        The evolution operator is given by

        .. math::
            U = e^{-\text{i}tH}

        where :math:`H` is a Hamiltonian matrix, and
        :math:`t` is the time.

        The evolution operator is constructed by Taylor Series expansion.

        .. warning::
            For floating time (not integer),
            the result is approximated. It is recommended to
            choose a small time interval and performing
            multiple matrix multiplications to
            mitigate uounding errors.
        """
        if time is None or time < 0:
            raise ValueError(
                "Expected non-negative `time` value."
            )

        if self._hamiltonian is None:
            raise AssertionError

        if hpc and not self._pyneblina_imported():
            hpc = False

        if hpc:
            # determining the number of terms in power series
            max_val = np.max(np.abs(self._hamiltonian))
            if max_val*time <= 1:
                nbl_U = nbl.matrix_power_series(
                        -1j*time*self._hamiltonian, 30)

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

                new_nbl_U = nbl.matrix_power_series(
                        -1j*new_time*self._hamiltonian, 20)
                nbl_U = nbl.multiply_matrices(new_nbl_U, new_nbl_U)
                for i in range(num_mult - 1):
                    nbl_U = nbl.multiply_matrices(nbl_U, new_nbl_U)

            U = nbl.retrieve_matrix(nbl_U)

        else:
            U = scipy.linalg.expm(-1j*time*self._hamiltonian.todense())

        self._evolution = U
        return U

    def simulate(self, time=None, initial_condition=None, hpc=True):
        r"""
        Simulate the Continuous Time Quantum Walk Hamiltonian.

        Analogous to the :meth:`QuantumWalk.simulate`
        but uses the Hamiltonian to construct the evolution operator.
        The Hamiltonian may be the previously set or
        passed in the arguments.

        Parameters
        ----------
        time : float or tuple of floats
            Analogous to the parameters of :meth:`QuantumWalk.simulate`,
            but accepts float inputs.
            ``step`` is used to construct the evolution operator.
            The states in the interval
            ***[* ``start/step``, ``end/step`` **]** are saved.
            The values that describe this interval are
            rounded up if the decimal part is greater than ``1 - 1e-5``,
            and rounded down otherwise.

        hamiltonian : :class:`numpy.ndarray` or None
            Hamiltonian matrix to be used for constructing
            the evolution operator.
            If ``None``, uses the previously set Hamiltonian

        Other Parameters
        ----------------
        initial_condition :
            See :meth:`QuantumWalk.simulate`.
        hpc :
            See :meth:`QuantumWalk.simulate`.


        Raises
        ------
        ValueError
            If ``time_range=None`` or ``initial_condition=None``,
            or ``hamiltonian`` has invalid Hilbert space dimension.


        Notes
        -----
        It is recommended to call this method with ``hamiltonian=None``
        to guarantee that a valid Hamiltonian was used.
        If the Hamiltonian is passed by the user,
        there is no guarantee that the Hamiltonian is local.

        See Also
        --------
        :meth:`QuantumWalk.simulate`
        hamiltonian
        """
        if time is None:
            raise ValueError(
                "Invalid `time_range`. Expected a float, 2-tuple, "
                + "or 3-tuple of float."
            )

        if initial_condition is None:
            raise ValueError(
                "`initial_condition` not specified."
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

        states = super().simulate(time, initial_condition,  hpc)
        return states
