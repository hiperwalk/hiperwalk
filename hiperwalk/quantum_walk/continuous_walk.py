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

    adjacency : :class:`scipy.sparse.csr_array`, optional
        .. deprecated:: 2.0a1
            It will be removed in version 2.0.
            Use ``graph`` instead.

        Adjacency matrix of the graph on which
        the quantum walk takes place.

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
    :meth:`hiperwalk.ContinuousWalk.set_hamiltonian` method.

    The states of the computational basis are :math:`\ket{i}` for
    :math:`0 \leq i < |V|`, where
    :math:`\ket i` is associated with the :math:`i`-th vertex.

    This class can also facilitate the simulation of any Hamiltonian
    evolution. To do this, simply pass the desired Hamiltonian in place
    of the adjacency matrix.
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
        self._hamiltonian = None
        self._evolution = None

    def set_marked(self, marked=[]):
        super().set_marked(marked)
        self._hamiltonian = None

    def get_gamma(self):
        r"""
        Retrieves the gamma value used in
        the definition of the Hamiltonian.

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
            See :meth:`hiperwalk.ContinuousWalk.set_gamma` and
            :meth:`hiperwalk.ContinuousWalk.set_marked`.

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

    def get_hamiltonian(self):
        r"""
        Returns the Hamiltonian.

        Returns
        -------
        :class:`scipy.sparse.csr_array`
        """
        if self._hamiltonian is not None:
            return self._hamiltonian

        H = -self._gamma * self._graph.adj_matrix

        # creating oracle
        if len(self._marked) > 0:
            data = np.ones(len(self._marked), dtype=np.int8)
            oracle = scipy.sparse.csr_array(
                    (data, (self._marked, self._marked)),
                    shape=(self.hilb_dim, self.hilb_dim))

            H -= oracle

        self._hamiltonian = H
        # since the hamiltonian was changed,
        # the previous evolution operator may not be coherent.
        self._evolution_operator = None
        return H

    def set_evolution(self, **kwargs):
        r"""
        Alias for :meth:`hiperwalk.ContinuousWalk.set_hamiltonian`.
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
            Generate the evolution operator of the given time.

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
        """
        if time is None or time < 0:
            raise ValueError(
                "Expected non-negative `time` value."
            )

        H = self.get_hamiltonian()

        if hpc and not self._pyneblina_imported():
            hpc = False

        if hpc:
            # determining the number of terms in power series
            max_val = np.max(np.abs(H))
            if max_val*time <= 1:
                nbl_U = nbl.matrix_power_series(
                        -1j*time*H, 30)

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
                        -1j*new_time*H, 20)
                nbl_U = nbl.multiply_matrices(new_nbl_U, new_nbl_U)
                for i in range(num_mult - 1):
                    nbl_U = nbl.multiply_matrices(nbl_U, new_nbl_U)

            U = nbl.retrieve_matrix(nbl_U)

        else:
            U = scipy.linalg.expm(-1j*time*H.todense())

        self._evolution = U
        return U

    def simulate(self, time=None, initial_state=None,
                 initial_condition=None, hpc=True):
        r"""
        Analogous to :meth:`hiperwalk.QuantumWalk.simulate`
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
        See :meth:`hiperwalk.QuantumWalk.simulate`.

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

        states = super().simulate(time, initial_state,
                                  initial_condition, hpc)
        return states
