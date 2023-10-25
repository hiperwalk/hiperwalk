import numpy as np
import scipy.sparse
import scipy.linalg
from .quantum_walk import QuantumWalk
from ..simulator import HamiltonianSimulator

class ContinuousTime(QuantumWalk, HamiltonianSimulator):
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

    **kwargs :
        Arguments to set the evolution operator.

    See Also
    --------
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

    The Hamiltonian, which depends on the adjacency matrix and the
    location of the marked vertices,
    is described in the
    :meth:`hiperwalk.ContinuousTime.set_hamiltonian` method.

    The states of the computational basis are :math:`\ket{i}` for
    :math:`0 \leq i < |V|`, where
    :math:`\ket i` is associated with the :math:`i`-th vertex.

    This class can also facilitate the simulation of any Hamiltonian
    evolution. To do this, simply pass the desired Hamiltonian in place
    of the adjacency matrix.
    """

    _valid_kwargs = dict()

    def __init__(self, graph=None, **kwargs):

        QuantumWalk.__init__(self, graph=graph)

        # create attributes
        self.hilb_dim = self._graph.number_of_vertices()
        self._gamma = None
        self._hamil_type = None

        HamiltonianSimulator.__init__(self, **kwargs)

    def _set_gamma(self, gamma=None):
        if gamma is None or gamma.imag != 0:
            raise TypeError("Value of 'gamma' is not float.")

        if self._gamma != gamma:
            self._gamma = gamma
            return True
        return False

    def set_gamma(self, gamma=None, hpc=True):
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
        self.set_hamiltonian(gamma=gamma,
                             type=self._hamil_type,
                             marked=self._marked,
                             hpc=hpc)

    def get_gamma(self):
        r"""
        Retrieves the gamma value used in
        the definition of the Hamiltonian.

        Returns
        -------
        float
        """
        return self._gamma

    def _set_marked(self, marked=[]):
        raise NotImplementedError()

    def set_marked(self, marked=[], hpc=True):
        self.set_hamiltonian(gamma=self._gamma,
                             type=self._hamil_type,
                             marked=marked,
                             hpc=hpc)

    def _set_hamiltonian(self, gamma=None, type="adjacency", marked=[]):
        update = False
        if self._gamma != gamma:
            self._gamma = gamma
            update = True
        if self._hamil_type != type:
            self._hamil_type = type
            update = True
        if id(self._marked) != id(marked):
            self._marked = marked
            update = True

        if update:
            if type == 'adjacency':
                H = -self._gamma * self._graph.adjacency_matrix()
            else:
                raise NotImplementedError()

            # creating oracle
            if len(self._marked) > 0:
                data = np.ones(len(self._marked), dtype=np.int8)
                oracle = scipy.sparse.csr_array(
                        (data, (self._marked, self._marked)),
                        shape=(self.hilb_dim, self.hilb_dim))

                H -= oracle

            self._hamiltonian = H

        return update

    def set_hamiltonian(self, gamma=None, type="adjacency", marked=[],
                        hpc=True):
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
        :math:`M` is the set of marked vertices [1]_ [2]_.

        References
        ----------
        .. [1] E. Farhi and S. Gutmann.
            "Quantum computation and decision trees".
            Physical Review A, 58(2):915–928, 1998.
            ArXiv:quant-ph/9706062.

        .. [2] A. M. Childs and J. Goldstone.
            "Spatial search by quantum walk",
            Phys. Rev. A 70, 022314, 2004.
        """
        self.set_evolution(time=self._time,
                           terms=self._terms,
                           gamma=gamma,
                           type=type,
                           marked=marked,
                           hpc=hpc)

    def _set_evolution(self, **kwargs):
        HamiltonianSimulator._set_evolution(self, **kwargs)

    def set_time(self, time=None, hpc=True):
        self.set_evolution(time=time,
                           terms=self._terms,
                           gamma=self._gamma,
                           type=self._hamil_type,
                           marked=self._marked,
                           hpc=hpc)

    def set_terms(self, terms=21, hpc=True):
        self.set_evolution(time=self._time,
                           terms=terms,
                           gamma=self._gamma,
                           type=self._hamil_type,
                           marked=self._marked,
                           hpc=hpc)
