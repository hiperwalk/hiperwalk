import numpy as np
import scipy.sparse
import scipy.linalg
from .quantum_walk import QuantumWalk
from ..simulator import HamiltonianSimulator

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

    def __init__(self, graph=None, gamma=None, **kwargs):

        super().__init__(graph=graph)

        # create attributes
        self.hilb_dim = self._graph.number_of_vertices()
        self._gamma = None
        self._hamil_type = None

        time = kwargs['time'] if 'time' in kwargs else 0

        # simulator matrix will be updated
        self._simulator = HamiltonianSimulator(time=time,
                                               hamiltonian=[[1]],
                                               hpc=False)
        if 'terms' in kwargs:
            self.set_terms(terms=kwargs.pop['terms'], hpc=False)
        self.set_hamiltonian(gamma=gamma, **kwargs)

    def _set_gamma(self, gamma=None):
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
        self.set_hamiltonian(gamma=gamma, type=self._hamil_type,
                marked=self._marked)
        if self._set_gamma(gamma=gamma):
            self._set_hamiltonian()
            self._set_evolution(**kwargs)

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
        self.set_hamiltonian(gamma=self._gamma,
                             type=self._hamil_type,
                             marked=marked)

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
        :math:`M` is the set of marked vertices.
        """
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

        self._simulator.set_hamiltonian(H, hpc=hpc)

    def get_hamiltonian(self, **kwargs):
        r"""
        See :meth:`HamiltonianSimulator.get_hamiltonian`.
        """
        return self._simulator.get_hamiltonian(**kwargs)

    def set_time(self, **kwargs):
        r"""
        See :meth:`HamiltonianSimulator.set_time`.
        """
        self._simulator.set_time(**kwargs)

    def get_time(self):
        return self._simulator.get_time()

    def set_terms(self, **kwargs):
        self._simulator.set_terms(**kwargs)

    def get_terms(self):
        return self._simulator

    def simulate(self, time=None, initial_state=None, hpc=True):
        return self._simulator.simulate(time=time,
                                        vector=initial_state,
                                        hpc=hpc)

    def set_evolution(self, **kwargs):
        self._simulator.set_evolution(**kwargs)

    def get_evolution(self, **kwargs):
        return self._simulator.get_evolution(**kwargs)
