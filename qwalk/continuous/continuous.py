from ..base_walk import BaseWalk
from constants import DEBUG

class Continuous(BaseWalk):
    r"""
    Manage instance of the continuous time quantum walk model
    on unweighted graphs.

    For implemantation details see Notes Section.

    Parameters
    ----------
    H : :class:`scipy.sparse.csr_array`
        Hamiltonian to be simulated.
        Any Hermitian matrix.
        For the Quantum Walk,
        it may be the graph Adjacency matrix.

        .. todo::
            * Accept other types such as numpy array

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
    matrix in the format

    .. math::

        U = e^{\text{i}H}

    as long as :math:`H` is hermitian.
    """

    def __init__(self, H):
        super().__init__(H)

        self.hilb_dim = self.adj_matrix.shape[0]

    def oracle(self, vertices):
        return None

    def evolution_operator(self, hpc=True, **kwargs):
        return None

    def search_evolution_operator(self, vertices, hpc=True, **kwargs):
        return None

    def probability_distribution(self, states):
        return None

    def simulate_walk(self, initial_condition, save_interval=0,
                      hpc=True):
        return super().simulate_walk(
            self._evolution_operator, initial_condition, num_steps,
            save_interval=save_interval, hpc=hpc
        )
