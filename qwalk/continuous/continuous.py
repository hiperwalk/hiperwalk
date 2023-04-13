from ..base_walk import BaseWalk
from constants import DEBUG

class Continuous(BaseWalk):
    r"""
    Manage instance of the continuous time quantum walk model
    on unweighted graphs.

    For implemantation details see Notes Section.

    Parameters
    ----------
    adj_matrix : :class:`scipy.sparse.csr_array`
        Adjacency matrix of the graph on which
        the quantum walk occurs.

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
    continuous time matrix in the format

    .. math::

        U = e^{\mathrm{i}H}

    as long as :math:`H` is hermitian.
    """

    def __init__(self, adj_matrix):
        return None

    def oracle(self, vertices):
        return None

    def evolution_operator(self, hpc=False, **kwargs):
        return None

    def search_evolution_operator(self, vertices, hpc=False, **kwargs):
        return None

    def probability_distribution(self, states):
        return None

