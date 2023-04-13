from ..base_walk import BaseWalk
from constants import DEBUG

class Continuous(BaseWalk):
    """
    Manage instance of the continuous time quantum walk model
    on unweighted graphs.

    Parameters
    ----------
    adj_matrix : :class:`scipy.sparse.csr_array`
        Adjacency matrix of the graph on which
        the quantum walk occurs.

        .. todo::
            * Accept other types such as numpy array

    Notes
    -----
    This class can be used to simulate the evolution of any
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

