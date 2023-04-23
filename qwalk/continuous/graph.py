import numpy as np
from ..base_walk import BaseWalk

class Graph(BaseWalk):
    r"""
    Manage instance of the continuous time quantum walk model
    on unweighted graphs.

    For implemantation details see Notes Section.

    Parameters
    ----------
    adj_matrix : :class:`scipy.sparse.csr_array`
        Adjacency matrix of the graph on which the quantum occurs
        is going to occur.

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
    any Hamiltonian.
    Simply pass the disered Hamiltonian instead of the adjacency matrix.
    """

    def __init__(self, adj_matrix):
        super().__init__(adj_matrix)

        self.hilb_dim = self.adj_matrix.shape[0]

    def oracle(self, marked_vertices=0):
        r"""
        Creates the oracle matrix.

        The oracle is created and set
        to be used by other methods.

        Parameters
        ----------
        marked_vertices = None, int, list of int, default=0
            Vertices to be marked.
            If ``None``, no vertex is marked and
            the oracle is also set to ``None``
            (which is equivalent to the zero matrix).

        Notes
        -----
        The oracle matrix has format

        .. math::
            \sum_{m \in M} \ket{m}\bra{m}

        where :math:`M` is the set of marked vertices.
        """

        if marked_vertices is None:
            self._oracle = None
            return None

        if not hasattr(marked_vertices, '__iter__'):
            marked_vertices = [marked_vertices]
        self._oracle = np.array(marked_vertices, dtype=int)

        R = np.zeros((self.hilb_dim, self.hilb_dim))
        for m in marked_vertices:
            R[m, m] = 1

        return R

    def hamiltonian(self, gamma=0, laplacian=False, **kwargs):
        r"""
        Creates the Hamiltonian.

        Creates the Hamiltonian based on the previously set oracle.
        If no oracle was set, it is ignored.
        If any valid ``**kwargs`` is sent, an oracle is created and set.
        The new oracle is then used to construct the Hamiltonian.

        See Also
        ------
        oracle
        """
        return None

    def evolution_operator(self, time=0, hpc=True, **kwargs):
        r"""
        Creates the evolution operator.

        Creates the evolution operator based on the previously
        set Hamiltonian.
        If any valid ``**kwargs`` is passed,
        a new Hamiltonian is created and set.
        The evolution operator is then constructed based on the
        new Hamiltonian.

        Parameters
        ----------
        hpc : bool, default = True
            Whether or not to use neblina hpc functions to
            generate the evolution operator.

        **kwargs :
            Arguments to construct the Hamiltonian.
            See :meth:`hamiltonian` for the list of arguments.
            If no ``**kwrags`` is passed,
            the previously set Hamiltonian is used.
            

        See Also
        ------
        hamiltonian

        Notes
        -----
        The evolution operator is given by

        .. math::
            U = e^{\text{i}tH}

        where :math:`H` is a Hamiltonian matrix, and
        :math:`t` is the time.

        The evolution operator is constructed by Taylor Series expansion.
        """
        return None

    def probability_distribution(self, states):
        return None

    def state(self, entries):
        return None

    def simulate(self, initial_time=None, final_time=None,
                 delta_time=None, hpc=True):

        return super().simulate_walk(
            self._evolution_operator, initial_condition, num_steps,
            save_interval=save_interval, hpc=hpc
        )
