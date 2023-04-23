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

    def __init__(self, H):
        super().__init__(H)

        self.hilb_dim = self.adj_matrix.shape[0]

    def oracle(self, marked_vertices=[0]):
        return None

    def hamiltonian(self, laplacian=False, marked_vertices=None):
        return None

    def evolution_operator(self, hpc=True, **kwargs):
        r"""
        Creates the evolution operator.

        Creates the evolution operator based on the previously
        created Hamiltonian.
        Or creates the evolution operator based on the
        Hamiltonian constructed using the ``**kwargs`` values.

        Parameters
        ----------
        hpc : bool, default = True
            Whether or not to use neblina hpc functions to
            generate the evolution operator.

        **kwargs :
            Arguments to construct the Hamiltonian.
            See :meth:`hamiltonian` for the list of arguments.
            If ``None`` is passed,
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

    def simulate(self, hpc=True):
        return super().simulate_walk(
            self._evolution_operator, initial_condition, num_steps,
            save_interval=save_interval, hpc=hpc
        )
