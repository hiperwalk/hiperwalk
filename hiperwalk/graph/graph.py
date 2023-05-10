class Graph():
    r"""
    Graph on which a quantum walk occurs.

    Used for generic graphs.

    Parameters
    ----------
    adj_matrix : :class:`scipy.sparse.csr_array`
        Adjacency matrix of the graph on
        which the quantum walk occurs.

    Raises
    ------
    TypeError
        if ``adj_matrix`` is not an instance of
        :class:`scipy.sparse.csr_array`.

    Notes
    -----
    Makes a plethora of methods available.
    These methods may be used by a Quantum Walk model for
    generating a valid walk.
    The default arguments for a given model are given by the graph.
    The Quantum Walk model is ignorant with this regard.

    This class may be passed as argument to plotting methods.
    Then the default representation for the specific graph will be shown.
    """

    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix
        self.coloring = None

    def get_default_coin(self):
        r"""
        Returns the default coin for the given graph.

        The default coin for the coined quantum walk on general
        graphs is ``grover``.
        """
        return 'grover'

    def is_embeddable(self):
        r"""
        Returns whether the graph can be embedded on the plane or not.

        If a graph can be embedded on the plane,
        we can assign directions to edges and arcs.

        Notes
        -----
        The implementation is class-dependent.
        We do not check the graph structure to determine whether
        it is embeddable or not.
        """
        return False
