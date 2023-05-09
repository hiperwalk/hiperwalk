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
        raise NotImplementedError("Graph class")
