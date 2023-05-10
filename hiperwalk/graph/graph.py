from warnings import warn

def _binary_search(v, elem, start=0, end=None):
    r"""
    expects sorted array and executes binary search in the subarray
    v[start:end] searching for elem.
    Return the index of the element if found, otherwise returns -1
    Cormen's binary search implementation.
    Used to improve time complexity
    """
    if end == None:
        end = len(v)
    
    while start < end:
        mid = int((start + end)/2)
        if elem <= v[mid]:
            end = mid
        else:
            start = mid + 1

    return end if v[end] == elem else -1

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
        warn("Check if valid adjacency matrix")
        self.adj_matrix = adj_matrix
        self.coloring = None

    def default_coin(self):
        r"""
        Returns the default coin for the given graph.

        The default coin for the coined quantum walk on general
        graphs is ``grover``.
        """
        return 'grover'

    def embeddable(self):
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

    def arc_label(self, tail, head):
        return _binary_search(self.adj_matrix.indices, head,
                              start = self.adj_matrix.indptr[tail],
                              end = self.adj_matrix.indptr[tail + 1])

    def neighbors(self, vertex):
        start = self.adj_matrix.indptr[vertex]
        end = self.adj_matrix.indptr[vertex + 1]
        return self.adj_matrix.indices[start:end]

    def number_of_vertices(self):
        return self.adj_matrix.shape[0]

    def number_of_arcs(self):
        return self.adj_matrix.sum()

    def number_of_edges(self):
        return self.adj_matrix.sum() >> 1
