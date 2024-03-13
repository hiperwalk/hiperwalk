import numpy as np
from scipy.sparse import issparse, csr_array, diags

class WeigthedGraph(Graph):
    r"""
    Constructs an arbitrary weighted graph.

    Parameters
    ----------
    adj_matrix :
        The adjacency matrix of the graph
        (any integer Hermitian matrix).
        Two input types are accepted:

        * Any matrix -- for instance,
            * :class:`scipy.sparse.csr_array`,
            * :class:`numpy.ndarray`,
            * list of lists.
        * :class:`network.Graph`.
            * The adjacency matrix is extracted from the graph.

    copy : bool, default=False
        If ``True``, a hard copy of ``adj_matrix`` is stored.
        If ``False``, the pointer to ``adj_matrix`` is stored.

    Raises
    ------
    TypeError
        If ``adj_matrix`` is not a square matrix.

    Notes
    -----
    .. todo::
        Check if it is more efficient to store the adjacency matrix as
        sparse or dense.
    
    The graph :math:`G(V,E)` on which the quantum walk 
    takes place is specified by
    any real Hermitian matrix :math:`C`.
    """

    def __default_dtype(self):
        return np.float32

    def __manipulate_adj_matrix_data(self, adj_matrix):
        return

    def __init__(self, adj_matrix, copy=False):
        super().__init__(adj_matrix, copy)

    def adjacent(self, u, v):
        return self._adj_matrix[u, v] != 0

    def _entry(self, entry):
        return self._adj_matrix[u, v]

    def _find_entry(self, entry):
        # this should not be invoked
        raise AttributeError("WeightedGraph has no `_find_entry` method.")

    def laplacian_matrix(self):
        r"""
        Return the graph's Laplacian matrix.

        See Also
        --------
        adjacency_matrix

        Notes
        -----
        The Laplacian matrix is given by

        .. math::
            L = W - A,

        where :math:`A` is the graph's adjacency matrix
        and :math:`W` is a diagonal matrix whose entries are
        the sum of the weights of the edges incident to
        a given vertex.

        .. math::
            W_{i, j} = \begin{cases}
                \sum_{k = 0}^{|V| - 1}A_{ik}, & \text{if } i = j\\
                0, & \text{otherwise}.
            \end{cases}

        .. todo::
            See
            https://people.eecs.berkeley.edu/~satishr/cs270/sp11/rough-notes/Tree-metrics.pdf
            as reference
        """
        return super().laplacian_matrix()

    def is_simple(self):
        return False

    # TODO: add functions to manage weights
