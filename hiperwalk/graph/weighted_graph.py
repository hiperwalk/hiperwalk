import numpy as np
from scipy.sparse import issparse, csr_array, diags
from . import Graph

class WeightedGraph(Graph):
    r"""
    Construct an arbitrary weighted graph.

    This class enables the creation of a weighted graph, 
    defined by a Hermitian adjacency matrix that 
    represents the connections between nodes and 
    edge weights.

    Parameters
    ----------
    adj_matrix : various types accepted
        The adjacency matrix of the graph, 
        which must be a Hermitian matrix. 
        Two types of input are accepted:

            * Matrix formats such as:
                * :class:`scipy.sparse.csr_array`,
                * :class:`numpy.ndarray`,
                * List of lists.
            * :class:`networkx.Graph`:
                The adjacency matrix is derived from
                the provided NetworkX graph.

    copy : bool, default=False
        Determines how the adjacency matrix is stored:

            * If ``True``, a deep copy of ``adj_matrix`` is
              created and stored.
            * If ``False``, a reference to the original
              ``adj_matrix`` is stored.

    Raises
    ------
    TypeError
        If ``adj_matrix`` is not a square matrix, indicating 
        it cannot represent a valid adjacency matrix.

    Notes
    -----
    When defining an instance of the Coined class on a weighted graph, 
    the edge weights do not affect the simulation.
    
    When defining an instance of the ContinuousTime class on a weighted graph, 
    the edge weights do affect the simulation.
    """
    # def _default_dtype(self):
    #     return np.float32

    def _set_adj_matrix(self, adj_matrix):
        self._adj_matrix = adj_matrix

    def __init__(self, adj_matrix, copy=False):
        # TODO: Check if it is more efficient to store the
        # adjacency matrix as sparse or dense.
        super().__init__(adj_matrix, copy)

    def adjacent(self, u, v):
        return self._adj_matrix[u, v] != 0

    def _entry(self, row, col):
        return self._adj_matrix[row, col]

    def _find_entry(self, entry):
        # this should not be invoked
        raise AttributeError("WeightedGraph has no `_find_entry` method.")

    def adjacency_matrix(self, copy=True):
        r"""
        Return the adjacency matrix representation of the graph.

        Parameters
        ----------
        copy : bool, default=True
            If ``True``, return a hard copy of the adjacency matrix.
            If ``False``, return a pointer to the adjacency matrix.

        Returns
        -------
        :class:`scipy.sparse.csr_array`.

        Notes
        -----
        In weighted graphs,
        the entries of the adjacency matrix :math:`A` represent
        the weights of the edges. In general, the weight is a non-zero
        real number. In Hiperwalk, :math:`A` can be an arbitrary
        Hermitian matrix.
        """
        if copy:
            return self._adj_matrix.copy()
        return self._adj_matrix

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
        """
        # See
        # https://people.eecs.berkeley.edu/~satishr/cs270/sp11/rough-notes/Tree-metrics.pdf
        # as reference
        return super().laplacian_matrix()

    def is_simple(self):
        return False
