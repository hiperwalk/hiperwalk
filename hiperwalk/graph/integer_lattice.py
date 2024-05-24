from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse import csr_array
from scipy.sparse import eye as sparse_eye
from .graph import Graph
from types import MethodType
from .multigraph import Multigraph
from .weighted_graph import WeightedGraph

def __generate_valid_basis(euc_dim, basis=None):
    if basis is None:
        basis = np.arange(1, euc_dim + 1)

    try:
        basis.shape
    except AttributeError:
        basis = np.array(basis)

    if basis.shape[0] == euc_dim:
        # add negative arrays to basis
        # this explicits the neighbor order
        basis = np.concatenate((basis, -basis))

    if basis.shape[0] != 2*euc_dim:
        raise ValueError("Invalid number of basis vectors. "
                         + str(euc_dim)
                         + " or "
                         + str(2*euc_dim)
                         + " were expected.")

    if len(basis.shape) == 1:
        # generate standard basis
        valid_basis = np.zeros((basis.shape[0], euc_dim), dtype=np.int8)

        for i in range(basis.shape[0]):
            entry = basis[i]
            positive = entry > 0
            entry = entry if positive else -entry
            entry = entry - 1
            valid_basis[i, entry] = 1 if positive else - 1

        basis = valid_basis

    return basis

def __create_adj_matrix(graph):
    num_vert = graph.number_of_vertices()

    indices = [[]]*num_vert
    indptr = np.zeros(num_vert + 1, dtype=np.int32)

    for v in range(num_vert):
        coord = graph.vertex_coordinates(v)

        neigh = np.array([coord + graph._basis[i]
                          for i in range(len(graph._basis))])

        if not graph._periodic:
            adj = [graph._valid_vertex(n, exception=False)
                   for n in neigh]
            neigh = neigh[adj]

        indices[v] = [graph.vertex_number(n) for n in neigh]
        indptr[v + 1] = indptr[v] + len(neigh)

    if graph._periodic:
        indices = np.reshape(indices, indptr[-1])
    else:
        indices = np.array([elem for l in indices for elem in l],
                           dtype=np.int32)
    data = np.ones(indptr[-1], dtype=np.int8)

    adj_matrix = csr_array((data, indices, indptr), copy=False)
    return adj_matrix

def _valid_vertex(self, vertex, exception=False):
    try:
        # coordinates
        if len(vertex) != self._euc_dim:
            if not exception:
                return False
            raise ValueError("Vertex is not a "
                             + str(self._euc_dim)
                             + "-tuple.")

        if self._periodic:
            return True

        for i in range(self._euc_dim):
            if vertex[i] < 0 or vertex[i] >= self._dim[i]:
                if not exception:
                    return False
                raise ValueError("Inexistent vertex"
                                 + str(vertex) + ". "
                                 + "Lattice is not periodic.")

    except TypeError:
        # number
        if vertex < 0 or vertex >= self.number_of_vertices():
            if not exception:
                return False
            raise ValueError("Inexistent vertex " + str(vertex))

    return True

def vertex_number(self, coordinates):
    self._valid_vertex(coordinates, exception=True)
    # number
    try:
        len(coordinates)
    except TypeError:
        return coordinates

    # coordinates
    coordinates = list(coordinates)
    dim = self._dim
    mult = 1
    number = 0
    for i in range(self._euc_dim - 1, -1, -1):
        if self._periodic:
            coordinates[i] = coordinates[i] % self._dim[i]

        number += mult*coordinates[i]
        mult *= dim[i]

    return number

def vertex_coordinates(self, vertex):
    r"""
    Return the coordinates of the given vertex.

    Given the number of a vertex,
    return the corresponding coordinates in the integer lattice.

    Returns
    -------
    :class:`numpy.ndarray`

    See Also
    --------
    hiperwalk.Graph.vertex_number

    Notes
    -----
    The vertex number depends on its coordinates and the
    dimension of the lattice ``dim``.
    If the coordinates of a vertex are ``(v[0], ..., v[n-1])``,
    its number is
    ``v[n-1] + dim[n-1]*v[n-2] + ... + dim[n-1]*...*dim[1]*v[0]``.

    Examples
    --------
    .. testsetup::

        import hiperwalk as hpw

    The methods ``vertex_coordinates`` is the inverse of
    ``vertex_number``, and vice versa.

    .. doctest::

        >>> g = hpw.IntegerLattice((3, 3, 3))
        >>> tuple(g.vertex_coordinates(0))
        (0, 0, 0)
        >>> g.vertex_number((0, 0, 0))
        0
        >>> tuple(g.vertex_coordinates(13))
        (1, 1, 1)
        >>> g.vertex_number((1, 1, 1))
        13
    """
    self._valid_vertex(vertex, exception=True)

    dim = self._dim.copy()

    # coordinates
    try:
        len(vertex)
        if self._periodic:
            for i in range(self._euc_dim):
                vertex[i] = vertex[i] % dim[i]

        return vertex

    except TypeError:
        pass

    # input is number
    mult = np.prod(dim, dtype=np.int64)
    coordinates = np.zeros(self._euc_dim, dtype=np.int32)
    for i in range(self._euc_dim - 1):
        mult = mult // dim[i]
        coordinates[i] = vertex // mult

        # TODO: check which one is more efficient
        vertex = vertex % mult
        # vertex -= coordinates[i]*mult

    coordinates[-1] = vertex

    return coordinates

def dimensions(self):
    r"""
    Dimensions of integer lattice.

    Returns
    -------
    dim : tuple of int
        ``dim[i]`` is the number of vertices in the ``i``-th axis.
    """
    return self._dim

def neighbors(self, vertex):
    v_num = self.vertex_number(vertex)
    start = self._adj_matrix.indptr[v_num]
    end = self._adj_matrix.indptr[v_num + 1]
    neighs = self._adj_matrix.indices[start:end]

    if hasattr(vertex, '__iter__'):
        neighs = np.array([self.vertex_coordinates(n) for n in neighs],
                           dtype=neighs.dtype)

    return neighs

def IntegerLattice(dim, basis=None, periodic=True,
                  multiedges=None, weights=None, copy=False):
    r"""
    Integer lattice graph.

    An integer lattice is a lattice in an euclidean space
    such that every point is a tuple of integers.
    In the integer lattice graph,
    every vertex corresponds to a lattice point.

    Parameters
    ----------
    dim : tuple of int
        Lattice dimensions where
        ``dim[i]`` is the number of vertices in the ``i``-th-axis.

    basis : list of int or matrix, default=None
        Vectors used to determine the graph adjacency and
        the corresponding order of neighbors.
        ``basis`` can be specified in three different ways.
        Let ``n = len(dim)``.

        * ``None``
            Equivalent to the argument ``[1, ..., n]``.

        * list of int
            Adjacency is described by the standard basis where
            ``i`` corresponds to the array with all entries
            equal to ``0`` and the ``i-1``-th entry equal to ``1``.
            Analogously, ``-i`` corresponds to the same array
            but in the opposite direction (``-1`` instead of ``1``).

            The values of ``basis`` must satisfy
            ``1 <= abs(basis) <= n``.
            Note that 0 is not a valid value because
            ``-0 == 0``.

            It is expected that ``len(basis) == 2*n`` or
            ``len(basis) == n``.
            If ``len(basis) == n``,
            the equivalent argument is

            .. code-block:: python

                [basis[0], ..., basis[n - 1],
                 -basis[0], ..., -basis[n - 1]]

        * matrix
            A matrix with ``2*n`` rows and ``n`` columns.
            The ``i``-th neighbor of the
            vertex with coordinates ``(v[0], ..., v[n-1])`` is
            ``(v[0] + basis[i][0], ..., v[n-1] + basis[i][n-1])``.


    periodic : bool, default=True
        ``True`` if the grid has cyclic boundary conditions,
        ``False`` if it has borders.

    multiedges, weights: scipy.sparse.csr_array, default=None
        See :ref:`graph_constructors`.

    copy : bool, default=False
        See :ref:`graph_constructors`.

    Returns
    -------
    :class:`hiperwalk.Graph`
        See :ref:`graph_constructors` for details.

    See Also
    --------
    :ref:`graph_constructors`.

    Notes
    -----
    The **order of neighbors** depends on the value of ``basis``.

    The vertex number depends on its coordinates and ``dim``.
    If the coordinates of a vertex are ``(v[0], ..., v[n-1])``,
    its number is
    ``v[n-1] + dim[n-1]*v[n-2] + ... + dim[n-1]*...*dim[1]*v[0]``.

    Examples
    --------

    .. testsetup::

        import hiperwalk as hpw

    .. doctest::

        >>> g = hpw.IntegerLattice((3, 3), basis=None)
        >>> neigh = g.neighbors((1, 1))
        >>> [tuple(g.vertex_coordinates(v)) for v in neigh]
        [(2, 1), (1, 2), (0, 1), (1, 0)]

    .. doctest::

        >>> g = hpw.IntegerLattice((3, 3), basis=[-1, -2])
        >>> neigh = g.neighbors((1, 1))
        >>> [tuple(g.vertex_coordinates(v)) for v in neigh]
        [(0, 1), (1, 0), (2, 1), (1, 2)]

    .. doctest::

        >>> basis = [[0, 1], [-1, 1], [1, -1], [0, -1]]
        >>> g = hpw.IntegerLattice((3, 3), basis=basis)
        >>> neigh = g.neighbors((1, 1))
        >>> [tuple(g.vertex_coordinates(v)) for v in neigh]
        [(1, 2), (0, 2), (2, 0), (1, 0)]

    """
    if weights is not None and multiedges is not None:
        raise ValueError(
            "Both `weights` and `multiedges` arguments were set. "
            + "Cannot decide whether to create a weighted graph or "
            + "a multigraph."
        )

    if not hasattr(dim, '__iter__'):
        dim = [dim]
    dim = np.array(dim)
    num_vert = np.prod(dim)

    # create toy graph
    g = Graph(sparse_eye(num_vert).tocsr())

    # modify toy graph to IntegerLattice
    g._dim = dim
    g._periodic = periodic
    g._euc_dim = len(g._dim) # euclidian space dimension
    g._num_loops = 0

    # use provided (or generated) basis
    g._basis = __generate_valid_basis(g._euc_dim, basis)

    # bind methods before updating adjacency matrix
    g._valid_vertex = MethodType(_valid_vertex, g)
    g.vertex_number = MethodType(vertex_number, g)
    g.vertex_coordinates = MethodType(vertex_coordinates, g)
    g.dimensions = MethodType(dimensions, g)

    #create adjacency matrix
    g._set_adj_matrix(__create_adj_matrix(g))

    # create multigraph or weighted graph
    data = weights if weights is not None else multiedges
    if data is not None:
        if hasattr(data, 'keys'):
            data = g._dict_to_adj_matrix(data)
            copy = False
        else:
            g._rearrange_matrix_indices(data)

        if weights is not None:
            g2 = WeightedGraph(data, copy=copy)

        else:
            g2 = Multigraph(data, copy=copy)

        g2._dim = g._dim
        g2._periodic = g._periodic
        g2._euc_dim = g._euc_dim
        g2._num_loops = g._num_loops
        g2._basis = g._basis

        g2._valid_vertex = MethodType(_valid_vertex, g2)
        g2.vertex_number = MethodType(vertex_number, g2)
        g2.vertex_coordinates = MethodType(vertex_coordinates, g2)
        g2.dimensions = MethodType(dimensions, g2)

        g = g2

    return g
