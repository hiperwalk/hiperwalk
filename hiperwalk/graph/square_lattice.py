from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse import csr_array
from scipy.sparse import eye as sparse_eye
from .graph import Graph
from types import MethodType

# class SquareLattice(Graph):
#     r"""
#     Class for Square Lattice graphs.
# 
#     Lattice graphs are embedabble in a Euclidian Space,
#     forming a regular tiling.
# 
#     Notes
#     -----
#     Given an embedding,
#     directions can be assigned to the arcs.
#     """

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
    dim = self._dim
    mult = 1
    number = 0
    for i in range(self._euc_dim):
        if self._periodic:
            coordinates[i] = coordinates[i] % self._dim[i]

        number += mult*coordinates[i]
        mult *= dim[i]

    return number

def vertex_coordinates(self, vertex):
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

    mult = 1
    for i in range(0, self._euc_dim - 1):
        mult *= dim[i]

    coordinates = np.zeros(self._euc_dim, dtype=np.int32)
    for i in range(self._euc_dim - 1, 0, -1):
        coordinates[i] = vertex // mult

        # TODO: check which one is more efficient
        vertex = vertex % mult
        # vertex -= coordinates[i]*mult

        mult = mult % dim[i - 1]

    coordinates[0] = vertex

    return coordinates

def dimensions(self):
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

def SquareLattice(dim, basis=None, periodic=True,
                  weights=None, multiedges=None):
    r"""
    TODO docs
    """
    if weights is not None or multiedges is not None:
        raise NotImplementedError()

    if not hasattr(dim, '__iter__'):
        dim = [dim]
    dim = np.array(dim)
    num_vert = np.prod(dim)

    # create toy graph
    g = Graph(sparse_eye(num_vert))

    # modify toy graph to SquareLattice
    g._dim = dim
    g._periodic = periodic
    g._euc_dim = len(g._dim) # euclidian space dimension

    # use provided (or generated) basis
    g._basis = __generate_valid_basis(g._euc_dim, basis)

    # bind methods before updating adjacency matrix
    g._valid_vertex = MethodType(_valid_vertex, g)
    g.vertex_number = MethodType(vertex_number, g)
    g.vertex_coordinates = MethodType(vertex_coordinates, g)
    g.dimensions = MethodType(dimensions, g)

    #create adjacency matrix
    g._adj_matrix = __create_adj_matrix(g)
    del g._adj_matrix.data
    g._adj_matrix.data = None

    return g
