from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse
from .graph import Graph

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

def __generate_valid_basis(self, basis=None):
    if basis is None:
        basis = np.arange(1, self._euc_dim - 1)

    try:
        basis.shape
    except AttributeError:
        basis = np.array(basis)

    if basis.shape[0] == self._euc_dim:
        # add negative arrays to basis
        # this explicits the neighbor order
        basis = np.concatenate((basis, -basis))

    if basis.shape[0] != 2*self._euc_dim:
        raise ValueError("Invalid number of basis vectors."
                         + str(self._euc_dim)
                         + " or "
                         + str(2*self._euc_dim)
                         + "were expected.")

    if len(basis.shape == 1):
        # generate standard basis
        _basis = np.zeros((basis.shape[0], self._euc_dim), dtype=np.int8)

        for i in range(basis.shape):
            entry = basis[i]
            positive = entry > 0
            entry = entry - 1 if positive else entry + 1
            _basis[i, entry] = 1 if positive else - 1

    return _basis

def __create_adj_matrix(self):
    num_vert = self.number_of_vertices()

    indices = [[]]*num_vert
    indptr = np.zeros(num_vert + 1)

    for v in range(num_vert):
        coord = self.vertex_coordinates(v)

        neigh = np.array([coord + self.basis[i]
                          for i in len(self.basis)])

        if not self._periodic:
            adj = [self.__valid_vertex(n, raise_exception=False)
                   for n in neigh]
            neigh = neigh[adj]

        indices[v] = [self.vertex_number(n) for n in neigh]
        indptr[v + 1] = indptr[v] + len(neigh)

    indices = np.reshape(indices, indptr[-1])
    data = np.ones(indptr[-1], dtype=np.int8)

    adj_matrix = scipy.sparse.csr_array((data, indices, indptr),
                                        copy=False)
    return adj_matrix

def __valid_vertex(self, vertex, raise_exception=False):
    try:
        # coordinates
        if len(vertex) != self._euc_dim:
            if not raise_exception:
                return False
            raise ValueError("Vertex is not a "
                             + str(self._euc_dim)
                             + "-tuple.")

        if self._periodic:
            return True

        for i in range(self._euc_dim):
            if vertex[i] < 0 and vertex[i] >= dim[i]:
                if not raise_exception:
                    return False
                raise ValueError("Inexistent vertex"
                                 + str(vertex) + ". "
                                 + "Lattice is not periodic.")

    except TypeError:
        # number
        number = coordinates
        if number < 0 or number >= self.number_of_vertices():
            if not raise_exception:
                return False
            raise ValueError("Inexistent vertex " + str(number))

    return True


def __vertex_number(self, coordinates):
    self.__valid_vertex(coordinates, raise_exception=True)
    # number
    try:
        len(coordinates)
    except TypeError:
        return number

    # coordinates
    dim = self._dim
    mult = 1
    number = 0
    for i in range(self._euc_dim):
        if self._periodic:
            coordinates[i] = coordinates[i] % self._dim

        number += mult*coordinates[i]
        mult *= dim[i]

    return number

def __vertex_coordinates(self, vertex):
    self.__valid_vertex(vertex, raise_exception=True)

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

    coordinates = np.zeros(self._euc_dim, dtype=np.int)
    for i in range(self._euc_dim - 1, 0, -1):
        coordinates[i] = vertex // mult

        # TODO: check which one is more efficient
        vertex = vertex % mult
        # vertex -= coordinates[i]*mult

        mult = mult % dim[i - 1]

    coordinates[0] = vertex

    return coordinates

def SquareLattice(dim, basis=None, periodic=True):
    self._dim = list(dim)
    self._periodic = periodic
    self._euc_dim = len(dim) # euclidian space dimension

    # use provided (or generated) basis
    self._basis = self.__generate_valid_basis(basis)

    # TODO: create and bind is_periodic() somewhere
    if not hasattr(self, vertex_coordinates):
        #TODO: bind vertex_number and vertex_coordinates
        raise NotImplementedError

    #create adjacency matrix
    self._adj_matrix = self.__create_adj_matrix()


def next_arc(self, arc):
    r"""
    Next arc with respect to the default embedding.

    Parameters
    ----------
    arc
        The arc in any of the following notations.

        * arc notation: tuple of vertices
            In ``(tail, head)`` format where
            ``tail`` and ``head`` must be valid vertices.
        * arc number: int.
            The numerical arc label.

    Returns
    -------
    Next arc in the same notation as the ``arc`` argument.

    See Also
    --------
    arc
    arc_number
    """
    raise NotImplementedError

def previous_arc(self, arc):
    r"""
    Previous arc with respect to the default embedding.

    Parameters
    ----------
    arc
        The arc in any of the following notations.

        * arc notation: tuple of vertices
            In ``(tail, head)`` format where
            ``tail`` and ``head`` must be valid vertices.
        * arc number: int.
            The numerical arc label.

    Returns
    -------
    Previous arc in the same notation as the ``arc`` argument.

    See Also
    --------
    arc
    arc_number
    """
    raise NotImplementedError
