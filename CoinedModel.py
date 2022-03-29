# Coined Quantum Walk Model

import numpy
import scipy
import networkx
from AuxiliaryFunctions import *
from time import time as now

DEBUG = True
if DEBUG:
    from guppy import hpy #used to check memory usage


def UniformInitialCondition(AdjMatrix):
    G = networkx.from_numpy_matrix(AdjMatrix)
    N = sum([G.degree(i) for i in range(AdjMatrix.shape[0])])
    return numpy.matrix([[1]]*N)/numpy.sqrt(N)
    #TODO: USE np.ones

# Creates flip-flop shift operator based on adjacency matrix.
# For more information about the general flip-flop shift operator,
# check "Quantum Walks and Search Algorithms" Section 7.2: Coined Walks on Arbitrary Graphs.
# TODO: explain resulting matrix (edges labeled similarly to position-coin notation)
# Parameter: expects adjacency matrix of an unweighted undirected graph
def FlipFlopShiftOperator(AdjMatrix):
    start_time = now()
    n = AdjMatrix.shape[0]

    #creates array with edges. For example, if vertices 0 and 1 are adjacent, 
    #then both [0, 1] and [1, 0] are inside the array, i.e. edges = [[0,1], ..., [1, 0], ...]
    #TODO: this uses |E| extra memory, optimize it
    edges = [[i,j] for i in range(n) for j in range(n) if AdjMatrix[i,j] == 1]

    n = len(edges)
    #dtype int8 to use less memory
    S = scipy.sparse.lil_matrix((n, n), dtype=numpy.int8)

    #TODO: notes about complexity
    #O(|E|^2) since index is O(|E|)
    #TODO: to optimize: implement binary search and use adjacency matrix indices?
    for i in range(n):
        e = edges[i]
        j = edges.index([e[1], e[0]]) #find the index of the same edge with the opposite direction
        S[i, j] = 1

    if DEBUG:
        print("OldFlipFlopShiftOperator Memory: " + str(hpy().heap().size))
        print("OldFlipFlopShiftOperator Time: " + str(now() - start_time))

    return S.tocsr()

#Expects sparse matrix #TODO: throw exception
def NewFlipFlopShiftOperator(AdjMatrix):
    if DEBUG:
        start_time = now()

    num_edges = AdjMatrix.sum() #expects weights to be 1 if adjacent

    #storing indexes edges in data.
    #obs.: for some reason this does not throw exception,
    #   so technically it is a sparse matrix that stores zero
    AdjMatrix.data = numpy.arange(num_edges)

    #calculating FlipFlopShift columns (to be used as indices of a csr_matrix)
    row = 0
    S_cols = numpy.zeros(num_edges)
    for edge in range(num_edges):
        if edge >= AdjMatrix.indptr[row + 1]:
            row += 1
        col_index = BinarySearch(AdjMatrix.data, edge, start = AdjMatrix.indptr[row],
                end = AdjMatrix.indptr[row+1])
        S_cols[edge] = AdjMatrix[AdjMatrix.indices[col_index], row]

    # using csr_matrix((data, indices, indptr), shape)
    S = scipy.sparse.csr_matrix(
            (numpy.ones(num_edges, dtype=numpy.int8), S_cols, numpy.arange(num_edges+1)),
            shape=(num_edges, num_edges))

    #restores original data to AdjMatrix
    AdjMatrix.data = numpy.ones(num_edges)

    #TODO: compare with old approach for creating S

    if DEBUG:
        print("NewFlipFlopShiftOperator Memory: " + str(hpy().heap().size))
        print("NewFlipFlopShiftOperator Time: " + str(now() - start_time))

    return S

def CoinOperator(AdjMatrix, coin='grover'):
    n = AdjMatrix.shape[0]
    G = networkx.from_numpy_matrix(AdjMatrix)
    if coin == 'grover':
        L = [GroverOperator(G.degree(i)) for i in range(n)]
    elif coin == 'hadamard':
        L = [HadamardOperator() for i in range(n)]
    else:
        return None
    return scipy.sparse.csr_matrix(scipy.linalg.block_diag(*L))

def GroverOperator(N):
    return numpy.matrix(2/N*numpy.ones(N)-numpy.identity(N))

def HadamardOperator():
    return 1/numpy.sqrt(2) * numpy.matrix([[1, 1], [1, -1]])

def OracleR(N):
    R = numpy.identity(N)
    R[0,0] = -1
    return numpy.matrix(R)

def EvolutionOperator_CoinedModel(AdjMatrix, CoinOp=None):
    #TODO: should these matrix multiplication be performed by neblina?
    if CoinOp is None:
        return ShiftOperator(AdjMatrix) @ CoinOperator(AdjMatrix)
    return ShiftOperator(AdjMatrix) @ CoinOp

def EvolutionOperator_SearchCoinedModel(AdjMatrix):
    S = ShiftOperator(AdjMatrix)
    C = CoinOperator(AdjMatrix)
    N = S.shape[0]
    #TODO: should this matrix multiplication be performed by neblina?
    return S*C*OracleR(N)

#TODO: check numpy vectorize documentation
#TODO: move to auxiliary functions?
#TODO: test with complex state
def UnvectorizedElementwiseProbability(elem):
    return numpy.conj(elem) * elem

#vectorized
ElementwiseProbability = numpy.vectorize(UnvectorizedElementwiseProbability)

#TODO: documentation
def ProbabilityDistribution(AdjMatrix, state):
    #TODO: check if dimensions match and throw exception if necessary
    degrees = [AdjMatrix[i].sum() for i in range(AdjMatrix.shape[0])]
    prob = ElementwiseProbability(state)
    print(degrees)
    return prob
