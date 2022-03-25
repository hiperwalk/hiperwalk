# Coined Quantum Walk Model


import numpy
import scipy
import networkx


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
    n = AdjMatrix.shape[0]

    #creates array with edges. For example, if vertices 0 and 1 are adjacent, 
    #then both [0, 1] and [1, 0] are inside the array, i.e. edges = [[0,1], ..., [1, 0], ...]
    #TODO: this uses |E| extra memory, optimize it
    edges = [[i,j] for i in range(n) for j in range(n) if AdjMatrix[i,j] == 1]

    n = len(edges)
    #dtype int8 to use less memory
    S = scipy.sparse.coo_matrix((n, n), dtype=numpy.int8)

    #TODO: notes about complexity
    #O(|E|^2) since index is O(|E|)
    #TODO: to optimize: implement binary search and use adjacency matrix indices?
    for i in range(n):
        e = edges[i]
        j = edges.index([e[1], e[0]]) #find the index of the same edge with the opposite direction
        S[i, j] = 1
    return S.tocsr()

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
