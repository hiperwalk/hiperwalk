# Coined Quantum Walk Model


import numpy
import scipy
import networkx

def UniformInitialCondition(AdjMatrix):
    G = networkx.from_numpy_matrix(AdjMatrix)
    N = sum([G.degree(i) for i in range(AdjMatrix.shape[0])])
    return numpy.matrix([[1]]*N)/numpy.sqrt(N)
    #TODO: USE np.ones

def ShiftOperator(AdjMatrix):
    n = AdjMatrix.shape[0]

    start = time.time()
    CB = [[j,i] for j in range(n) for i in range(n) if AdjMatrix[j,i]==1]

    n = len(CB)
    S = scipy.sparse.csr_matrix((n, n))

    for i in range(n):
        S[i,CB.index([CB[i][1],CB[i][0]])] = 1
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
    #return ShiftOperator(AdjMatrix)*CoinOperator(AdjMatrix)
    if CoinOp is None:
        return ShiftOperator(AdjMatrix) @ CoinOperator(AdjMatrix)
    return ShiftOperator(AdjMatrix) @ CoinOp

def EvolutionOperator_SearchCoinedModel(AdjMatrix):
    S = ShiftOperator(AdjMatrix)
    C = CoinOperator(AdjMatrix)
    N = S.shape[0]
    return S*C*OracleR(N)

