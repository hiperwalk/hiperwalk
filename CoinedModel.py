###################################################################
#################### Coined Quantum Walk Model ####################
###################################################################
import numpy
import scipy
from scipy.linalg import block_diag as scipy_block_diag
import networkx
from AuxiliaryFunctions import *
from PyneblinaInterface import *

#TODO: create module with global constants?
DEBUG = False
if DEBUG:
    from time import time as now
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
#Expects sparse matrix #TODO: throw exception or convert
def FlipFlopShiftOperator(AdjMatrix):
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
        print("FlipFlopShiftOperator Memory: " + str(hpy().heap().size))
        print("FlipFlopShiftOperator Time: " + str(now() - start_time))

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
    return scipy.sparse.csr_matrix(scipy_block_diag(*L))

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
        return FlipFlopShiftOperator(AdjMatrix) @ CoinOperator(AdjMatrix)
    return FlipFlopShiftOperator(AdjMatrix) @ CoinOp

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
    #this is more efficient than:
    #(numpy.conj(elem) * elem).real
    #elem.real**2 + elem.imag**2
    return elem.real*elem.real + elem.imag*elem.imag

#vectorized
ElementwiseProbability = numpy.vectorize(UnvectorizedElementwiseProbability)

#TODO: documentation
#TODO: test with nonregular graph
#TODO: test with nonuniform condition
def ProbabilityDistribution(AdjMatrix, states):
    if len(states.shape) == 1:
        states = [states]

    #TODO: check if dimensions match and throw exception if necessary
    edges_indices = AdjMatrix.indptr #TODO: check if just creates reference (no hard copy)

    #TODO: check it is more efficient on demand or using extra memory (aux_prob)
    #aux_prob = ElementwiseProbability(state) 
    #first splits state per vertex, then calculates probability of each vertex direction,
    #then sums the probabilities resulting in the vertix final probability
    prob = numpy.array([[
            ElementwiseProbability(states[i][edges_indices[j]:edges_indices[j+1]]).sum()
            for j in range(len(edges_indices)-1)
        ] for i in range(len(states)) ])

    #TODO: benchmark (time and memory usage)
    return prob


#Simulating walk. Needed: U, state, stop_steps
#num_steps: int. Number of iterations to be simulated, i.e. U^num_steps |initial_state>
#save_interval: int. Number of steps to execute before saving the state.
#   For example if num_steps = 10 and save_interval = 5, the states at iterations
#   5 and 10 will be saved and case save_interval = 3, the states at iterations
#   3, 6, 9, and 10 will be saved.
#   Default: None, i.e. saves only the final state
#save_initial_condition: boolean. If True, adds the initial condition into the saved states.
#returns array with saved states
def SimulateWalk(U, initial_state, num_steps, save_interval=None, save_initial_state=False):
    #preparing walk
    nbl_matrix = NeblinaSendSparseMatrix(U)
    nbl_vec = NeblinaSendVector(initial_state)

    #number of states to save
    num_states = int(numpy.ceil(num_steps/save_interval)) if save_interval is not None else 1
    if save_initial_state:
        num_states += 1
    save_final_state = save_interval is None or num_steps % save_interval != 0

    #TODO: change dtype accordingly
    saved_states = numpy.zeros((num_states, initial_state.shape[0]), dtype=complex)
    state_index = 0 #index of the state to be saved
    if save_initial_state:
        saved_states[0] = initial_state
        state_index += 1

    #simulating walk
    #TODO: request multiple multiplications at once to neblina-core
    #TODO: check if intermediate states are being freed from memory
    for i in range(1, num_steps + 1):
        #TODO: request to change parameter order
        nbl_vec = sparse_matvec_mul(nbl_vec, nbl_matrix)

        if save_interval is not None and i % save_interval == 0:
            saved_states[state_index] = NeblinaRetrieveVector(nbl_vec, initial_state.shape[0],
                    deleteVector=True)
            state_index += 1

    if save_final_state:
        saved_states[state_index] = NeblinaRetrieveVector(nbl_vec, initial_state.shape[0],
                deleteVector=True)

    return saved_states
