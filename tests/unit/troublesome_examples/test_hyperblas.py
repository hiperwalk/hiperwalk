from sys import path as sys_path
sys_path.append('../../')
sys_path.append('../../../')
import numpy as np
import hiperwalk as hpw
from hiperwalk.quantum_walk._pyneblina_interface import *
from scipy.sparse import issparse
import neblina as phb

# simulation using hiperwalk with no hpc
print('test_coined_complete_search')
num_vert = 100
g = hpw.Complete(num_vert)
qw = hpw.Coined(g, coin='G', marked={'-G': [0]})
r = int(2*np.sqrt(num_vert)) + 1
expected_states = qw.simulate(range=r, state=qw.uniform_state())
expected_prob_dist = qw.probability_distribution(states)
expected_succ_prob = qw.success_probability(states)

# implementing simulation using pyhiperblas

U = qw.get_evolution()
init_state = qw.uniform_state()

phb.init_engine('cpu', 0)

start, end, step = (0, int(2*np.sqrt(num_vert)) + 1, 1)
num_states = 1 + (end - 1 - start) // step

# def _prepare_engine(self, state, hpc):
# autocast. hiperblas-core only allows same-time multipl
mat_complex = np.issubdtype(U.dtype,
                            np.complexfloating)
vec_complex = np.issubdtype(init_state.dtype,
                            np.complexfloating)
if mat_complex != vec_complex:
    if not mat_complex:
        U = U.astype(complex)
    else:
        init_state = init_state.astype(complex)

### sending sparse matrix ###
is_complex = False
n = U.shape[0]

simul_mat = (phb.sparse_matrix_new(n, n, phb.COMPLEX) if is_complex
        else phb.sparse_matrix_new(n, n, phb.FLOAT))

# using the reverse logic to add elements to the matrix
for row in range(n):
    start = U.indptr[row]
    end = U.indptr[row + 1]

    # columns must be added in reverse order
    for index in range(end - 1, start - 1, -1):
        col = U.indices[index]

        if is_complex:
            phb.sparse_matrix_set(simul_mat, row, col,
                                      U[row, col].real,
                                      U[row, col].imag)
        else:
            phb.sparse_matrix_set(simul_mat, row, col,
                                      U[row, col].real, 0)

phb.sparse_matrix_pack(simul_mat)
phb.move_sparse_matrix_device(simul_mat)
#############################

########## sending vector ##########
is_complex = np.issubdtype(init_state.dtype, np.complexfloating)
simul_vec = phb.load_numpy_array(init_state)
phb.move_vector_device(simul_vec)
####################################

dtype = (np.complex128 if (np.iscomplexobj(U)
                           or np.iscomplexobj(init_state))
         else np.double)

saved_states = np.zeros(
    (num_states, state.shape[0]), dtype=dtype
)
state_index = 0 # index of the state to be saved

saved_states[0] = state.copy()
state_index += 1

is_sparse = issparse(U)
while state_index < num_states:
    # simulate step
    for i in range(step):
        ### multiply_matrix_vector ###
        simul_vec = phb.sparse_matvec_mul(simul_vec, simul_mat)
        ##############################

    # save_simul_vec
    continue_simulation = state_index + 1 < num_states
    ##### retrieve_vector #####
    phb.move_vector_host(simul_vec)
    ret = phb.retrieve_numpy_array(simul_vec)
    ###########################
    if continue_simulation:
        # does it need to be a copy?
        ##### send_vector #####
        simul_vec = phb.load_numpy_array(ret.copy())
        phb.move_vector_device(simul_vec)
        #######################

    saved_states[state_index] = ret
    state_index += 1

np.allclose(expected_states, saved_states)
np.allclose(expected_states, saved_states, atol=1e-6, rtol=1e-6)
np.allclose(expected_states, saved_states, atol=1e-8, rtol=1e-8)
np.allclose(expected_states, saved_states, atol=1e-10, rtol=1e-10)
np.allclose(expected_states, saved_states, atol=1e-12, rtol=1e-12)
np.allclose(expected_states, saved_states, atol=1e-14, rtol=1e-14)
np.allclose(expected_states, saved_states, atol=1e-16, rtol=1e-16)

print('produto interno estado-a-estado (quao mais perto de 1 melhor)')
print([expected_states[i] @ saved_states[i]
       for i in range(len(saved_states))])
