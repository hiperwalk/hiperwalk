import sys
import numpy as np
import scipy.sparse
sys.path.append('..')
import qwalk as qw
import plot as hplot

num_vert = 5
seg = qw.Segment(num_vert)
# print(seg.adj_matrix.todense())
# print(seg.flip_flop_shift_operator())

# opposite order due to how Coined is implemented
# encapsule this to the user
# these initial conditions only work if num_vert is odd
middle_right = num_vert - 1
middle_left = num_vert - 2
# init_cond[middle_left] = 1
# init_cond[middle_right] = 1/np.sqrt(2)
# init_cond[middle_left] = -1j/np.sqrt(2)
# init_cond = seg.state([middle_right, middle_left],
#                       amplitudes=np.array([1, -1j]),
#                       type="arc_order")
init_cond = seg.state(([2], [0]),
                      np.array([1]),
                      type="arc_notation")
#init_cond = seg.state(([2], [0]),
#                      np.array([1]))
print(init_cond)


# U = seg.evolution_operator(coin='hadamard')
# U = seg.flip_flop_shift_operator()
U = seg.shift_operator()
I = U.transpose() @ U
print("Is unitary? " + str(
    (I != scipy.sparse.identity(I.shape[0])).nnz == 0
))
num_steps = int(2*num_vert - 2)

states = seg.simulate_walk(U, init_cond, num_steps, save_interval=1)
prob = seg.probability_distribution(states)
hplot.plot_probability_distribution(prob, animate=True, plot_type='line',
                                    interval=250)
