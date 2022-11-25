import sys
import numpy as np
import scipy.sparse
sys.path.append('..')
import qwalk as qw
import plot as hplot

num_vert = 5
cycle = qw.Cycle(num_vert)
print(cycle.adj_matrix.todense())
# print(cycle.flip_flop_shift_operator())

"""
# opposite order due to how Coined is implemented
# encapsule this to the user
# these initial conditions only work if num_vert is odd
middle_right = num_vert - 1
middle_left = num_vert - 2
# init_cond = cycle.state([[1, 1]], type="arc_order")
# init_cond = cycle.state([(1, 5, 4)],
#                       type="arc_notation")
init_cond = cycle.state([(1, int(num_vert/2), 0),
                       [-1j, int(num_vert/2), 1]])
print(init_cond)

U = cycle.evolution_operator(coin='hadamard')
# U = cycle.flip_flop_shift_operator()
# U = cycle.shift_operator()
num_steps = int(num_vert/2)

states = cycle.simulate_walk(U, init_cond, num_steps, save_interval=1)
prob = cycle.probability_distribution(states)
hplot.plot_probability_distribution(prob, plot_type='line',
                                    filename_prefix="segment/anim", show=False)
"""
