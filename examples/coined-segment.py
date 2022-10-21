import sys
import numpy as np
sys.path.append('..')
import qwalk as qw
import plot as hplot

num_vert = 21
seg = qw.Segment(num_vert)

init_cond = np.zeros(2*(num_vert-1), dtype=complex)
# opposite order due to how Coined is implemented
# encapsule this to the user
middle_left = num_vert - 1
middle_right = num_vert - 2

# init_cond[middle_left] = 1
init_cond[middle_right] = 1/np.sqrt(2)
init_cond[middle_left] = -1j/np.sqrt(2)


U = seg.evolution_operator(coin='hadamard')
num_steps = int(num_vert/2)
seg.prepare_walk(U, init_cond, num_steps)

states = seg.simulate_walk(save_interval=1)
prob = seg.probability_distribution(states)
hplot.plot_probability_distribution(prob, animate=True, plot_type='line',
                                    interval=250)
