import sys
import numpy as np
import scipy.sparse
sys.path.append('..')
import qwalk as qw
import plot as hplot

# state_entries = [[1, 0, 0], [1, -1, 1]] # vertex_dir
# line = qw.Line(10, state_entries)

state_entries = [[1, 0, 1], [1, -1, -2]] # arc_notation
line = qw.Line(10, state_entries, 'arc_notation')
print(state_entries)

# U = seg.evolution_operator(coin='hadamard')
# num_steps = int(num_vert/2)
# 
# states = seg.simulate_walk(U, init_cond, num_steps, save_interval=1)
# prob = seg.probability_distribution(states)
# hplot.plot_probability_distribution(prob, plot_type='line',
#                                     filename_prefix="segment/anim", show=False)
