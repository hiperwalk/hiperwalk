import sys
import numpy as np
import scipy.sparse
sys.path.append('..')
import qwalk as qw
import plot as hplot

num_vert = 200
cycle = qw.Cycle(num_vert)

init_cond = cycle.state([(1, 0, 0)])

# U = cycle.evolution_operator(coin='hadamard')
S = cycle.shift_operator()
H = cycle.coin_operator(coin='hadamard')
U = S@H
# U = cycle.flip_flop_shift_operator()
# U = cycle.shift_operator()
num_steps = 100

states = cycle.simulate_walk(U, init_cond, num_steps,
                             save_interval=int(num_steps/2))
prob = cycle.probability_distribution(states)
hplot.plot_probability_distribution(prob, plot_type='line')
