import sys
sys.path.append('..')
import qwalk.coined as coined_qw
import plot as hplot

num_vert = 20
cycle = coined_qw.Cycle(num_vert)

init_cond = cycle.state([(1, 0, 0)])

S = cycle.persistent_shift_operator()
num_steps = num_vert

states = cycle.simulate_walk(S, init_cond, num_steps,
                             save_interval=1)

prob = cycle.probability_distribution(states)
hplot.plot_probability_distribution(prob, plot_type='bar', animate=True)
