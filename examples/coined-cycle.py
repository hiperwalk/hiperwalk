import sys
sys.path.append('..')
import qwalk.coined as coined_qw
import qwplot

num_vert = 20
cycle = coined_qw.Cycle(num_vert)

psi0 = cycle.state([(1, 0, 0)], 'vertex_dir')

S = cycle.persistent_shift_operator()

states = cycle.simulate((num_vert, 1), psi0, S)

prob = cycle.probability_distribution(states)
qwplot.probability_distribution(prob, plot_type='bar', animate=True)
