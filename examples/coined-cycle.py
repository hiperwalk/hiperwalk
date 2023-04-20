import sys
sys.path.append('..')
import qwalk.coined as coined_qw
import plot as hplot

num_vert = 20
cycle = coined_qw.Cycle(num_vert)

cycle.initial_condition([(1, 0, 0)])

S = cycle.persistent_shift_operator()
cycle.set_evolution_operator(S)
cycle.step((0, num_vert))

states = cycle.simulate()

prob = cycle.probability_distribution(states)
hplot.plot_probability_distribution(prob, plot_type='bar', animate=True)
