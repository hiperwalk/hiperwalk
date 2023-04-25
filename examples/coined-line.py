import sys
sys.path.append('..')
import qwalk.coined as coined_qw
import qwplot

state_entries = [[1, 0, 1], [1, -1, -2]] # arc_notation
num_steps = 10
line = coined_qw.Line(num_steps, state_entries, 'arc_notation')

U = line.evolution_operator() 
states = line.simulate((num_steps, 1))

prob = line.probability_distribution(states)
qwplot.probability_distribution(prob, plot_type='line',
                                filename_prefix="coined-line",
                                animate=True, show=True)
