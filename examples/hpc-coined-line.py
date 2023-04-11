import sys
sys.path.append('..')
import qwalk.coined as coined_qw
import plot as hplot

state_entries = [[1, 0, 1], [1, -1, -2]] # arc_notation
num_steps = 10
line = coined_qw.Line(num_steps, state_entries, 'arc_notation')

U = line.evolution_operator(coin='hadamard') 
print(U)
states = line.simulate_walk(U, save_interval=1, hpc=True)

prob = line.probability_distribution(states)
hplot.plot_probability_distribution(prob, plot_type='line',
                                    filename_prefix="coined-line",
                                    animate=True, show=True)
