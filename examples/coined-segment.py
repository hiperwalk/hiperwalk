import sys
sys.path.append('..')
import qwalk.coined as coined_qw
import plot as hplot

num_vert = 21
mid_vert = int(num_vert/2)
seg = coined_qw.Segment(num_vert)

# Initial condition starting in the middle vertex
# with the coin pointing to the right (real amplitude)
# and the left (complex amplitude).
init_cond = seg.state([(1, mid_vert, 0),
                       [-1j, mid_vert, 1]])

U = seg.evolution_operator(coin='hadamard')
num_steps = int(15)
states = seg.simulate_walk(U, init_cond, (0, num_steps))

prob = seg.probability_distribution(states)
hplot.plot_probability_distribution(prob, animate=True, plot_type='line')
