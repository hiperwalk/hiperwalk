import sys
sys.path.append('..')
import qwalk.coined as coined_qw
import qwplot

num_vert = 21
mid_vert = int(num_vert/2)
seg = coined_qw.Segment(num_vert)

# Initial condition starting in the middle vertex
# with the coin pointing to the right (real amplitude)
# and the left (complex amplitude).
psi0 = seg.state(
    [(1, mid_vert, 0), [-1j, mid_vert, 1]],
    type='vertex_dir'
)

hpcU = seg.evolution_operator()

num_steps = 15
states = seg.simulate((num_steps, 1), psi0, hpcU)

prob = seg.probability_distribution(states)
qwplot.probability_distribution(prob, animate=True, plot_type='line')
