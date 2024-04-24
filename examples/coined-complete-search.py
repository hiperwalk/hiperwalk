import hiperwalk as hpw
import numpy as np

num_vert = 100
g = hpw.Complete(num_vert)
qw = hpw.Coined(g, coin='G', marked={'-G': [0]})
sim_range = int(2*np.sqrt(num_vert)) + 1
states = qw.simulate(range=sim_range,
                     state=qw.uniform_state())
marked_prob = qw.success_probability(states)
hpw.plot_success_probability(sim_range, marked_prob)
