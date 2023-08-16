import hiperwalk as hpw
import numpy as np

num_vert = 100
g = hpw.Complete(num_vert)
qw = hpw.Coined(g, coin='G', marked={'-G': [0]})
time = (int(2*np.sqrt(num_vert)), 1)
states = qw.simulate(time=time,
                     initial_state=qw.uniform_state())
marked_prob = qw.success_probability(states)
hpw.plot_success_probability(time, marked_prob)
