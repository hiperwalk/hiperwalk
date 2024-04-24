import hiperwalk as hpw
import matplotlib.pyplot as plt
import numpy as np
import scipy

dim = 25
g = hpw.Grid((dim, dim))
N = g.number_of_vertices()
qw = hpw.Coined(graph=g, coin='G', shift='ff', marked={'-I': [0]})
psi0 = qw.uniform_state()
num_steps = int(1.5*np.sqrt(N*np.log(N))) + 1
states = qw.simulate(range=num_steps,
                     state=psi0)
succ_prob = qw.success_probability(states)
hpw.plot_success_probability(num_steps, succ_prob)
