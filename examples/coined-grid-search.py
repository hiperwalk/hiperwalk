import hiperwalk as hpw
import matplotlib.pyplot as plt
import numpy as np
import scipy

dim = 25
g = hpw.Grid((dim, dim))
N = g.number_of_vertices()
qw = hpw.Coined(graph=g, coin='G', shift='ff', marked={'-I': [0]})
psi0 = qw.uniform_state()
num_steps = int(1.5*np.sqrt(N*np.log(N)))
states = qw.simulate(time=(num_steps, 1),
                     initial_state=psi0,
                     hpc=False)
succ_prob = qw.success_probability(states)
plt.plot(list(range(num_steps + 1)), succ_prob,
         marker='o')
plt.show()
