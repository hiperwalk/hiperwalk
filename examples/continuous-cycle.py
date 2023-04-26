import sys
sys.path.append('..')
import numpy as np
import networkx as nx
import qwalk.continuous as ctqw
import qwplot

num_vert = 201
nx_cycle = nx.cycle_graph(num_vert)

c = ctqw.Graph(nx.adjacency_matrix(nx_cycle))

psi0 = c.state([[1, int(num_vert/2)]])
H = c.hamiltonian(gamma=1/(2*np.sqrt(2)))
final_state = c.simulate(int(num_vert/2), psi0)

probs = c.probability_distribution(final_state)
qwplot.probability_distribution(probs, plot_type='line',
        marker=None)
