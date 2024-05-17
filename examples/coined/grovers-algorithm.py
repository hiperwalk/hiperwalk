import hiperwalk as hpw
import numpy as np
import networkx as nx

N = 128
K_N = nx.complete_graph(N)
A = nx.adjacency_matrix(K_N)+np.eye(N)
graph = hpw.Graph(A)
qw = hpw.Coined(graph, shift='flipflop', coin='G', marked={'-G': [0]})
t_final = round(4*np.pi*np.sqrt(N)/4) + 1
states = qw.simulate(range=t_final,
                     state=qw.uniform_state())
marked_prob = qw.success_probability(states)
hpw.plot_success_probability(t_final, marked_prob)
