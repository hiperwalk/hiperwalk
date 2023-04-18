import networkx as nx
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
import qwalk.continuous as cont_qw
import plot as hplot

# create random graph and obtain its adjacency matrix
num_vert = 20
edge_prob = 0.25
G = nx.gnp_random_graph(num_vert, edge_prob)
adj = nx.adjacency_matrix(G)
del G

# Quantum Walk preparation and simulation
random = cont_qw.Continuous(adj)
# By default uses flip-flop shift operator and Grover coin
U = random.evolution_operator()
# Initial state at vertex 0 pointing to the first directiong possible
init_state = random.state([[1, 0, 0]])
num_steps = num_vert
states = random.simulate_walk(init_state)

## Calculating probabilities and plotting
#prob = random.probability_distribution(states)
#hplot.plot_probability_distribution(
#    prob, plot_type='graph', animate=True, adj_matrix=adj, interval=1000,
#    cmap='default', fixed_probabilities=False
#)
