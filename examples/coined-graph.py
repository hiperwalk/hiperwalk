import networkx as nx
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
import qwalk.coined as coined_qw
import qwplot

# create random graph and obtain its adjacency matrix
num_vert = 20
edge_prob = 0.25
G = nx.gnp_random_graph(num_vert, edge_prob)
adj = nx.adjacency_matrix(G)
del G

# Quantum Walk preparation and simulation
random = coined_qw.Graph(adj)
# By default uses flip-flop shift operator, Grover coin and no oracle
random.evolution_operator()
# Initial state at vertex 0 pointing to the first direction possible
psi0 = random.state([[1, 0, 0]], 'vertex_dir')
# applies the evolution operator multiple times to psi0
num_steps = num_vert
states = random.simulate((num_steps, 1), psi0)

# Calculating probabilities and plotting
prob = random.probability_distribution(states)
qwplot.probability_distribution(
    prob, plot_type='graph', animate=True, adj_matrix=adj, interval=1000,
    cmap='default', fixed_probabilities=False
)
