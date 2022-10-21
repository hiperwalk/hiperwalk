import numpy as np
import networkx as nx
import scipy.sparse
import sys
sys.path.append('..')
import qwalk as hpw

hpc = True

if hpc:
    # initialises neblina-core
    # TODO: transfer this to inferface (check if it is already initialised)
    from neblina import init_engine, stop_engine
    init_engine(0)

# generating adjacency matrix of a 5x5 2d-horizontal-latiice
grid_dim = 5
G = nx.grid_graph(dim=(grid_dim, grid_dim), periodic=True)
adj_matrix = scipy.sparse.csr_array(nx.adjacency_matrix(G))
del G # only the adjacency matrix is going to be used

# creating specific initial condition
mid_vert = int(np.ceil(grid_dim**2 / 2)) - 1
# beware that the coin is 4-sided,
# thus, the initial condition being in the middle vertex,
psi0 = np.zeros(4*grid_dim**2, dtype=float)
psi0[4*mid_vert] = 1        # pointing downward
psi0[4*mid_vert + 1] = -1   # poiting leftward
psi0[4*mid_vert + 2] = -1   # pointing rightward
psi0[4*mid_vert + 3] = 1    # pointing upward
psi0 = psi0 / 2

chl = hpw.Coined(adj_matrix) #coined horizontal lattice
#psi0 = chl.uniform_initial_condition()
print(len(psi0))
# simulating walk
U = chl.evolution_operator()

num_steps = 9

states = chl.simulate_walk(U, psi0, num_steps, save_interval=1)

prob = chl.probability_distribution(states)
print(prob)

chl.plot_probability(
    prob, adj_matrix=adj_matrix, plot_type='graph', cmap='viridis',
    animate=True, fixed_probabilities=False,
    filename_prefix='animation', interval=1000
)

# checking with python result
epsilon = 1e-15
print([
    np.linalg.norm(U**i @ psi0 - states[i]) <= epsilon
    for i in range(len(states))
])

# stops neblina-core
if hpc:
    stop_engine()
