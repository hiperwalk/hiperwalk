import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PatchCollection
from neblina import *

from collections.abc import Iterable
from time import time
from CoinedModel import *
from PlotModule import *
from ModifiedNetworkXFunctions import *

init_engine(0)

grid_dim = 25
G = nx.grid_graph(dim=(grid_dim, grid_dim), periodic=True)
adj_matrix = nx.adjacency_matrix(G)
#G.add_edges_from([(0,1),(1,2),(2,0)])
fig = plt.figure(figsize=(10,8))
pos = nx.spring_layout(G)
num_vert = len(G.nodes())
nc = np.random.random(num_vert)
nodes = nx.draw_networkx_nodes(G,pos,node_color=nc)
edges = nx.draw_networkx_edges(G,pos)
nx.draw_networkx_labels(G, pos)
edges = nx.draw_networkx_edges(G,pos)
prev = time()
ax = plt.gca()

#creating specific initial condition
mid_vert = int(np.ceil(grid_dim**2 / 2)) - 1
#beware that the coin is 4-sided,
#thus, the initial condition being in the middle vertex,
psi0 = np.zeros(4*grid_dim**2, dtype=float)
psi0[4*mid_vert] = 1 #pointing downward
psi0[4*mid_vert + 1] = -1 #poiting leftward
psi0[4*mid_vert + 2] = -1 #pointing rightward
psi0[4*mid_vert + 3] = 1 #pointing upward
psi0 = psi0 / 2
U = EvolutionOperator_CoinedModel(adj_matrix)
num_steps = 9
states = SimulateWalk(U, psi0, num_steps, save_interval=1, save_initial_state=True)
prob = ProbabilityDistribution(adj_matrix, states)

start = time()
time_sum = 0
time_count = 0
config = True
num_frames = 10

def time_info():
    global time_count
    global config
    global num_frames
    time_count +=1
    if config and time_count == num_frames + 1:
        time_count = 0
        config = False
    elif not config:
        global start
        global time_sum
        end = time()
        diff = end - start
        time_sum += diff
        print('diff: ' + str(diff) + '\tmean: ' + str(time_sum/time_count))
        start = end

def update(n):
    nc = np.random.random(num_vert)
    nodes.set_array(nc)
    nodes.set_sizes(np.random.random(num_vert) * 1500)
    cmap = plt.get_cmap('YlOrRd')
    
    edges.set_color(cmap(nc))
    edges.set_linewidth(np.random.random(num_vert)*50)
    #ax.text(pos[0][0], pos[0][1], 'HELLO', size=20, zorder=2000))

    #if directed graph, i.e. for collections (arrows=True)
    #[edges[i].set_color(cmap(nc[i])) for i in range(len(edges))]
    #pc_edges = PatchCollection(edges)


    #print(type(nodes))
    #print(isinstance(nodes, Iterable))
    #print(type(edges))
    #print(type(edges[0]))
    #print(type(pc_edges))
    #print(type(pc_edges[0]))
    #return nodes,
    #nodes: <class 'matplotlib.collections.PathCollection'>
    #edges: <class 'list'>
    #edges[0] :<class 'matplotlib.patches.FancyArrowPatch'>
    #pc_edges: <class 'matplotlib.collections.PatchCollection'>
    #pc_edges: PatchCollection is not subscriptable => not iterable?

    time_info()

    #return nodes, edges,
    return nodes, edges,
    #<class 'matplotlib.collections.PathCollection'>
    #<class 'matplotlib.collections.LineCollection'>

def update2(n):
    nc = np.random.random(num_vert)
    nodes = nx.draw_networkx_nodes(G, pos, node_color=nc)
    cmap = plt.get_cmap('YlOrRd')
    edges = nx.draw_networkx_edges(G, pos, edge_color=cmap(nc),
            edge_cmap=cmap, width=np.random.random(num_vert)*50)

    time_info()

    return nodes, edges,

def update3(probabilities, G, ax):
    #nodes, edges, labels = nx_draw(G, ax=ax)]
    #nc = np.random.random(num_vert)
    nc = probabilities
    #nodes = nx.draw_networkx_nodes(G, pos, node_color=nc)
    #cmap = plt.get_cmap('YlOrRd')
    #edges = nx.draw_networkx_edges(G, pos, edge_color=cmap(nc),
    #        edge_cmap=cmap, width=np.random.random(num_vert)*3)

    nodes, edges, _ = nx_draw(G, pos, node_color=nc)

    time_info()
    return nodes, edges,

def update4(probabilities):
    nc = probabilities
    nodes.set_array(nc)
    nodes.set_sizes(np.random.random(num_vert) * 1500)

    #cmap = plt.get_cmap('YlOrRd')
    #edges.set_color(cmap(nc))
    #edges.set_linewidth(np.random.random(num_vert)*50)

    time_info()

    return nodes, edges,

def init_func():
    nc = prob[0]
    nodes, edges, _ = nx_draw(G, pos, node_color=nc)
    return nodes, edges,
#print(prob)

#PlotProbabilityDistributionOnGraph(adj_matrix, prob, cmap='viridis', animate=True)

#anim = FuncAnimation(fig, update2, frames=num_frames, interval=200, blit=True)
#anim = FuncAnimation(fig, update3, frames=prob, interval=200, repeat_delay=200, blit=True,
#        fargs=(G, ax))
anim = FuncAnimation(fig, update4, frames=prob, interval=200, repeat_delay=200, blit=True,
        init_func=init_func)

plt.tight_layout()
anim.save('anim.gif')
plt.show()
