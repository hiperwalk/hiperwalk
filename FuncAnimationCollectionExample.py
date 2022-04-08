import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PatchCollection

from collections.abc import Iterable
from time import time

G = nx.Graph()
G.add_edges_from([(0,1),(1,2),(2,0)])
fig = plt.figure(figsize=(8,8))
pos = nx.spring_layout(G)
nc = np.random.random(3)
nodes = nx.draw_networkx_nodes(G,pos,node_color=nc)
edges = nx.draw_networkx_edges(G,pos,arrows=True)
#edges = nx.draw_networkx_edges(G,pos)
prev = time()

def update(n):
    nc = np.random.random(3)
    nodes.set_array(nc)
    nodes.set_sizes(np.random.random(3) * 1500)
    cmap = plt.get_cmap('YlOrRd_r')
    
    #edges.set_color(cmap(nc))

    #if directed graph, i.e. for collections (arrows=True)
    [edges[i].set_color(cmap(nc[i])) for i in range(len(edges))]
    pc_edges = PatchCollection(edges)


    print(type(nodes))
    print(isinstance(nodes, Iterable))
    print(type(edges))
    print(type(edges[0]))
    print(type(pc_edges))
    print(type(pc_edges[0]))
    return nodes,
    #nodes: <class 'matplotlib.collections.PathCollection'>
    #edges: <class 'list'>
    #edges[0] :<class 'matplotlib.patches.FancyArrowPatch'>
    #pc_edges: <class 'matplotlib.collections.PatchCollection'>
    #pc_edges: PatchCollection is not subscriptable => not iterable?


    #return nodes, edges,
    #<class 'matplotlib.collections.PathCollection'>
    #<class 'matplotlib.collections.LineCollection'>



anim = FuncAnimation(fig, update, frames=10, interval=200, blit=True)

plt.tight_layout()
anim.save('anim.gif')
plt.show()
