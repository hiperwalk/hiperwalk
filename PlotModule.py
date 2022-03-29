import networkx as nx #TODO: import only needed functions?
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import linspace

#TODO: probabilities expects numpy array or matrix
#TODO: use graphviz to draw as noted by networkx's documentation:
#Proper graph visualization is hard, and we highly recommend that
#people visualize their graphs with tools dedicated to that task. 
#https://networkx.org/documentation/stable/reference/drawing.html
def PlotProbabilityDistributionOnGraph(AdjMatrix, probabilities):
    
    G = nx.from_numpy_matrix(AdjMatrix)
    
    #TODO: set figure size according to graphdimension
    fig_width = 10
    fig_height = 8
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = plt.axes()
    
    #calculates vertices positions.
    #needed to do beforehand in order to fix position for multiple steps
    #TODO: check position calculation method
    pos = nx.kamada_kawai_layout(G)
    min_prob = probabilities.min()
    max_prob = probabilities.max()
    
    #drawing graph
    node_size = 300 * len(str(AdjMatrix.shape[0])) #standard size times number of largest label's characters
    nx.draw(G, node_size=node_size, node_color=probabilities, vmin=min_prob, vmax=max_prob,
            cmap='YlOrRd_r', linewidths=1, edgecolors='black',
            font_color='black', with_labels=True, pos=pos)

    #setting colorbar
    sm = plt.cm.ScalarMappable(cmap='YlOrRd_r', norm=plt.Normalize(vmin=min_prob, vmax=max_prob))
    cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0,
        0.02, ax.get_position().height])
    cbar = plt.colorbar(sm, ticks=linspace(min_prob, max_prob, num=5), cax=cax)


    cbar.ax.tick_params(labelsize=14, length=7)
    plt.show()
