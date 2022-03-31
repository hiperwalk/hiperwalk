import networkx as nx #TODO: import only needed functions?
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import linspace

#exepcts kwargs as a reference to the dictionary **kwargs
#min_prob and max_prob send separately to give the possibility
#of min_prob and max_prob of the whole walk (instead of a single step)
def ConfigureNodes(G, probabilities, min_prob, max_prob, prob_node_size, kwargs):
    #setting colormap related attributes
    if 'cmap' in kwargs:
        if kwargs['cmap'] == 'default':
            kwargs['cmap'] = 'YlOrRd_r'

        kwargs['node_color'] = probabilities
        kwargs['vmin'] = min_prob
        kwargs['vmax'] = max_prob

    #setting node attributes
    if 'edgecolors' not in kwargs:
        kwargs['edgecolors'] = 'black'

    if 'linewidths' not in kwargs:
        kwargs['linewidths'] = 1

    if 'with_labels' not in kwargs:
        kwargs['with_labels'] = True

    if kwargs['with_labels'] and 'font_color' not in kwargs:
        kwargs['font_color'] = 'black'

    if prob_node_size:
        if 'node_size' not in kwargs:
            kwargs['node_size'] = (300, 3000)

        #calculating size of each node acording to probability 
        #as a function f(x) = ax + b where b = min_size and
        #max_size = a*max_prob + min_size
        a = (kwargs['node_size'][1] - kwargs['node_size'][0]) / max_prob
        b = kwargs['node_size'][0]
        kwargs['node_size'] = list(map( lambda x: a*x + b, probabilities))

    elif 'node_size' not in kwargs:
        #standard size times number of largest label's characters
        kwargs['node_size'] = 300 * len(str(G.number_of_nodes())) 

#TODO: probabilities expects numpy array or matrix
#TODO: use graphviz to draw as noted by networkx's documentation:
#Proper graph visualization is hard, and we highly recommend that
#people visualize their graphs with tools dedicated to that task. 
#https://networkx.org/documentation/stable/reference/drawing.html
#parameters.
#prob_node_size: Boolean. Node size represents probability.
#   The larger the node, the larger the probability.
#   Check node_size kwarg
#prob_transparency: Boolean. Node transparency represents probability.
#   The more opaque the noode, the larger the probability.
#   check alpha kwarg.
#...
#For detailed info about **kwargs check networkx's documentation for
#draw_networkx, draw_networkx_nodes, drawnetworkx_edges, etc.
#Here, a few useful optional keywords are listed
#cmap: the colormap name to be used to represent probabilities (consult matplotlib colormap options);
#   if cmap='default', uses 'YlOrRd_r' colormap.
#   The optional kwargs vmin, vmax will be computed from probabilites
#node_size: either an integer for fixed node size or a tuple: (min_size, max_size).
#   if ommited and plot_node_size is true, uses default size.
#alpha: either a float in the [0, 1] interval for fixed node transparency or a float tuple:
#   (min_alpha, max_alpha). If ommited and plot_transparency is true, uses default values.
def PlotProbabilityDistributionOnGraph(AdjMatrix, probabilities, prob_node_size=True, **kwargs):

    min_prob = probabilities.min()
    max_prob = probabilities.max()

    G = nx.from_numpy_matrix(AdjMatrix)

    #setting kwargs for plotting
    #kwargs dict is updated by reference
    ConfigureNodes(G, probabilities, min_prob, max_prob, prob_node_size, kwargs)
    
    #TODO: set figure size according to graphdimension
    fig_width = 10
    fig_height = 8
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = plt.axes()
    
    #calculates vertices positions.
    #needed to do beforehand in order to fix position for multiple steps
    #TODO: check position calculation method
    #TODO: give the user the option to choose the position calculation method
    kwargs['pos'] = nx.kamada_kawai_layout(G)
    
    nx.draw(G, **kwargs)

    #setting colorbar
    if 'cmap' in kwargs:
        sm = plt.cm.ScalarMappable(cmap=kwargs['cmap'],
                norm=plt.Normalize(vmin=min_prob, vmax=max_prob))

        cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0,
            0.02, ax.get_position().height])

        cbar = plt.colorbar(sm, ticks=linspace(min_prob, max_prob, num=5), cax=cax)
        cbar.ax.tick_params(labelsize=14, length=7)

    plt.show()
