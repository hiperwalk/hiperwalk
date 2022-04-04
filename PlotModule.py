import networkx as nx #TODO: import only needed functions?
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import linspace

#exepcts kwargs as a reference to the dictionary **kwargs
#min_prob and max_prob send separately to give the possibility
#of min_prob and max_prob of the whole walk (instead of a single step)
#Expects kwargs['vmin'] = min_prob and kwargs['vmax'] = max_prob
def ConfigureNodes(G, probabilities, min_node_size, max_node_size, kwargs):
    #setting colormap related attributes
    if 'cmap' in kwargs:
        if kwargs['cmap'] == 'default':
            kwargs['cmap'] = 'YlOrRd_r'
        kwargs['node_color'] = probabilities

    #setting node attributes
    if 'edgecolors' not in kwargs:
        kwargs['edgecolors'] = 'black'

    if 'linewidths' not in kwargs:
        kwargs['linewidths'] = 1

    if 'with_labels' not in kwargs:
        kwargs['with_labels'] = True

    if kwargs['with_labels'] and 'font_color' not in kwargs:
        kwargs['font_color'] = 'black'

    if kwargs['node_size'] is None:
        if min_node_size is None:
            min_node_size = 300
        if max_node_size is None:
            max_node_size = 3000
    if min_node_size is not None and max_node_size is not None:
        #calculating size of each node acording to probability 
        #as a function f(x) = ax + b where b = min_size and
        #max_size = a*(max_prob-min_prob) + min_size
        a = (max_node_size - min_node_size) / (kwargs['vmax'] - kwargs['vmin'])
        kwargs['node_size'] = list(map( lambda x: a*x + min_node_size, probabilities ))

    #calculates vertices positions.
    #needed to do beforehand in order to fix position for multiple steps
    #TODO: check position calculation method
    #TODO: give the user the option to choose the position calculation method
    kwargs['pos'] = nx.kamada_kawai_layout(G)

#TODO: probabilities expects numpy array or matrix
#TODO: use graphviz to draw as noted by networkx's documentation:
#Proper graph visualization is hard, and we highly recommend that
#people visualize their graphs with tools dedicated to that task. 
#https://networkx.org/documentation/stable/reference/drawing.html
#By default, node sizes are larger if the probability is larger.
#to fix node size, set "node_size" to a constant or an array, as described in networx documentation.
#parameters.
#min_node_size, max_node_size: node size representing minimum/maximum probability.
#   default: None. If optional argument node_size is not set,
#   min_node_size and max_node_size will assume default values (check configure nodes).
#   If optional argument node_size is set and either min_node_size or max_node_size is None,
#   all nodes will have the size as described by node_size.
#   If min_node_size, max_node_size and node_size are set, node_size is disregarded.
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
def PlotProbabilityDistributionOnGraph(AdjMatrix, probabilities, min_node_size=None,
        max_node_size=None, **kwargs):

    #vmin and vmax are default keywords used by networkx_draw.
    #if an invalid keyword is passed to nx.draw(), it does not execute
    kwargs['vmin'] = probabilities.min() #min_prob
    kwargs['vmax'] = probabilities.max() #max_prob

    G = nx.from_numpy_matrix(AdjMatrix)

    if len(probabilities.shape) == 1:
        probabilities = [probabilities]
    restoreNodeSize = None if 'node_size' not in kwargs else kwargs['node_size']

    for i in range(len(probabilities)):
        #TODO: set figure size according to graphdimension
        fig, ax = ConfigureFigure()

        #setting kwargs for plotting
        #kwargs dict is updated by reference
        kwargs['node_size'] = restoreNodeSize
        ConfigureNodes(G, probabilities[i], min_node_size, max_node_size, kwargs)
        nx.draw(G, **kwargs)

        ##setting colorbar
        if 'cmap' in kwargs:
            ConfigureColorbar(ax, kwargs)

        #showing img
        #TODO: add saving option
        plt.tight_layout()
        plt.show()

#TODO: set figure size according to graphdimension
def ConfigureFigure(fig_width=10, fig_height=8):
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = plt.gca()
    return fig, ax

def ConfigureColorbar(ax, kwargs):
    sm = plt.cm.ScalarMappable(cmap=kwargs['cmap'],
            norm=plt.Normalize(vmin=kwargs['vmin'], vmax=kwargs['vmax']))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2.5%', pad=0.01)
    cbar = plt.colorbar(sm, ticks=linspace(kwargs['vmin'], kwargs['vmax'], num=5), cax=cax)

    cbar.ax.tick_params(labelsize=14, length=7)

