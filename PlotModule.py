import networkx as nx #TODO: import only needed functions?
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import linspace

#exepcts kwargs as a reference to the dictionary **kwargs
#min_prob and max_prob send separately to give the possibility
#of min_prob and max_prob of the whole walk (instead of a single step)
#Expects kwargs['vmin'] = min_prob and kwargs['vmax'] = max_prob
def ConfigureNodes(G, probabilities, prob_node_size, kwargs):
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

    if prob_node_size:
        if 'node_size' not in kwargs:
            kwargs['node_size'] = (300, 3000)

        #calculating size of each node acording to probability 
        #as a function f(x) = ax + b where b = min_size and
        #max_size = a*(max_prob-min_prob) + min_size
        a = (kwargs['node_size'][1] - kwargs['node_size'][0]) / (kwargs['vmax'] - kwargs['vmin'])
        b = kwargs['node_size'][0]
        kwargs['node_size'] = list(map( lambda x: a*x + b, probabilities))

    elif 'node_size' not in kwargs:
        #standard size times number of largest label's characters
        kwargs['node_size'] = 300 * len(str(G.number_of_nodes())) 

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

    #vmin and vmax are default keywords used by networkx_draw.
    #if an invalid keyword is passed to nx.draw(), it does not execute
    kwargs['vmin'] = probabilities.min() #min_prob
    kwargs['vmax'] = probabilities.max() #max_prob

    G = nx.from_numpy_matrix(AdjMatrix)

    #TODO: set figure size according to graphdimension
    fig, ax = ConfigureFigure()

    #setting kwargs for plotting
    #kwargs dict is updated by reference
    ConfigureNodes(G, probabilities, prob_node_size, kwargs)
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

