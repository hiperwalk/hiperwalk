import networkx as nx #TODO: import only needed functions?
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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

    if 'node_size' not in kwargs:
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
#   If min_node_size or max_node_size are not set and optional argument node_size is not set,
#   min_node_size and max_node_size will assume default values (check configure nodes).
#   If optional argument node_size is set and either min_node_size or max_node_size is not set,
#   all nodes will have the size as described by node_size.
#   If min_node_size, max_node_size and node_size are set, node_size is disregarded.
#animate: Boolean. If False, generates one image for each of the probabilities.
#   If True, generates an animation.
#show_plot: Boolean. If False, does not show generated plot.
#   If True, shows the generated plot
#filename_prefix: str or None, default: None.
#   If None and show_plot is True, shows the plot and do not save in an output file.
#   If it is a string, saves plot in an output file;
#   if animate is True, the animation will be saved in a gif file, e.g. filename_prefix.gif;
#   if animate is False, saves a .png file for each of the probabilities,
#   e.g. filename_prefix-1.png, filename_prefix-2.png, etc.
#   IMPORTANT: we do NOT recommend to use animate=True, show_plot=True and filename_prefix as a str
#   because the show process will not be optimized (blit = False, check matplotlib documentation);
#   hence, the time between frames may not be the one expected.
#
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
def PlotProbabilityDistributionOnGraph(AdjMatrix, probabilities, animate=False,
        show_plot=True, filename_prefix=None, **kwargs):

    #vmin and vmax are default keywords used by networkx_draw.
    #if an invalid keyword is passed to nx.draw(), it does not execute
    kwargs['vmin'] = probabilities.min() #min_prob
    kwargs['vmax'] = probabilities.max() #max_prob

    G = nx.from_numpy_matrix(AdjMatrix)

    if len(probabilities.shape) == 1:
        probabilities = [probabilities]

    if not animate:
        for i in range(len(probabilities)):
            #TODO: set figure size according to graphdimension
            fig, ax = ConfigureFigure()
            DrawFigure(G, probabilities[i], ax=ax, **kwargs)

            #show or save image (or both)
            if filename_prefix is not None:
                filename_suffix = ( '-' + (len(probabilities)-1)//10 * '0' + str(i)
                        if len(probabilities) > 1 else '' )
                plt.savefig(filename_prefix + filename_suffix)
                if not show_plot:
                    plt.close()
            if show_plot:
                plt.show()

    else:
        fig, ax = ConfigureFigure()
        blit = filename_prefix is None #because optimization
        anim  = FuncAnimation(fig, AnimateFigure, frames=probabilities,
                fargs=(G, ax, kwargs), interval=200, repeat_delay=200, blit=blit)

        if filename_prefix is not None:
            anim.save(filename_prefix + '.gif')
        if show_plot:
            plt.show()

def AnimateFigure(probabilities, G, ax, kwargs):
    ax.clear()
    DrawFigure(G, probabilities, ax, **kwargs)

    return ax,

def DrawFigure(G, probabilities, ax, **kwargs):

    #setting kwargs for plotting
    #removes invalid keys for networkx draw
    #kwargs dictionary is updated by reference
    ConfigureNodes(G, probabilities,
            kwargs.pop['min_node_size'] if 'min_node_size' in kwargs else None,
            kwargs.pop['max_node_size'] if 'max_node_size' in kwargs else None,
            kwargs)

    nx.draw(G, ax=ax, **kwargs)

    #setting and drawing colorbar
    if 'cmap' in kwargs:
        ConfigureColorbar(ax, kwargs)

    #does not call plt.tight_layout() because it dramatically interferes
    #with animation time between frames


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

