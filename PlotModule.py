"""
Module for plotting a probability distribution.
"""

import networkx as nx #TODO: import only needed functions?
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import linspace
from numpy import arange
from ModifiedNetworkXFunctions import *
from Constants import DEBUG
from PIL import Image
from AnimationModule import *

if DEBUG:
    from time import time

# TODO: move to constants
plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams["figure.dpi"] = 100

# TODO: add documentation for 'fixed_probabilities' kwarg
# TODO: add option for changing figsize and dpi
# histogram is alias for bar width=1
def PlotProbabilityDistribution(
        probabilities, plot_type='bar', animate=False, show_plot=True,
        filename_prefix=None, interval=250, repeat_delay=250, **kwargs):
    """
    #TODO: probabilities expects numpy array or matrix
    #TODO: use graphviz to draw as noted by networkx's documentation:
    #TODO: if that's the case, try to optimize plot and animations before changing to use graphviz
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
    #interval: int (kwargs). Interval between frames case animate=True,
    #   check matplotlib.animation.FuncAnimation for more details.
    #   If interval is set and animate=False, an exception will be thrown
    #repeat_delay: int (kwargs). Delay before repeating the animation from the start
    #   (the duration is summed up with interval). Check matplotlib.animation.FuncAnimation for details.
    #   If repeat_delay is set and animate=False, an exception will be thrown
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
    """

    plot_type = plot_type.lower()
    if plot_type == 'hist':
        plot_type = 'histogram'
    valid_plot_types = ['bar', 'line', 'graph', 'histogram']

    if plot_type not in valid_plot_types:
        raise ValueError('Unexpected value for plot_type:' + str(plot_type) +
                '. One of the following was expected: ' + str(valid_plot_types))

    # dictionaries for function pointers
    # preconfiguration: executed once before the loop starts
    preconfigs = {valid_plot_types[0]: PreconfigurePlot,
            valid_plot_types[1]: PreconfigurePlot,
            valid_plot_types[2]: PreconfigureGraphPlot,
            valid_plot_types[3]: PreconfigurePlot}
    # configuration: executed every iteration before plotting
    # expects return of fig, ax to be used for animations
    configs = {valid_plot_types[0]: _ConfigurePlotFigure,
            valid_plot_types[1]: _ConfigurePlotFigure,
            valid_plot_types[2]: _ConfigureGraphFigure,
            valid_plot_types[3]: _ConfigurePlotFigure}
    # plot functions: code for plotting the graph accordingly
    plot_funcs = {valid_plot_types[0]: _PlotProbabilityDistributionOnBars,
            valid_plot_types[1]: _PlotProbabilityDistributionOnLine,
            valid_plot_types[2]: _PlotProbabilityDistributionOnGraph,
            valid_plot_types[3]: _PlotProbabilityDistributionOnHistogram}

    # preparing probabilities to shape requested by called functions
    if len(probabilities.shape) == 1:
        probabilities = [probabilities]

    # passes kwargs by reference to be updated accordingly
    preconfigs[plot_type](probabilities, kwargs)

    # TODO: duplicated code with _PlotProbabilityDistributionOnGraph... refactor

    if animate:
        anim = Animation()

    for i in range(len(probabilities)):
        # TODO: set figure size according to graph dimension
        # TODO: check for kwargs
        fig, ax = configs[plot_type](probabilities.shape[1]) 

        plot_funcs[plot_type](probabilities[i], ax, **kwargs)

        plt.tight_layout()

        # saves or shows image (or both)
        if not animate:
            if filename_prefix is not None:
                # enumarating the plot
                filename_suffix = ( '-' + (len(probabilities)-1)//10 * '0' + str(i)
                        if len(probabilities) > 1 else '' )
                plt.savefig(filename_prefix + filename_suffix)
                if not show_plot:
                    plt.close()
            if show_plot:
                plt.show()

        else:
            anim.AddFrame(fig)

    if animate:
        anim.CreateAnimation(interval, repeat_delay)

        if filename_prefix is not None:
            anim.SaveAnimation(filename_prefix)
        if show_plot:

def PreconfigurePlot(probabilities, kwargs):
    """
    Configure static parameters for matplotlib plot.

    Set parameters that need not to be changed for multiple plots
    or animation of a quantum walk.
    For example: minimum and maximum allowed node size.

    Parameters
    ----------
    probabilities : list of floats
        Probabilities of the walker being found on each vertex.
    kwargs : dict
        Reference of kwargs containing all extra keywords.
    """

    if 'fixed_probabilities' not in kwargs or kwargs.pop('fixed_probabilities'):
        kwargs['min_prob'] = 0
        kwargs['max_prob'] = probabilities.max()

#kwargs passed by reference
def PreconfigureGraphPlot(probabilities, kwargs):
    """
    Configure static parameters for graph plot.

    Set parameters that need not to be changed for multiple plots
    or animation of a quantum walk.
    For example: minimum and maximum allowed node size.

    Parameters
    ----------
    probabilities : list of floats
        Probabilities of the walker being found on each vertex.
    kwargs : dict
        Reference of kwargs containing all extra keywords.

    See Also
    --------
    _ConfigureNodes
    """

    # vmin and vmax are default keywords used by networkx_draw.
    # if an invalid keyword is passed to nx.draw(), it does not execute
    if 'fixed_probabilities' not in kwargs or kwargs['fixed_probabilities']:
        kwargs['vmin'] = 0 #min_prob
        kwargs['vmax'] = probabilities.max() #max_prob

    if 'graph' not in kwargs:
        kwargs['graph'] = nx.from_numpy_matrix( kwargs.pop('adj_matrix') )
    if 'adj_matrix' in kwargs: #then kwargs['graph'] is set and the adj_matrix can be disregarded
        kwargs.pop('adj_matrix')

    if 'min_node_size' not in kwargs:
        kwargs['min_node_size'] = None
    if 'max_node_size' not in kwargs:
        kwargs['max_node_size'] = None

    # setting static kwargs for plotting
    # kwargs dictionary is updated by reference
    # TODO: change ConfigureNodes parameters (remove G and use information from kwargs)
    _ConfigureNodes(kwargs['graph'], probabilities, kwargs)

def _ConfigureFigure(num_vert, fig_width=None, fig_height=None):
    """
    Set basic figure configuration.

    Creates a figure with size depending on the parameters.

    Parameters
    ----------
    num_vert: int
        number of vertices in the graph
        .. todo:: set the figure size according to num_vert
    fig_width, fig_height : float, optional
        Custom figure width and height, respectively.
        If not set, the default value is used.

    Returns
    -------
    fig : current figure
    ax : current figure axes
    """

    #TODO: set figure size according to graph dimension
    if fig_width is None:
        fig_width = plt.rcParams["figure.figsize"][0]
    if fig_height is None:
        fig_height = plt.rcParams["figure.figsize"][1]

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = plt.gca()

    return fig, ax


def _ConfigurePlotFigure(num_vert, fig_width=None, fig_height=None):
    """
    Set basic figure configuration for matplotlib plots.
    """
    
    fig, ax = _ConfigureFigure(num_vert, fig_width, fig_height)

    plt.xlabel("Vertex ID", size=18)
    plt.ylabel("Probability", size=18)

    plt.tick_params(length=7, labelsize=14)

    return fig, ax


def _ConfigureGraphFigure(num_vert=None, fig_width=None, fig_height=None):
    return _ConfigureFigure(num_vert, fig_width, fig_height)


def _PlotProbabilityDistributionOnBars(probabilities, ax, labels=None,
        graph=None, min_prob=None, max_prob=None, **kwargs):
    """
    Plot probability distribution using matplotlib bar plot.

    Parameters
    ----------
    probabilities : list of floats
        Probabilities of the walker being found on each vertex.
    ax
        matplotlib ax on which the figure is drawn.
    {labels, graph, min_prob, max_prob} : optional
        Final configuration parameters. Refer to _PosconfigurePlotFigure.
    **kwargs : dict, optional
        Extra parameters for plotting. Refer to matplotlib.pyplot.bar

    See Also
    --------
    _PosconfigurePlotFigure : Sets final configuraation for exhibition.
    matplotlib.pyplot.bar
    """

    bars = plt.bar(arange(len(probabilities)), probabilities, **kwargs)
    _PosconfigurePlotFigure(ax, len(probabilities), labels, graph,
                             min_prob, max_prob)

    return bars, #used for animation


def _PlotProbabilityDistributionOnHistogram(probabilities, ax, labels=None,
        graph=None, min_prob=None, max_prob=None, **kwargs):
    """
    Plot probability distribution as histogram.

    Alias for no space between bars in bar plot.

    Parameters
    ----------
    Refer to _PlotProbabilityDistributionOnBars

    See Also
    --------
    _PlotProbabilityDistributionOnBars
    """
    
    kwargs['width'] = 1
    _PlotProbabilityDistributionOnBars(probabilities, ax, labels, graph,
                                        min_prob, max_prob, **kwargs)


def _PlotProbabilityDistributionOnLine(probabilities, ax, labels=None,
        graph=None, min_prob=None, max_prob=None, **kwargs):
    """
    Plots probability distribution using matplotlib's line plot.

    Parameters
    ----------
    probabilities : list of floats
        Probabilities of the walker being found on each vertex.
    ax
        matplotlib ax on which the figure is drawn.
    {labels, graph, min_prob, max_prob} : optional
        Final configuration parameters. Refer to _PosconfigurePlotFigure.
    **kwargs : dict, optional
        Extra parameters for plotting. Refer to matplotlib.pyplot.plot

    See Also
    --------
    _PosconfigurePlotFigure : Sets final configuraation for exhibition.
    matplotlib.pyplot.plot
    """

    if 'marker' not in kwargs:
        kwargs['marker'] = 'o'
    plt.plot(arange(len(probabilities)), probabilities, **kwargs)

    _PosconfigurePlotFigure(ax, len(probabilities), labels, graph, min_prob, max_prob)


def _PosconfigurePlotFigure(ax, num_vert, labels=None, graph=None, min_prob=None, max_prob=None):
    """
    Add final touches to the plotted figure.

    The labels are added to the figure and
    the figure's y-axis range is configured.

    Parameters
    ----------
    ax
        matplotlib ax on which the figure is drawn
    num_vert : int
        number of vertices in the graph
    labels : dict, optional
        Labels to be shown on graph.
        If `graph` is `None`, `labels.keys()` must be integers
        from 0 to `num_vert` - 1.
    graph : networkx graph, optional
        Graph on which the quantum walk occured.
        if not None, the graph nodes are used to match `labels` keys.
    min_prob, max_prob : int, optional
        If both are set, describe the y-axis limit.
    """
    if labels is not None:
        if graph is None:
            ax.set_xticks( list(labels.keys()), list(labels.values()) )
        else:

            nodes = list(graph.nodes())
            nodes = {i : labels[ nodes[i] ] for i in range(num_vert)
                        if nodes[i] in labels}

            ax.set_xticks( list(nodes.keys()), list(nodes.values()) )

    else:
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(nbins=num_vert, integer=True))
        if graph is not None:
            loc = ax.xaxis.get_major_locator()
            ind = loc().astype('int')
            ind = [i for i in ind if i >=0 and i < num_vert]

            nodes = list(graph.nodes())

            ax.set_xticks(ind, [nodes[i] for i in ind])

    if min_prob is not None and max_prob is not None:
        plt.ylim((min_prob, max_prob))


def _PlotProbabilityDistributionOnGraph(probabilities, ax, **kwargs):
    """
    Draw graph and illustrates the probabilities depending on the
    volatile parameters.

    See Also
    --------
    _UpdateNodes : sets volatile parameters
    nx_draw
    _ConfigureColorbar
    """

    # UpdateNodes may create kwargs['node_size']
    # min_node_size and max_node_size are not valid keys for nx_draw kwargs
    _UpdateNodes(probabilities, kwargs.pop('min_node_size'),
                  kwargs.pop('max_node_size'), kwargs)

    nx_draw(kwargs.pop('graph'), ax=ax,
            node_size=kwargs.pop('node_size'),
            **kwargs)
    # Note: nx.draw_networkx_labels dramatically increases plotting time.
    # It is called by nx_draw

    # setting and drawing colorbar
    if 'cmap' in kwargs:
        _ConfigureColorbar(ax, kwargs)

    if DEBUG:
        global start
        end = time()
        print("_PlotProbabilityDistributionOnGraph: " + str(end - start) +'s')
        start = end

    return nodes, labels

def _ConfigureNodes(G, probabilities, kwargs):
    """
    Configure static attributes of nodes.

    Configures nodes attributes that will not change
    during multiple plots or an animation of a quantum walk.

    Parameters
    ----------
    kwargs: dict
        Reference to the dictionary **kwargs.
        Some entries may be added or altered.

    See Also
    --------
    _UpdateNodes
    """
    # setting colormap related attributes
    if 'cmap' in kwargs:
        if kwargs['cmap'] == 'default':
            kwargs['cmap'] = 'YlOrRd_r'

    # setting node attributes
    if 'edgecolors' not in kwargs:
        kwargs['edgecolors'] = 'black'

    if 'linewidths' not in kwargs:
        kwargs['linewidths'] = 1

    if 'with_labels' not in kwargs:
        kwargs['with_labels'] = True

    if kwargs['with_labels'] and 'font_color' not in kwargs:
        kwargs['font_color'] = 'black'

    # calculates vertices positions.
    # needed to do beforehand in order to fix position for multiple steps
    # the user may choose any networkx graph_layout function as long as only the graph is
    # the required parameter. Check
    # https://networkx.org/documentation/stable/reference/drawing.html#module-networkx.drawing.layout
    # For further customisation, the user may call any networkx graph layout function
    # BEFORE calling PlotProbabilityDistribution and using its return as the 'pos' kwarg.
    if 'pos' not in kwargs:
        if 'graph_layout' in kwargs:
            func = kwargs.pop('graph_layout')
            kwargs['pos'] = func(G)
        else:
            kwargs['pos'] = nx.kamada_kawai_layout(G)

def _UpdateNodes(probabilities, min_node_size, max_node_size, kwargs):
    """
    Configure probability-related attributes of nodes.

    Configures attributes that may change depending on the probability
    at each (saved) step of the quantum walk.

    See Also
    --------
    _ConfigureNodes

    Notes
    -----
    The separation between UpdateNodes and ConfigureNodes optimizes
    plotting multiple images.
    """
    if 'cmap' in kwargs:
        kwargs['node_color'] = probabilities

    if 'node_size' not in kwargs:
        if min_node_size is None:
            min_node_size = 300
        if max_node_size is None:
            max_node_size = 3000

    if min_node_size is not None and max_node_size is not None:
        if 'fixed_probabilities' in kwargs and not kwargs.pop('fixed_probabilities'):
            kwargs['vmin'] = 0
            kwargs['vmax'] = probabilities.max()

        # calculating size of each node acording to probability 
        # as a function f(x) = ax + b where b = min_size and
        # max_size = a*(max_prob-min_prob) + min_size
        a = (max_node_size - min_node_size) / (kwargs['vmax'] - kwargs['vmin'])
        kwargs['node_size'] = list(map(
                lambda x: a*x + min_node_size, probabilities
            ))


def _ConfigureColorbar(ax, kwargs):
    """
    Add a colorbar in the figure besides the given ax

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Ax on which the plot was drawn.
    kwargs : dict
        Dictionary containing the keys 'cmap', 'vmin' and 'vmax'.
        'vmin' and 'vmax' describe the inferior and superior limit for
        colorbar values, respectively.
        'cmap' describes a valid matplotlib colormap
    """

    sm = plt.cm.ScalarMappable(cmap=kwargs['cmap'],
            norm=plt.Normalize(vmin=kwargs['vmin'], vmax=kwargs['vmax']))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2.5%', pad=0.01)
    cbar = plt.colorbar(sm, ticks=linspace(kwargs['vmin'], kwargs['vmax'], num=5), cax=cax)

    cbar.ax.tick_params(labelsize=14, length=7)


#########################################################################################

if DEBUG:
    start = time()
