"""
Module for plotting a probability distribution.
"""

import networkx as nx #TODO: import only needed functions?
import matplotlib.pyplot as plt
import numpy as np
from constants import DEBUG
from PIL import Image
from ._animation import *

if DEBUG:
    from time import time

# TODO: move to constants
plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams["figure.dpi"] = 100


# TODO: add documentation for 'fixed_probabilities' kwarg
# TODO: add option for changing figsize and dpi
# histogram is alias for bar width=1
def plot_probability_distribution(
        probabilities, plot_type='bar', animate=False, show=True,
        filename_prefix=None, interval=250, **kwargs):
    """
    Plots probability distribution of quantum walk.

    The probability distribution of multiple steps of a quantum walk
    may be plotted.
    The generated figures may be shown step-by-step,
    saved in multiple files, animated or saved as animation.

    Parameters
    ----------
    probabilities : :class:`numpy.ndarray`
        The probabilities of the walker to be found on each step
        of the quantum walk.
        Columns represent vertices and rows represent the walk steps.
    plot_type : str, default='bar'
        The type of graph to be plotted.
        The valid options are
        ``{'bar', 'line', 'graph', 'hist', 'histogram'}`` where
        ``'hist'`` is an alias for ``'histogram'``.
    animate : bool, default=False
        Whether or not to animate multiple plots.
        If ``False``, each quantum walk step generates an image.
        If ``True``, each quantum walk step is used as an animation frame.
    show : bool, default=True
        Whether or not to show plots or animation.
        With ``show=True`` we have:
        **Using Terminal**:
        After shown, press *q* to quit.
        If ``animate==False``,
        the quantum walk will be shown step-by-step;
        If ``animate==True``,
        the entire animation is shown in a new window.
        **Using Jupyter Notebook**:
        If ``animate==False``,
        all the quantum walk steps are shown at once.
        If ``animate==True``,
        the entire animation is shown as a html video.
    filename_prefix : str, default=None
        The filename path (with no format) where
        the plot(s) will be saved.
        If ``None`` no file is saved.
        Otherwise, if ``animate==False``,
        the j-step is saved in the ``filename_prefix-j.png`` file;
        if ``animate==True``,
        the entire walk is saved in the ``filename_prefix.fig`` file.
    interval : int, default=250
        Time in milliseconds that each frame is shown if ``animate==True``.
    graph : :class:`networkx.classes.graph`, optional
        The structure of the graph on which the walk occurs.
        The graph labels are used as plotting labels.
        **Important**: check Graph Plots subsection in other parameters.
    **kwargs : dict, optional
        Extra arguments to further customize plotting.
        Valid arguments depend on ``plot_type``.
        Check Other Parameters Section for details.

    Other Parameters
    ----------------
    fixed_probabilities : bool, optional
        If ``True`` or omitted, the reference maximum probability
        is the global one.
        If ``False``, the reference maximum probability depends on
        the current step, changing every image or frame.
        For example, if the global maximum probability is 1,
        ``min_node_size, max_node_size = (300, 3000)``,
        and the maximum probability of a given step is 0.5;
        then for ``fixed_probabilities=True``,
        the step maximum node size shown is halfway betweeen 300 and 3000,
        while for ``fixed_probabilities=False``,
        the step maximum node size shown is 3000.

    Bar Plots
        See :obj:`matplotlib.pyplot.bar` for more optional keywords.

    Graph Plots
        See :obj:`networkx.draw <networkx.drawing.nx_pylab.draw>`
        for more optional keywords.

        graph : :class:`networkx.classes.graph`, optional
            Either ``graph`` or ``adj_matrix`` must be set.
            If both are set, ``adj_matrix`` is discarded.
        adj_matrix : :class:`scipy.sparse.csr_matrix`, optional
            Adjacency matrix of the graph on which the walk occurs.
            Either ``graph`` or ``adj_matrix`` must be set.
        min_node_size, max_node_size : scalar, default=(300, 3000)
            By default, nodes sizes depend on the probability.
            ``min_node_size`` and ``max_node_size`` describe the
            inferior and superior limits for the node sizes, respectively.
        node_size : scalar or list of scalars, optional
            If ``node_size`` is a scalar,
            all nodes will have the same size.
            If ``node_size`` is a list of scalars,
            vertices may have different sizes and
            the length of ``node_size`` must match ``probabilities``.
            The ``node_size`` argument is ignored if both
            ``min_node_size`` and ``max_node_size`` are set.
        cmap : str, optional
            A colormap for representing vertices probabilities.
            if ``cmap='default'``, uses the ``'YlOrRd_r'`` colormap.
            For more colormap options, check
            `Matplolib's Colormap reference <https://matplotlib.org/stable/gallery/color/colormap_reference.html>`_.

    Histogram Plots
        See :obj:`matplotlib.pyplot.bar` for more optional keywords.
        The ``width`` keyword is always overriden.

    Line Plots
        See :obj:`matplotlib.pyplot.plot` for more optional keywords.


    Raises
    ------
    ValueError
        If ``plot_type`` has an invalid value.
    KeyError
        If ``plot_type == True`` and neither of the keywords
        ``graph`` or ``adj_matrix`` is set.

    Notes
    -----
    The core logic of the main implementation loop is more or less like follows.

    >>> preconfigure()
    >>> for prob in probabilities:
    >>>     configure()
    >>>     plot(prob)

    ``preconfigure()`` executes configurations that do
    not change between plots, e.g. nodes positions in graph plots.
    ``configure()`` executes configurates that must be
    (re)done for each iteration, e.g. setting figure size.
    ``plot()`` calls the appropriated plotting method with
    customization parameters,
    i.e. bar, line or graph plot with the respective valid kwargs.

    .. todo::
        - Accept the ``probabilities`` parameter as a list.
        - Use Graphviz instead of NetworkX to draw graphs.
            As noted by networkx's documentation:
            Proper graph visualization is hard, and we highly recommend
            that people visualize their graphs with tools dedicated to
            that task. 
            https://networkx.org/documentation/stable/reference/drawing.html
        - Implement ``repeat_delay`` parameter:
            An extra time to wait before the animation is repeated.
            Pull requests to the
            `Matplotlib animation writers
            <https://matplotlib.org/stable/api/animation_api.html#writer-classes>`_
            are needed.
        - Implement ``transparency`` parameter:
            change nodes transparency depending on probability.

    Examples
    --------
    .. todo::
        probabilities expects numpy array or matrix
    """
    plot_type = plot_type.lower()
    if plot_type == 'hist':
        plot_type = 'histogram'
    valid_plot_types = ['bar', 'line', 'graph', 'histogram']

    if plot_type not in valid_plot_types:
        raise ValueError(
            'Unexpected value for plot_type:' + str(plot_type) +
            '. One of the following was expected: ' + str(valid_plot_types)
        )

    # dictionaries for function pointers
    # preconfiguration: executed once before the loop starts
    preconfigs = {valid_plot_types[0]: _preconfigure_plot,
            valid_plot_types[1]: _preconfigure_plot,
            valid_plot_types[2]: _preconfigure_graph_plot,
            valid_plot_types[3]: _preconfigure_plot}
    # configuration: executed every iteration before plotting
    # expects return of fig, ax to be used for animations
    configs = {valid_plot_types[0]: _configure_plot_figure,
            valid_plot_types[1]: _configure_plot_figure,
            valid_plot_types[2]: _configure_graph_figure,
            valid_plot_types[3]: _configure_plot_figure}
    # plot functions: code for plotting the graph accordingly
    plot_funcs = {
        valid_plot_types[0]: _plot_probability_distribution_on_bars,
        valid_plot_types[1]: _plot_probability_distribution_on_line,
        valid_plot_types[2]: _plot_probability_distribution_on_graph,
        valid_plot_types[3]: _plot_probability_distribution_on_histogram
    }

    # preparing probabilities to shape requested by called functions
    if len(probabilities.shape) == 1:
        probabilities = [probabilities]

    # passes kwargs by reference to be updated accordingly
    preconfigs[plot_type](probabilities, kwargs)

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
                filename_suffix = (
                    '-' + (len(probabilities)-1)//10 * '0' + str(i)
                    if len(probabilities) > 1 else ''
                )
                plt.savefig(filename_prefix + filename_suffix)
                if not show:
                    plt.close()
            if show:
                plt.show()

        else:
            anim.add_frame(fig)

    if animate:
        anim.create_animation(interval)

        if filename_prefix is not None:
            anim.save_animation(filename_prefix)
        if show:
            anim.show_animation()


def _preconfigure_plot(probabilities, kwargs):
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

    if ('fixed_probabilities' not in kwargs
            or kwargs.pop('fixed_probabilities')):
        kwargs['min_prob'] = 0
        kwargs['max_prob'] = probabilities.max()


#kwargs passed by reference
def _preconfigure_graph_plot(probabilities, kwargs):
    """
    Configure static parameters for graph plot.

    Set parameters that need not to be changed for multiple plots
    or animation of a quantum walk.
    For example: the graph.
    The graph must be sent via kwargs,
    either via the 'graph' or the 'adj_matrix' keywords.

    Parameters
    ----------
    probabilities : list of floats
        Probabilities of the walker being found on each vertex.
    kwargs : dict
        Reference of kwargs containing all extra keywords.

    See Also
    --------
    _configure_nodes
    """

    # vmin and vmax are default keywords used by networkx_draw.
    # if an invalid keyword is passed to nx.draw(), it does not execute
    if ('fixed_probabilities' not in kwargs
            or kwargs['fixed_probabilities']):
        kwargs['vmin'] = 0 #min_prob
        kwargs['vmax'] = probabilities.max() #max_prob

    if 'graph' not in kwargs:
        if 'adj_matrix' not in kwargs:
            raise KeyError('One of the following keys must be provided:'
                           + str(['graph', 'adj_matrix']))
        adj_matrix = kwargs.pop('adj_matrix')
        kwargs['graph'] = nx.from_scipy_sparse_array(adj_matrix)
    if 'adj_matrix' in kwargs:
        # Pops adj_matrix if both graph and adj_matrix keywords are set;
        # otherwise invalid keyword may be raisen by networkx
        kwargs.pop('adj_matrix')

    if 'min_node_size' not in kwargs:
        kwargs['min_node_size'] = None
    if 'max_node_size' not in kwargs:
        kwargs['max_node_size'] = None

    # setting static kwargs for plotting
    # kwargs dictionary is updated by reference
    # TODO: change ConfigureNodes parameters
    # (remove G and use information from kwargs)
    _configure_nodes(kwargs['graph'], probabilities, kwargs)


def _configure_figure(num_vert, fig_width=None, fig_height=None):
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


def _configure_plot_figure(num_vert, fig_width=None, fig_height=None):
    """
    Set basic figure configuration for matplotlib plots.
    """
    
    fig, ax = _configure_figure(num_vert, fig_width, fig_height)

    plt.xlabel("Vertex ID", size=18)
    plt.ylabel("Probability", size=18)

    plt.tick_params(length=7, labelsize=14)

    return fig, ax


def _configure_graph_figure(num_vert=None, fig_width=None,
                            fig_height=None):
    return _configure_figure(num_vert, fig_width, fig_height)


def _plot_probability_distribution_on_bars(
        probabilities, ax, labels=None, graph=None,
        min_prob=None, max_prob=None, **kwargs
    ):
    """
    Plot probability distribution using matplotlib bar plot.

    Parameters
    ----------
    probabilities : list of floats
        Probabilities of the walker being found on each vertex.
    ax
        matplotlib ax on which the figure is drawn.
    {labels, graph, min_prob, max_prob} : optional
        Final configuration parameters.
        Refer to _posconfigure_plot_figure.
    **kwargs : dict, optional
        Extra parameters for plotting. Refer to matplotlib.pyplot.bar

    See Also
    --------
    _posconfigure_plot_figure : Sets final configuraation for exhibition.
    matplotlib.pyplot.bar
    """

    plt.bar(np.arange(len(probabilities)), probabilities, **kwargs)
    _posconfigure_plot_figure(ax, len(probabilities), labels, graph,
                             min_prob, max_prob)


def _plot_probability_distribution_on_histogram(
        probabilities, ax, labels=None, graph=None,
        min_prob=None, max_prob=None, **kwargs
    ):
    """
    Plot probability distribution as histogram.

    Alias for no space between bars in bar plot.

    Parameters
    ----------
    Refer to _plot_probability_distribution_on_bars

    See Also
    --------
    _plot_probability_distribution_on_bars
    """
    
    kwargs['width'] = 1
    _plot_probability_distribution_on_bars(
        probabilities, ax, labels, graph, min_prob, max_prob, **kwargs
    )


def _plot_probability_distribution_on_line(
        probabilities, ax, labels=None, graph=None,
        min_prob=None, max_prob=None, **kwargs
    ):
    """
    Plots probability distribution using matplotlib's line plot.

    Parameters
    ----------
    probabilities : list of floats
        Probabilities of the walker being found on each vertex.
    ax
        matplotlib ax on which the figure is drawn.
    {labels, graph, min_prob, max_prob} : optional
        Final configuration parameters.
        Refer to _posconfigure_plot_figure.
    **kwargs : dict, optional
        Extra parameters for plotting. Refer to matplotlib.pyplot.plot

    See Also
    --------
    _posconfigure_plot_figure : Sets final configuraation for exhibition.
    matplotlib.pyplot.plot
    """

    if 'marker' not in kwargs:
        kwargs['marker'] = 'o'
    plt.plot(np.arange(len(probabilities)), probabilities, **kwargs)

    _posconfigure_plot_figure(
        ax, len(probabilities), labels, graph, min_prob, max_prob
    )


def _posconfigure_plot_figure(ax, num_vert, labels=None, graph=None,
                              min_prob=None, max_prob=None):
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

        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=num_vert, integer=True)
        )
        if graph is not None:
            loc = ax.xaxis.get_major_locator()
            ind = loc().astype('int')
            ind = [i for i in ind if i >=0 and i < num_vert]

            nodes = list(graph.nodes())

            ax.set_xticks(ind, [nodes[i] for i in ind])

    if min_prob is not None and max_prob is not None:
        plt.ylim((min_prob, max_prob))


def _plot_probability_distribution_on_graph(probabilities, ax, **kwargs):
    """
    Draw graph and illustrates the probabilities depending on the
    volatile parameters.

    See Also
    --------
    _update_nodes : sets volatile parameters
    :obj:`networkx.draw <networkx.drawing.nx_pylab.draw>`
    _configure_colorbar
    """

    # UpdateNodes may create kwargs['node_size']
    # min_node_size and max_node_size are not valid keys
    # for nx.draw kwargs
    _update_nodes(probabilities, kwargs.pop('min_node_size'),
                  kwargs.pop('max_node_size'), kwargs)

    nx.draw(kwargs.pop('graph'), ax=ax,
            node_size=kwargs.pop('node_size'),
            **kwargs)
    # Note: nx.draw_networkx_labels dramatically increases plotting time.
    # It is called by nx.draw

    # setting and drawing colorbar
    if 'cmap' in kwargs:
        _configure_colorbar(ax, kwargs)

    if DEBUG:
        global start
        end = time()
        print("_plot_probability_distribution_on_graph: "
              + str(end - start) + 's')
        start = end


def _configure_nodes(G, probabilities, kwargs):
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
    _update_nodes
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
    # the user may choose any networkx graph_layout function
    # as long as only the graph is
    # the required parameter. Check
    # https://networkx.org/documentation/stable/reference/drawing.html#module-networkx.drawing.layout
    # For further customisation,
    # the user may call any networkx graph layout function
    # BEFORE calling plot_probability_distribution and
    # using its return as the 'pos' kwarg.
    if 'pos' not in kwargs:
        if 'graph_layout' in kwargs:
            func = kwargs.pop('graph_layout')
            kwargs['pos'] = func(G)
        else:
            kwargs['pos'] = nx.kamada_kawai_layout(G)


def _update_nodes(probabilities, min_node_size, max_node_size, kwargs):
    """
    Configure probability-related attributes of nodes.

    Configures attributes that may change depending on the probability
    at each (saved) step of the quantum walk.

    See Also
    --------
    _configure_nodes

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
        if ('fixed_probabilities' in kwargs
                and not kwargs.pop('fixed_probabilities')):
            kwargs['vmin'] = 0
            kwargs['vmax'] = probabilities.max()

        # calculating size of each node acording to probability 
        # as a function f(x) = ax + b where b = min_size and
        # max_size = a*(max_prob-min_prob) + min_size
        a = ((max_node_size - min_node_size)
             / (kwargs['vmax'] - kwargs['vmin']))
        kwargs['node_size'] = list(map(
            lambda x: a*x + min_node_size, probabilities
        ))


def _configure_colorbar(ax, kwargs):
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
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    sm = plt.cm.ScalarMappable(
        cmap=kwargs['cmap'],
        norm=plt.Normalize(vmin=kwargs['vmin'],
                           vmax=kwargs['vmax'])
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2.5%', pad=0.01)
    cbar = plt.colorbar(
        sm,
        ticks=np.linspace(kwargs['vmin'], kwargs['vmax'], num=5),
        cax=cax
    )

    cbar.ax.tick_params(labelsize=14, length=7)


#########################################################################################

if DEBUG:
    start = time()
