import networkx as nx #TODO: import only needed functions?
import matplotlib.pyplot as plt
import numpy as np
import re
from PIL import Image
from ..graph import *
from ..quantum_walk import QuantumWalk
from matplotlib.animation import FuncAnimation
from scipy.sparse import csr_array

python_range = range

# histogram is alias for bar width=1
def plot_probability_distribution(
        probabilities, plot=None, animate=False, show=True,
        filename=None, interval=250, figsize=(10, 6), dpi=100,
        repeat=True, repeat_delay=2500, range=None,
        time_step=1, **kwargs):
    """
    Plot the probability distributions of quantum walk states.

    This function allows plotting the probability distributions 
    for multiple steps of a quantum walk evolution.
    The generated figures can be displayed step-by-step, 
    saved as individual files, or used to create and save animations.

    Parameters
    ----------
    probabilities : :class:`numpy.ndarray`
        An array representing the probabilities of the walker being 
        found at each vertex during the quantum walk evolution.
        Each column corresponds to a vertex, 
        while each row represents a step in the walk.

    plot : str, default=None
        The plot type.
        The valid options are
        ``{'line', 'bar', 'graph', 'histogram', 'plane'}``.
        If ``None``, uses default plotting. Usually ``bar``,
        but default plotting changes according to ``graph``.

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

    filename : str, default=None
        The filename path (with no format) where
        the plot(s) will be saved.
        If ``None`` no file is saved.
        Otherwise, if ``animate==False``,
        the j-step is saved in the ``filename-j.png`` file;
        if ``animate==True``,
        the entire walk is saved in the ``filename.fig`` file.

    graph : optional
        The structure of the graph on which the walk takes place.
        The graph labels are used as plotting labels.
        **Important**: check Graph Plots subsection in other parameters.

        The following types are acceptable.
        
        * :class:`hiperwalk.Graph`
            Hiperwalk Graph.
            It is used to generate default plotting for specific graphs.
            User-specified values are not overridden.
        * :class:`networkx.classes.graph`,
            NetworkX Graph
        * :class:`scipy.sparse.csr_matrix`
            Adjacency matrix.

    rescale : bool, optional
        If ``False`` or omitted, the reference maximum probability
        is the global one.
        If ``True``, the reference maximum probability depends on
        the current step, changing every image or frame.
        For example, if the global maximum probability is 1,
        ``min_node_size, max_node_size = (300, 3000)``,
        and the maximum probability of a given step is 0.5;
        then for ``rescale=False``,
        the step maximum node size shown is 1650
        (halfway betweeen 300 and 3000),
        while for ``rescale=True``,
        the step maximum node size shown is 3000.

    animate : bool, default=False
        Whether or not to animate multiple plots.
        If ``False``, each quantum walk step generates an image.
        If ``True``, each quantum walk step is used as an animation frame.

    interval : int, default=250
        Time in milliseconds that each frame is shown if ``animate==True``.

    figsize : tuple, default=(10, 6)
        Figure size in inches. Must be a tuple in the format (WIDTH, HEIGHT).

    dpi : float, default=100
        Figure resolution in dots-per-inch.

    repeat : bool, default: True
        Whether or not to repeat the animation.
        See :class:`matplotlib.animation.FuncAnimation`

    repeat_delay : int, default=2500
        Delay in milliseconds between animation repetitions.
        See :class:`matplotlib.animation.FuncAnimation`

    range: tuple of int, default=None
        Range used for the quantum walk simulation.
        See :meth:`QuantumWalk.simulate` for details.
        If ``range is not None``,
        each figure or frame will have a label with the corresponding time.
        The values shown are
        ``time_step * np.arange(*range)``.

    time_step: float, default=1
        Time interval between simulation steps.

    labels : dict, optional
        Labels to be shown on graph.
        If `graph` is `None`, `labels.keys()` must be integers
        from 0 to `num_vert` - 1.
        If `graph is not None`, `labels.keys()` must be
        labels of the corresponding :class:`networkx.Graph`'s nodes.

    **kwargs : dict, optional
        Extra arguments to further customize plotting.
        Valid arguments depend on ``plot``.
        Check Other Parameters Section for details.

    Other Parameters
    ----------------
    Line Plots
        See :obj:`matplotlib.pyplot.plot` for more optional keywords.

    Bar Plots
        See :obj:`matplotlib.pyplot.bar` for more optional keywords.

    Graph Plots
        See :obj:`networkx.draw <networkx.drawing.nx_pylab.draw>`
        for more optional keywords.

        graph :
            Graph structure.
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
            if ``cmap='default'``, uses the ``'viridis'`` colormap.
            For more colormap options, check
            `Matplolib's Colormap reference
            <https://matplotlib.org/stable/gallery/color/colormap_reference.html>`_.

    Histogram Plots
        See :obj:`matplotlib.pyplot.bar` for more optional keywords.
        The ``width`` keyword is always overriden.

    Plane Plots
        dimensions: 2-tuple of int
            plane dimensions in ``(x_dim, y_dim)`` format.


    Raises
    ------
    ValueError
        If ``plot`` has an invalid value.
    KeyError
        If ``plot == 'graph' `` and keyword ``graph`` is not set.

    Notes
    -----
    The core logic of the main implementation loop is more or less like follows.

    >>> preconfigure() # doctest: +SKIP
    >>> for prob in probabilities: # doctest: +SKIP
    >>>     configure() # doctest: +SKIP
    >>>     plot(prob) # doctest: +SKIP

    ``preconfigure()`` executes configurations that do
    not change between plots, e.g. nodes positions in graph plots.
    ``configure()`` executes configurates that must be
    (re)done for each iteration, e.g. setting figure size.
    ``plot()`` calls the appropriated plotting method with
    customization parameters,
    i.e. bar, line or graph plot with the respective valid kwargs.
    """
    # TODO: option to use Graphviz instead of NetworkX to draw graphs.
    #      As noted by networkx's documentation:
    #      Proper graph visualization is hard, and we highly recommend
    #      that people visualize their graphs with tools dedicated to
    #      that task. 
    #      https://networkx.org/documentation/stable/reference/drawing.html

    # Figure size

    if len(figsize) == 2:
        fig_width, fig_height = figsize
    else:
        raise ValueError(
            'figsize must be a tuple in the format (WIDTH, HEIGHT)'
        )

    # passes kwargs by reference to be updated accordingly

    if 'graph' in kwargs:
        plot = _default_graph_kwargs(kwargs, plot)

    if plot is None:
        plot = 'bar'

    plot = plot.lower()
    valid_plots = ['bar', 'line', 'graph', 'histogram', 'plane']

    if plot not in valid_plots:
        raise ValueError(
            'Unexpected value for plot:' + str(plot) +
            '. One of the following was expected: ' + str(valid_plots)
        )

    # dictionaries for function pointers
    # preconfiguration: executed once before the loop starts
    preconfigs = {valid_plots[0]: _preconfigure_plot,
            valid_plots[1]: _preconfigure_plot,
            valid_plots[2]: _preconfigure_graph_plot,
            valid_plots[3]: _preconfigure_plot,
            valid_plots[4]: _preconfigure_plot}
    # configuration: executed every iteration before plotting
    # expects return of fig, ax to be used for animations
    configs = {valid_plots[0]: _configure_plot_figure,
            valid_plots[1]: _configure_plot_figure,
            valid_plots[2]: _configure_graph_figure,
            valid_plots[3]: _configure_plot_figure,
            valid_plots[4]: _configure_plane_figure}
    # plot functions: code for plotting the graph accordingly
    plot_funcs = {
        valid_plots[0]: _plot_probability_distribution_on_bars,
        valid_plots[1]: _plot_probability_distribution_on_line,
        valid_plots[2]: _plot_probability_distribution_on_graph,
        valid_plots[3]: _plot_probability_distribution_on_histogram,
        valid_plots[4]: _plot_probability_distribution_on_plane
    }

    update_animation = {
        valid_plots[0]: _update_animation_bars,
        valid_plots[1]: _update_animation_line,
        valid_plots[2]: None,
        valid_plots[3]: _update_animation_bars,
        valid_plots[4]: None
    }

    # preparing probabilities to shape requested by called functions
    if len(probabilities.shape) == 1:
        probabilities = np.array([probabilities])

    # passes kwargs by reference to be updated accordingly
    preconfigs[plot](probabilities, kwargs)

    # matches a valid filename
    filename_with_ext_pattern = re.compile(r'(.+)\.(.{3,4})$')

    if not animate:
        t = [None] * len(probabilities)
        if range is not None:
            t = QuantumWalk._range_to_tuple(range)
            t = time_step*np.arange(*t)

        for i in python_range(len(probabilities)):
            # TODO: set figure size according to graph dimensions
            # TODO: check for kwargs
            fig, ax = configs[plot](fig_width=fig_width,
                                    fig_height=fig_height,
                                    dpi=dpi)

            plot_funcs[plot](probabilities[i], ax, time=t[i], **kwargs)

            plt.tight_layout()

            # saves or shows image (or both)
            if filename is not None:
                filename_suffix = str(i).zfill(
                        len(str(len(probabilities) - 1)))
                # Use the base name, suffix, and extension to assemble the
                # name if filename was provided with an extension.
                if m := filename_with_ext_pattern.match(filename):
                    name = m.group(1)
                    ext = m.group(2)
                    plt.savefig(f'{name}-{filename_suffix}.{ext}')
                else:
                    plt.savefig(filename if len(probabilities) == 1
                                else filename + '-' + filename_suffix)
                if not show:
                    plt.close()
            if show:
                plt.show()

    else:
        fig, ax = configs[plot](fig_width=fig_width,
                                fig_height=fig_height,
                                dpi=dpi)

        from itertools import cycle
        t = None
        if range is not None:
            t = QuantumWalk._range_to_tuple(range)
            t[0] -= t[2]
            t = time_step*np.arange(*t)
            t = cycle(t)

        if plot == 'plane':
            from functools import partial
            surf, cbar = plot_funcs[plot](probabilities[0], ax,
                                          **kwargs)

            anim = FuncAnimation(
                    fig,
                    partial(plot_funcs[plot],
                            ax=ax,
                            surf=surf,
                            cbar=cbar,
                            time=t,
                            **kwargs),
                    frames=probabilities,
                    interval=interval,
                    repeat=repeat,
                    repeat_delay=repeat_delay)

        elif plot == 'graph':
            from functools import partial
            ax, cbar = plot_funcs[plot](probabilities[0], ax,
                                        **kwargs)

            anim = FuncAnimation(
                    fig,
                    partial(plot_funcs[plot], ax=ax, cbar=cbar,
                            time=t, **kwargs),
                    frames=probabilities,
                    interval=interval,
                    repeat=repeat,
                    repeat_delay=repeat_delay)
        else:
            artists = plot_funcs[plot](probabilities[0], ax, **kwargs)
            anim = FuncAnimation(
                    fig,
                    update_animation[plot],
                    frames=probabilities,
                    fargs=(artists,
                           ax if 'min_prob' not in kwargs else None,
                           t),
                    interval=interval,
                    repeat=repeat,
                    repeat_delay=repeat_delay)

        if filename is not None:
            # Use .gif extension if the user didn't provide an
            # extension for the animation file.
            if not filename_with_ext_pattern.match(filename):
                filename = filename + '.gif'
            anim.save(filename)
        if show:
            if _is_in_notebook():
                from IPython import display

                # embedding animation in jupyter notebook
                video = anim.to_jshtml()
                html = display.HTML(video)
                display.display(html)

                plt.close()
            else:
                plt.show()

def _default_graph_kwargs(kwargs, plot):
    if ((plot is None or plot == 'graph' or plot == 'plane')
        and not 'cmap' in kwargs
    ):
        kwargs['cmap'] = 'default'

    if 'cmap' in kwargs:
        if kwargs['cmap'] == 'default':
            kwargs['cmap'] = 'viridis'

    graph = kwargs['graph']

    # Hiperwalk Grid
    if hasattr(graph, '_euc_dim') and graph._euc_dim == 2:
        if plot is None:
            plot = 'plane'
        if plot == 'plane' and 'dimensions' not in kwargs:
            kwargs['dimensions'] = graph.dimensions()
        return plot

    # Hiperwalk Hypercube
    plot = 'graph' if plot is None else plot

    is_hypercube = False
    try:
        is_hypercube = graph.degree(0) == graph.dimension()
    except AttributeError:
        pass

    if is_hypercube and (plot is None or plot == 'graph'):
        plot = 'graph'

        nx_graph = nx.from_scipy_sparse_array(graph.adjacency_matrix())
        for v in nx_graph:
            try:
                nx_graph.nodes[v]["subset"] = v.bit_count()
            except AttributeError:
                nx_graph.nodes[v]["subset"] = bin(v).count('1')

        kwargs['pos'] = nx.multipartite_layout(nx_graph)
        dim = graph.dimension()
        kwargs['graph'] = nx_graph
        kwargs['edge_color'] = (0, 0, 0, max(0.002, 2**(-dim/2)))
        kwargs['labels'] = {0: 0, 2**dim - 1 : 2**dim - 1}
        kwargs['min_node_size'] = 1
        kwargs['max_node_size'] = 1000
        kwargs['edgecolors'] = None

        return plot

    # other graphs
    if hasattr(graph, 'number_of_vertices'):
        nx_graph = nx.Graph(graph.adjacency_matrix())
    elif hasattr(graph, 'number_of_nodes'):
        nx_graph = graph
    else:
        nx_graph = nx.Graph(graph)

    kwargs['graph'] = nx_graph

    return plot


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

    if ('rescale' not in kwargs or not kwargs.pop('rescale')):
        kwargs['min_prob'] = 0
        kwargs['max_prob'] = probabilities.max()


#kwargs passed by reference
def _preconfigure_graph_plot(probabilities, kwargs):
    """
    Configure static parameters for graph plot.

    Set parameters that need not to be changed for multiple plots
    or animation of a quantum walk.
    For example: the graph.
    The graph must be sent via kwargs via the 'graph' keyword.

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

    if ('rescale' not in kwargs or not kwargs['rescale']):
        kwargs['min_prob'] = 0 #min_prob
        kwargs['max_prob'] = probabilities.max() #max_prob
        kwargs['rescale'] = False

    if 'graph' not in kwargs:
        raise KeyError("'graph' kwarg not provided.")

    graph = kwargs['graph']
    if isinstance(graph, csr_array):
        kwargs['graph'] = nx.from_scipy_sparse_array(graph)
    elif isinstance(graph, Graph):
        adj_matrix = graph.adjacency_matrix()
        kwargs['graph'] = nx.from_scipy_sparse_array(adj_matrix)

    if 'min_node_size' not in kwargs:
        kwargs['min_node_size'] = None
    if 'max_node_size' not in kwargs:
        kwargs['max_node_size'] = None

    # setting static kwargs for plotting
    # kwargs dictionary is updated by reference
    # TODO: change ConfigureNodes parameters
    # (remove G and use information from kwargs)
    _configure_nodes(kwargs['graph'], probabilities, kwargs)


def _configure_figure(fig_width, fig_height, dpi):
    """
    Set basic figure configuration.

    Creates a figure with size depending on the parameters.

    Parameters
    ----------
    num_vert: int
        number of vertices in the graph

    fig_width, fig_height : float, optional
        Custom figure width and height, respectively.
        If not set, the default value is used.

    Returns
    -------
    fig : current figure
    ax : current figure axes
    """

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    ax = plt.gca()

    return fig, ax


def _configure_plot_figure(fig_width, fig_height, dpi):
    """
    Set basic figure configuration for matplotlib plots.
    """
    
    fig, ax = _configure_figure(fig_width, fig_height, dpi)

    plt.xlabel("Vertex", size=18)
    plt.ylabel("Probability", size=18)

    plt.tick_params(length=7, labelsize=14)

    return fig, ax


def _configure_graph_figure(fig_width,
                            fig_height,
                            dpi):
    return _configure_figure(fig_width, fig_height, dpi)

def _configure_plane_figure(fig_width,
                            fig_height,
                            dpi):

    #fig, ax =_configure_figure(num_vert, fig_width, fig_height) 
    #ax = fig.add_subplot(projection='3d')
    fig, ax = plt.subplots(figsize=(fig_width, fig_height),
                           subplot_kw={"projection": "3d"},
                           dpi=dpi)

    ax.tick_params(length=10, width=1, labelsize=16, pad=10)
    ax.set_xlabel('Vertex X', labelpad=15, fontsize=18)
    ax.set_ylabel('Vertex Y', labelpad=15, fontsize=18)
    ax.set_zlabel('Probability', labelpad=30, fontsize=18)
    return fig, ax


def _plot_probability_distribution_on_bars(
        probabilities, ax, labels=None, graph=None,
        min_prob=None, max_prob=None, time=None, **kwargs
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

    bars = plt.bar(np.arange(len(probabilities)), probabilities, **kwargs)
    _posconfigure_plot_figure(ax, len(probabilities), labels, graph,
                             min_prob, max_prob, time)
    return [bars]

def _update_animation_bars(frame, bars, ax, time):
    bars = bars[0]
    for i, bar in enumerate(bars):
        bar.set_height(frame[i])

    if ax is not None:
        _rescale_axis(ax, 0, frame.max())

    if time is not None:
        t = next(time)
        title = plt.title('Time: ' + str(t), loc='right')

    return [bars]


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
    return _plot_probability_distribution_on_bars(
        probabilities, ax, labels, graph, min_prob, max_prob, **kwargs
    )


def _plot_probability_distribution_on_line(
        probabilities, ax, labels=None, graph=None,
        min_prob=None, max_prob=None, time=None, **kwargs
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

    line = plt.plot(np.arange(len(probabilities)),
                     probabilities, **kwargs)

    _posconfigure_plot_figure(
        ax, len(probabilities), labels, graph, min_prob, max_prob, time
    )

    return line

def _update_animation_line(frame, line, ax, time):
    line = line[0]
    line.set_ydata(frame)
    if ax is not None:
        _rescale_axis(ax, 0, frame.max())

    if time is not None:
        t = next(time)
        title = plt.title('Time: ' + str(t), loc='right')

    return [line]

def _posconfigure_plot_figure(ax, num_vert, labels=None, graph=None,
                              min_prob=None, max_prob=None, time=None):
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

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if graph is not None:

            ax.set_xlim((0, num_vert - 1))
            loc = ax.xaxis.get_major_locator()
            ind = loc().astype('int')
            ind = [i for i in ind if i >=0 and i < num_vert]

            nodes = list(range(0, graph.number_of_vertices()))

            ax.set_xticks(ind, [nodes[i] for i in ind])
        else:
            ax.set_xlim((0, num_vert - 1))
            loc = ax.xaxis.get_major_locator()
            ind = loc().astype('int')
            ind = [i for i in ind if i >=0 and i < num_vert]

            ax.set_xticks(ind)

    if min_prob is not None and max_prob is not None:
        # plt.ylim((min_prob, max_prob*1.02))
        _rescale_axis(ax, min_prob, max_prob)

    if time is not None:
        plt.title('Time: ' + str(time), loc='right')

def _rescale_axis(ax, min_prob, max_prob):
    ax.set_ylim((min_prob, max_prob*1.02))


def _plot_probability_distribution_on_graph(probabilities, ax, time=None,
                                            **kwargs):
    """
    Draw graph and illustrates the probabilities depending on the
    volatile parameters.

    See Also
    --------
    _update_nodes : sets volatile parameters
    :obj:`networkx.draw <networkx.drawing.nx_pylab.draw>`
    _configure_colorbar
    """
    cbar = kwargs.pop('cbar') if 'cbar' in kwargs else None
    # UpdateNodes may create kwargs['node_size']
    # min_node_size and max_node_size are not valid keys
    # for nx.draw kwargs
    _update_nodes(probabilities, kwargs.pop('min_node_size'),
                  kwargs.pop('max_node_size'), kwargs)

    vmin = kwargs.pop('min_prob')
    vmax = kwargs.pop('max_prob')
    ax.clear()
    nx.draw(kwargs.pop('graph'), ax=ax,
            node_size=kwargs.pop('node_size'),
            vmin=vmin, vmax=vmax, **kwargs)
    # Note: nx.draw_networkx_labels dramatically increases plotting time.
    # It is called by nx.draw
    kwargs['min_prob'] = vmin
    kwargs['max_prob'] = vmax

    # setting and drawing colorbar
    if 'cmap' in kwargs:
        cbar = _configure_colorbar(ax, cbar, kwargs)

    if time is not None:
        try:
            time = next(time)
        except TypeError:
            pass

        ax.set_title('Time: ' + str(time), loc='right')

    return [ax, cbar]


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
            kwargs['cmap'] = 'viridis'

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
        if ('rescale' in kwargs and kwargs.pop('rescale')):
            kwargs['min_prob'] = 0
            kwargs['max_prob'] = probabilities.max()

        # calculating size of each node acording to probability 
        # as a function f(x) = ax + b where b = min_size and
        # max_size = a*(max_prob-min_prob) + min_size
        a = ((max_node_size - min_node_size)
             / (kwargs['max_prob'] - kwargs['min_prob']))
        kwargs['node_size'] = list(map(
            lambda x: a*x + min_node_size, probabilities
        ))


def _configure_colorbar(ax, cbar, kwargs):
    """
    Add a colorbar in the figure besides the given ax

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Ax on which the plot was drawn.
    kwargs : dict
        Dictionary containing the keys 'cmap', 'min_prob' and 'max_prob'.
        'min_prob' and 'max_prob' describe
        the inferior and superior limit for colorbar values, respectively.
        'cmap' describes a valid matplotlib colormap
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    sm = plt.cm.ScalarMappable(
        cmap=kwargs['cmap'],
        norm=plt.Normalize(vmin=kwargs['min_prob'],
                           vmax=kwargs['max_prob'])
    )

    if cbar is None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2.5%', pad=0.01)
        cbar = plt.colorbar(
            sm,
            ticks=np.linspace(kwargs['min_prob'],
                              kwargs['max_prob'],
                              num=5),
            cax=cax
        )
        cbar.ax.tick_params(labelsize=14, length=7)
    else:
        cbar.update_normal(sm)

    return cbar

def _default_plane_kwargs(kwargs):
    if not 'cmap' in kwargs:
        kwargs['cmap'] = 'default'

    if 'cmap' in kwargs:
        if kwargs['cmap'] == 'default':
            kwargs['cmap'] = 'viridis'

    if 'linewidth' not in kwargs:
        kwargs['linewidth'] = 0
    if 'antialiased' not in kwargs:
        kwargs['antialiased'] = False
    if 'cstride' not in kwargs:
        kwargs['cstride'] = 1
    if 'rstride' not in kwargs:
        kwargs['rstride'] = 1
    if 'alpha' not in kwargs:
        kwargs['alpha'] = 0.5

def _plot_probability_distribution_on_plane(
        probabilities, ax, surf=None, cbar=None, labels=None, graph=None,
        min_prob=None, max_prob=None, dimensions=None, time=None, **kwargs
    ):
    """
    Plots probability distribution on the plane.
    """
    x_dim, y_dim = dimensions

    X = np.arange(0, x_dim, 1)
    Y = np.arange(0, y_dim, 1)
    Y, X = np.meshgrid(Y, X)
    Z = np.reshape(probabilities, (x_dim, y_dim))

    _default_plane_kwargs(kwargs)

    cmap = kwargs.pop('cmap')
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(Z)
    if min_prob is not None and max_prob is not None:
        vmin = min_prob
        vmax = max_prob
    else: #rescale
        vmin = 0
        vmax = Z.max()
    mappable.set_clim(vmin, vmax)

    # division by 4 apparently normalize the colors
    if surf is None:
        surf = [0]
    else:
        surf[0].remove()

    surf[0] = ax.plot_surface(X, Y, Z, cmap=mappable.cmap,
                           vmin=vmin/4, vmax=vmax/4,
                           **kwargs)
    ax.set_zlim(vmin, vmax)

    if time is not None:
        try:
            time = next(time)
        except TypeError:
            pass

        ax.set_title('Time: ' + str(time), loc='right')

    kwargs['cmap'] = cmap # reinserts into kwargs
    if cbar is None:
        cbar = plt.colorbar(mappable,
                            shrink=0.4, aspect=20,
                            pad=0.15, ax=ax)
    else:
        cbar.update_normal(mappable)

    cbar.ax.tick_params(length=10, width=1, labelsize=16)

    return [[surf[0]], cbar]

def _is_in_notebook():
    ipython_shells = (
        'XPythonShell',        # jupyterlite-xeus-python
        'Interpreter',         # jupyterlite-pyodide-kernel
        'ZMQInteractiveShell', # ipykernel
    )
    try:
        shell = get_ipython().__class__.__name__
        if shell in ipython_shells:
            return True
        return False
    except:
        return False

##########################################################################

def plot_success_probability(range, probabilities, figsize=(10, 6),
                             dpi=100, time_step=1, **kwargs):
    r"""
    Plot the success probability as a function of the number of steps.

    Assumes that the probabilities have already been calculated.
    
    Parameters
    ----------
    range: tuple of int
        Range used for the quantum walk simulation.
        See :meth:`QuantumWalk.simulate` for details.

    probabilities:
        Success probabilities with respect to ``time``,
        such that ``probabilities[i]`` corresponds to ``i``-th
        timestamp described by ``time``.

    figsize : tuple, default=(10, 6)
        Figure size in inches. Must be a tuple in the format (WIDTH, HEIGHT).

    dpi : float, default=100
        Figure resolution in dots-per-inch.

    time_step: float, default=1
        Time interval between simulation steps.

    **kwargs:
        Additional arguments to customize plot.
        See :obj:`matplotlib.pyplot.plot` for the optional keywords.

    See Also
    --------
    QuantumWalk.simulate
    QuantumWalk.success_probability
    matplotlib.pyplot.plot
    """

    # Figure size

    if len(figsize) == 2:
        fig_width, fig_height = figsize
    else:
        raise ValueError(
            'figsize must be a tuple in the format (WIDTH, HEIGHT)'
        )

    _configure_figure(fig_width, fig_height, dpi)

    range = QuantumWalk._range_to_tuple(range)
    range = time_step*np.arange(*range)


    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Success probability', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    ax = plt.gca()
    ax.set_ylim(0, 1.05*max(probabilities))

    plt.plot(range, probabilities, **kwargs)
    plt.show()
