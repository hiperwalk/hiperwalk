#TODO: submit pull request to networkx and remove this code

#This module includes the return type of some NetworkX's functions.
#The returned values are particularly useful for animations using matplotlib's FuncAnimation.

import matplotlib.pyplot as plt
import networkx as nx

#this is a redefinition of networkx's draw method
#https://networkx.org/documentation/stable/_modules/networkx/drawing/nx_pylab.html#draw.
#The difference is that the redefinition returns the collections of nodes, lines and
#dictionary of labels.
#More specifically, it saves and returns the values from the reimplementation of draw_networkx,
def nx_draw(G, pos=None, ax=None, **kwds):

    if ax is None:
        cf = plt.gcf()
    else:
        cf = ax.get_figure()
    cf.set_facecolor("w")
    if ax is None:
        if cf._axstack() is None:
            ax = cf.add_axes((0, 0, 1, 1))
        else:
            ax = cf.gca()

    if "with_labels" not in kwds:
        kwds["with_labels"] = "labels" in kwds

    ret = nx_draw_networkx(G, pos=pos, ax=ax, **kwds)
    ax.set_axis_off()
    plt.draw_if_interactive()
    return ret


#this is a redefinition of networkx's draw_networkx method
#https://networkx.org/documentation/stable/_modules/networkx/drawing/nx_pylab.html#draw_networkx.
#The difference is that the redefinition returns the collections of nodes, lines and
#dictionary of labels.
#More specifically, it saves and returns the values from the original implementations of
#draw_networkx_nodes, draw_networkx_edges, draw_networkx_labels 
def nx_draw_networkx(G, pos=None, arrows=None, with_labels=True, **kwds):

    valid_node_kwds = (
        "nodelist",
        "node_size",
        "node_color",
        "node_shape",
        "alpha",
        "cmap",
        "vmin",
        "vmax",
        "ax",
        "linewidths",
        "edgecolors",
        "label",
    )

    valid_edge_kwds = (
        "edgelist",
        "width",
        "edge_color",
        "style",
        "alpha",
        "arrowstyle",
        "arrowsize",
        "edge_cmap",
        "edge_vmin",
        "edge_vmax",
        "ax",
        "label",
        "node_size",
        "nodelist",
        "node_shape",
        "connectionstyle",
        "min_source_margin",
        "min_target_margin",
    )

    valid_label_kwds = (
        "labels",
        "font_size",
        "font_color",
        "font_family",
        "font_weight",
        "alpha",
        "bbox",
        "ax",
        "horizontalalignment",
        "verticalalignment",
    )

    valid_kwds = valid_node_kwds + valid_edge_kwds + valid_label_kwds

    if any([k not in valid_kwds for k in kwds]):
        invalid_args = ", ".join([k for k in kwds if k not in valid_kwds])
        raise ValueError(f"Received invalid argument(s): {invalid_args}")

    node_kwds = {k: v for k, v in kwds.items() if k in valid_node_kwds}
    edge_kwds = {k: v for k, v in kwds.items() if k in valid_edge_kwds}
    label_kwds = {k: v for k, v in kwds.items() if k in valid_label_kwds}

    if pos is None:
        #TODO: make simple PR removing the "nx.drawing" because it is already
        #imported in the beginning of the networkx.drawing.nx_pylab file
        pos = nx.drawing.spring_layout(G)  # default to spring layout

    nodes = nx.draw_networkx_nodes(G, pos, **node_kwds)
    edges = nx.draw_networkx_edges(G, pos, arrows=arrows, **edge_kwds)
    labels = None
    if with_labels:
        labels = nx.draw_networkx_labels(G, pos, **label_kwds)
    plt.draw_if_interactive()

    return nodes, edges, labels
