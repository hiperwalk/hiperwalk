.. _docs_documentation_graph:

======
Graphs
======

.. currentmodule:: hiperwalk

Classes
-------

There are three classes of graph.

.. autosummary::
    :toctree: generated
    :nosignatures:

    Graph
    Multigraph
    WeightedGraph

:class:`hiperwalk.Multigraph` is used as input for the
:class:`hiperwalk.Coined` quantum walk model.
:class:`hiperwalk.WeightedGraph` is used as input for the
:class:`hiperwalk.ContinuousTime` quantum walk model.
:class:`hiperwalk.Graph` is used as input for
both quantum walk models.

.. _graph_constructors:

Graph Constructors
------------------

The graph constructors [#fn1]_
return an instance of commonly used graphs.

The graph constructors are used to change the default behavior of
specific graphs.
For instance, the order of the neighbors changes depending on
the graph.

Every constructor has the following template:

>>> GraphConstructor(args, multiedges=None, weights=None) #doctest: +SKIP

where ``multiedges`` and ``weights`` are optional arguments
whose default value is ``None``.
Every graph constructor returns an instance of

* :class:`hiperwalk.Graph` if ``multiedges is None``
  and ``weights is None``.
* :class:`hiperwalk.Multigraph` if ``multiedges is not None``
  and ``weights is None``.
* :class:`hiperwalk.WeightedGraph` if ``multiedges is None``
  and ``weights is not None``.

Both ``multiedges`` and ``weights`` are expected to be
instances of :class:`scipy.sparse.csr_array` such that

* ``multiedges[u, v]`` is the number of edges
  incident to both vertices ``u`` and ``v``.
* ``weights[u, v]`` is the weight of the edge
  incident to both vertices ``u`` and ``v``.

If ``multiedges is not None`` and ``weights is not None``,
a ValueError exception is raised.

The following is a list of the available graph constructors.

.. toctree::
   :maxdepth: 1

   graph_constructors/hypercube

.. [#fn1] NetworkX uses `graph generators
   <https://networkx.org/documentation/stable/reference/generators.html>`_
   for the same concept.
   However, both `constructor
   <https://docs.python.org/3/reference/datamodel.html#object.__init__>`_
   and `generator
   <https://docs.python.org/3/glossary.html#term-generator>`_
   are words that already have consolidated meanings.
   Thus, we decided to use the word *constructor* because
   it conveys the idea that an instance is going to be returned.
