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
instances of :class:`dict` or :class:`scipy.sparse.csr_array`.

* If they are an instance of :class:`dict`
  in the ``{(u, v): value}`` format.

  * ``multiedges[(u, v)]`` is the number of edges
    incident to both vertices ``u`` and ``v``.

  * ``weights[(u, v)]`` is the weight of the edge
    incident to both vertices ``u`` and ``v``.

  If an edge exists in the graph but it is not listed in
  the :class:`dict` keys,
  its value defaults to 1.
  Attempting to remove an edge by assigning its value to 0
  (e.g. ``weights[(u, v)] = 0``) or
  attempting to add an edge by assigning a value to it
  (e.g. ``weights[(-1, -1) = 10])`` raises a
  a `ValueError exception
  <https://docs.python.org/3/library/exceptions.html#ValueError>`_.

* If they are an instance of :class:`scipy.sparse.csr_array`,

  * ``multiedges[u, v]`` is the number of edges
    incident to both vertices ``u`` and ``v``.

  * ``weights[u, v]`` is the weight of the edge
    incident to both vertices ``u`` and ``v``.

  In this case, all valid edges must have been assigned a
  value different from 0.
  When explicitly specifying the adjacency matrix,
  the ``copy`` changes the method's behavior.

  * If ``copy = False`` (default), a pointer to the
    adjacency matrix is stored.

  * If ``copy = True``, a hard copy of the
    adjacency matrix is stored.

If ``multiedges is not None`` and ``weights is not None``,
a `ValueError exception
<https://docs.python.org/3/library/exceptions.html#ValueError>`_
is raised.

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

.. _docs_documentation_list_of_graph_constructors:

List of Graph Constructors
**************************

The following is the list of all available graph constructors.

.. toctree::
   :maxdepth: 1

   graph_constructors/complete
   graph_constructors/complete_bipartite
   graph_constructors/cycle
   graph_constructors/grid
   graph_constructors/hypercube
   graph_constructors/integer_lattice
   graph_constructors/line
