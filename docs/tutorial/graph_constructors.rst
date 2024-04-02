==================
Graph Constructors
==================

You can use graph constructors to create instances of
specific, well-known graphs.

Specific Graphs
---------------

Hiperwalk includes classes for well-known specific graphs,
such as Line, Cycle, and Grid.
For a complete list of specific graphs,
refer to :ref:`docs_documentation_graph`.

The key difference between creating an
arbitrary graph and a specific graph is
that for the former, you must explicitly provide the adjacency matrix.
However, for the latter, the adjacency matrix is automatically generated
based on the number of vertices.
Consequently, you can create a line with 10 vertices,
a cycle with 10 vertices,
and a grid of dimensions :math:`10 \times 10` using
valid commands, respectively.

>>> hpw.Line(10) #doctest: +SKIP
<hiperwalk.graph.line.Line object at 0x7f0a6bb700d0>
>>> hpw.Cycle(10) #doctest: +SKIP
<hiperwalk.graph.cycle.Cycle object at 0x7f0a6bbb32e0>
>>> hpw.Grid(10) #doctest: +SKIP
<hiperwalk.graph.grid.Grid object at 0x7f0a6bbb2da0>

Naturally, you can create specific graphs using NetworkX and
the :class:`hiperwalk.Graph` class.
However, this approach is not recommended, especially when dealing with
coined quantum walks, as it may result in undesirable behavior.

As an example, let's consider a Line with 10 vertices.
First, we create the line using the :class:`hiperwalk.Graph` class.

>>> path = nx.path_graph(10)
>>> adj_matrix = nx.adjacency_matrix(path)
>>> arbitrary_line = hpw.Graph(adj_matrix)

Next, we create the line using the :class:`hiperwalk.Line` class.

>>> specific_line = hpw.Line(10)

In a simple graph, we associate each edge with two arcs. Suppose we wish
to know the label of the arc that links vertex 1 to 2. This information
can be obtained using the :meth:`hiperwalk.Graph.arc_number` method.
Observe the following results:

For further details on arc labels for each graph, refer to the Notes
section of each graph class, in this case,
:class:`hiperwalk.Graph` and :class:`hiperwalk.Line`.
