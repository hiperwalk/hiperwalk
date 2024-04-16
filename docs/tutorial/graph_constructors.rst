==================
Graph Constructors
==================

Hiperwalk has specific commands (graph constructors) to
generate instances of well-known specific graphs,
such as Line, Cycle, and Grid.
For a complete list of graph constructors,
refer to :ref:`docs_documentation_list_of_graph_constructors`.

The key difference between creating an
arbitrary graph and using a graph constructor is
that for the former, you must explicitly provide the adjacency matrix.
However, for the latter, the adjacency matrix is automatically generated
based on specific graph parameters such as the number of vertices.
Nevertheless,
the order of neighbors is automatically embedded in
the generated adjacency matrix.

For example, you can create a line with 10 vertices,
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
Just be aware that the order of neighbors may vary,
which may result in undesirable behavior,
especially when dealing with coined quantum walks.

As an example, let's consider a Line with 10 vertices.
First, we create the line using the :class:`hiperwalk.Graph` class.

.. testsetup::

   import networkx as nx
   import hiperwalk as hpw

>>> path = nx.path_graph(10)
>>> adj_matrix = nx.adjacency_matrix(path)
>>> arbitrary_line = hpw.Graph(adj_matrix)

Next, we create the line using the :class:`hiperwalk.Line`
graph constructor.

>>> graph_constructor_line = hpw.Line(10)

The order of neighbors os these two instances is different.
For ``arbitrary_line``,
the neighbors are listed in ascending order,
while for ``graph_constructor_line``,
the neighbors are listed in descending order
(righmost neighbor followed by the leftmost neighbor).

>>> arbitrary_line.neighbors(1)
array([0, 2])
>>> graph_constructor_line.neighbors(1)
array([2, 0], dtype=int32)
