======
Graphs
======

It is possible to create arbitrary graphs or specific graphs.

----------------
Arbitrary Graphs
----------------

In the meantime, Hiperwalk only deals with simple graphs.
Any graph be created by passing the adjacency matrix to
the :class:`hiperwalk.Graph` constructor.

The adjacency matrix can be obtained via `NetworkX
<https://networkx.org/>`_.
For example, the adjacency matrix of a ladder graph can be obtained
by executing

>>> import networkx as nx
>>> ladder = nx.ladder_graph(10)
>>> adj = nx.adjacency_matrix(ladder)
>>> adj #doctest: +SKIP
<20x20 sparse array of type '<class 'numpy.int64'>'
    with 56 stored elements in Compressed Sparse Row format>


We can then create a Hiperwalk graph by passing the adjacency matrix.

.. testsetup::

   from sys import path as sys_path
   sys_path.append("../..")

>>> import hiperwalk as hpw
>>> hpw_ladder = hpw.Graph(adj)
>>> hpw_ladder #doctest: +SKIP
<hiperwalk.graph.graph.Graph object at 0x7f3bd0627eb0>

After doing this it is recommended do delete the NetworkX graph.
Specially when dealing with large graphs to avoid memory overload.

>>> del ladder

We then may use ``hpw_ladder`` to simulate any quantum walk.

.. todo::
   In the meantime,
   we only accept the adjacency matrix for creating a graph.
   In the future,
   we will accept NetworkX graphs.
   Although the process of deleting the NetworkX is still recommended.

---------------
Specific Graphs
---------------

There are classes for specific well-known graphs,
such as the Line, Cycle and Lattice.
Please refer to :ref:`docs_documentation_graph`
for the complete list of specific graphs.

Generally, the difference between creating an arbitrary graph and
a specific graph is that
in the former the adjacency matrix must be explicited,
while in the latter,
the adjacency matrix is generated from the number of vertices.
Thus, the following are valid commands for creating
a line with 10 vertices,
a cycle with 10 vertices,
and a :math:`10 \times 10`-dimensional lattice, respectively.

>>> hpw.Line(10) #doctest: +SKIP
<hiperwalk.graph.line.Line object at 0x7f0a6bb700d0>
>>> hpw.Cycle(10) #doctest: +SKIP
<hiperwalk.graph.cycle.Cycle object at 0x7f0a6bbb32e0>
>>> hpw.Lattice(10) #doctest: +SKIP
<hiperwalk.graph.lattice.Lattice object at 0x7f0a6bbb2da0>

Naturally, the specific graphs may be created using NetworkX and
using the arbitrary :class:`hiperwalk.Graph` class.
However, this is not recommended specifically when dealing with
the Coined Quantum Walk model
because it may result on undesirable behavior.

For example, let us take a Line with 10 vertices as example.
First, create the line using the :class:`hiperwalk.Graph` class.

>>> path = nx.path_graph(10)
>>> adj_matrix = nx.adjacency_matrix(path)
>>> arbitrary_line = hpw.Graph(adj_matrix)

Then create the line using the :class:`hiperwalk.Line` class.

>>> specific_line = hpw.Line(10)

We associate each edge of the simple graph with two arcs.
Let's say that we which to know the label of the arc
that goes from vertex 1 to 2.
This can be done using the :meth:`hiperwalk.Graph.arc_label`.
We have

>>> arbitrary_line.arc_label(1, 2)
2
>>> specific_line.arc_label(1, 2)
1

For more details on the arc labels on each graph,
refer to the Notes section of each graph class --
in this case, :class:`hiperwalk.Graph` and :class:`hiperwalk.Line`.
