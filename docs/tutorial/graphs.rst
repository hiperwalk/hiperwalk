======
Graphs
======

Hiperwalk gives support to three different types of graphs:
simples graphs, multigraphs and weighted graphs.
There is a class for each of these types of graph:
:class:`hiperwalk.Graph`,
:class:`hiperwalk.Multigraph`, and
:class:`hiperwalk.WeightedGraph`, respectively.

-------------
Simple Graphs
-------------

The :class:`hiperwalk.Graph` class is used for
handling simple graphs with self-loops.

You can create any graph by passing the adjacency matrix to
the :class:`hiperwalk.Graph` constructor.
You can obtain the adjacency matrix using `NetworkX
<https://networkx.org/>`_'s
:func:`networkx.adjacency_matrix` function.
For instance, to generate the adjacency matrix of a ladder graph,
you would execute the following commands:

>>> import networkx as nx
>>> nx_ladder = nx.ladder_graph(10)
>>> adj = nx.adjacency_matrix(nx_ladder)
>>> adj # doctest: +SKIP
<20x20 sparse array of type '<class 'numpy.int64'>'
    with 56 stored elements in Compressed Sparse Row format>

Once you have the adjacency matrix, you can then create a
Hiperwalk graph.

.. testsetup::

   from sys import path as sys_path
   sys_path.append("../..")

>>> import hiperwalk as hpw
>>> hpw_ladder = hpw.Graph(adj)
>>> hpw_ladder #doctest: +SKIP
<hiperwalk.graph.graph.Graph object at 0x7f3bd0627eb0>

Alternativaly, you can create a Hiperwalk graph by
passing an instance of :class:`networkx.Graph`.

>>> hpw_ladder2 = hpw.Graph(nx_ladder)
>>> hpw_ladder2 #doctest: +SKIP
<hiperwalk.graph.graph.Graph object at 0x74f25e41d4e0>
>>> # checking if the adjacency matrices are equal
>>> nx_adj = nx.adjacency_matrix(nx_ladder)
>>> hpw_adj1 = hpw_ladder.adjacency_matrix()
>>> hpw_adj2 = hpw_ladder2.adjacency_matrix()
>>> (nx_adj - hpw_adj1).nnz == 0
True
>>> (hpw_adj1 - hpw_adj2).nnz == 0
True

After instantiating a Hiperwalk graph,
it's advisable to delete the NetworkX graph.
This step is particularly important when dealing with large graphs,
as it helps to prevent memory overload.

>>> del nx_ladder

Following this, you can use ``hpw_ladder`` to simulate any quantum walk,
regardless of the model.

Order of Neighbors
------------------

-----------
Multigraphs
-----------

You can create a multigraph by passing its adjacency matrix.
The entries the adjacency matrix entries are the number of edges
simultaneously incident to pairs of vertices.

.. testsetup::

   import numpy as np

>>> # creating the adjacency matrix of a complete multigraph
>>> num_vert = 5
>>> adj_matrix = np.zeros((num_vert, num_vert))
>>> for i in range(num_vert):
...     for j in range(num_vert):
...         adj_matrix[i, j] = i + j
...
>>> # creating multigraph
>>> g = hpw.Multigraph(adj_matrix)
>>> # checking if multigraph was created properly
>>> np.all(np.array(
...         [g.number_of_edges(u, v) == u + v
...         for u in range(num_vert)
...         for v in range(num_vert)]
...       ) == True)
True

---------------
Weighted Graphs
---------------
