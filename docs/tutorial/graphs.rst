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

By default, the neighbors of a vertex are listed in ascending order.

>>> hpw_ladder.neighbors(1)
array([ 0,  2, 11])

Depending on the graph,
you may wish to specify a different order of neighbors.
This is particularly useful for the coined quantum walk model
(:class:`hiperwalk.Coined`).
To specify an order of neighbors,
create the adjacency matrix using :class:`scipy.sparse.csr_array`
such that :obj:`scipy.sparse.csr_array.has_sorted_indices` is ``False``.
This can be done by expliciting
:class:`scipy.sparse.csr_array`'s ``indices``.
Then, the neighbors of ``u`` are listed in the order
``indices[indptr[u]:indptr[u+1]]``.
For example,
the following commands create a
complete graph with self-loops where the neighbors of ``u`` are
listed in the order ``[u, u+1, ..., u-1]``.

>>> import scipy.sparse
>>> # creating the csr_array
>>> num_vert = 5
>>> data = np.ones(num_vert**2)
>>> indices = [(u + shift) % num_vert
...            for u in range(num_vert)
...            for shift in range(num_vert)]
>>> indptr = np.arange(0, num_vert**2 + 1, num_vert)
>>> adj_matrix = scipy.sparse.csr_array((data, indices, indptr))
>>> # creating graph with non-default order of neighbors
>>> g = hpw.Graph(adj_matrix)
>>> # testing the order of neighbors
>>> for u in range(num_vert):
...     print(g.neighbors(u))
[0 1 2 3 4]
[1 2 3 4 0]
[2 3 4 0 1]
[3 4 0 1 2]
[4 0 1 2 3]

-----------
Multigraphs
-----------

You can create a multigraph by passing its adjacency matrix.
The adjacency matrix entries are the number of edges
simultaneously incident to pairs of vertices.

.. testsetup::

   import numpy as np

>>> # creating the adjacency matrix of a complete multigraph
>>> num_vert = 5
>>> adj_matrix = np.zeros((num_vert, num_vert))
>>> for i in range(num_vert):
...     for j in range(num_vert):
...         adj_matrix[i, j] = i + j + 1
...
>>> # creating multigraph
>>> g = hpw.Multigraph(adj_matrix)
>>> # checking if multigraph was created properly
>>> np.all(np.array(
...         [g.number_of_edges(u, v) == u + v + 1
...         for u in range(num_vert)
...         for v in range(num_vert)]) == True)
True

Multigraphs are used to create coined quantum walks
(:class:`hiperwalk.Coined`).

Order of Neighbors
------------------

To specify a non-default order of neighbors,
you can follow the same steps used for simple graphs.
But be aware that the values of ``data`` must be rearranged
with respect to ``indices``.

>>> # creating data with the appropriate values
>>> data = [u + v + 1 for u in range(num_vert)
...         for v in indices[indptr[u]:indptr[u+1]]]
>>> adj_matrix = scipy.sparse.csr_array((data, indices, indptr))
>>> # create multigraph with desired order of neighbors
>>> g = hpw.Multigraph(adj_matrix)
>>> for u in range(num_vert):
...     print(g.neighbors(u))
[0 1 2 3 4]
[1 2 3 4 0]
[2 3 4 0 1]
[3 4 0 1 2]
[4 0 1 2 3]
>>> # checking if multigraph was created properly
>>> np.all(np.array(
...         [g.number_of_edges(u, v) == u + v + 1
...         for u in range(num_vert)
...         for v in range(num_vert)]) == True)
True


---------------
Weighted Graphs
---------------

You can create a weighted graph by passing its adjacency matrix.
The adjacency matrix entries are real values that
represent the edge weights.

.. testsetup::

   import numpy as np

>>> # creating the adjacency matrix of a complete weighted graph
>>> num_vert = 5
>>> adj_matrix = np.zeros((num_vert, num_vert))
>>> for i in range(num_vert):
...     for j in range(num_vert):
...         adj_matrix[i, j] = (i + j)/10
...
>>> # creating weighted
>>> g = hpw.WeightedGraph(adj_matrix)
>>> adj_matrix2 = g.adjacency_matrix().todense()
>>> np.all(np.isclose(adj_matrix, adj_matrix2))
True

Weighted graphs are used to create continuous-time quantum walks
(:class:`hiperwalk.ContinuousTime`).


Order of Neighbors
------------------
The order of neighbors does not play a crucial role in
the continuous-time quantum walk.
Even so,
you can specify the order of neighbors in weighted graphs
using commands akin to those used for multigraphs.
