===============
``CoinedModel``
===============

.. currentmodule:: CoinedModel

This module contains functions for the Coined Quantum Walk Model.

Functions for managing, simulating and generating matrices of
the coined quantum walk model for general and
specific graphs are available.

For implementation details, see the `Notes`_ Section.


Functions
---------
.. autosummary::
    :toctree: generated
    :nosignatures:

    CoinOperator
    EvolutionOperator
    FlipFlopShiftOperator
    GroverOperator
    HadamardOperator
    OracleR
    SearchEvolutionOperator
    UniformInitialCondition

.. _CoinedModel Notes:

Notes
-----
For more information about the general general Coined Quantum Walk Model,
check Quantum Walks and Search Algorithms's
Section 7.2: Coined Walks on Arbitrary Graphs [1]_.

The `CoinedModel`_ module uses the position-coin notation
and the Hilbert space :math:`\mathcal{H}^{2|E|}` for general Graphs.
Matrices and states respect the sorted edges order,
i.e. :math:`(v, u) < (v', u')` if either :math:`v < v'` or
:math:`v = v'` and :math:`u < u'`
where :math:`(v, u), (v', u')` are valid edges.

For example, the graph :math:`G(V, E)` shown in Figure 1 has adjacency matrix `A`.

>>> import numpy as np
>>> A = np.matrix([[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]])
>>> A
matrix([[0, 1, 0, 0],
        [1, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 1, 1, 0]])


.. graphviz:: ../graphviz/coined-model-sample.dot
    :align: center
    :layout: neato
    :caption: Figure 1

Letting :math:`(v, u)` denote the edge from vertex :math:`v` to :math:`u`,
the `edges` of :math:`G` are

>>> edges = [(i, j) for i in range(4) for j in range(4) if A[i,j] == 1]
>>> edges
[(0, 1), (1, 0), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]

Note that `edges` is already sorted, hence the labels are

>>> edges_labels = {edges[i]: i for i in range(len(edges))}
>>> edges_labels
{(0, 1): 0, (1, 0): 1, (1, 2): 2, (1, 3): 3, (2, 1): 4, (2, 3): 5, (3, 1): 6, (3, 2): 7}

The edges labels are illustrated in Figure 2.

.. graphviz:: ../graphviz/coined-model-edges-labels.dot
    :align: center
    :layout: neato
    :caption: Figure 2

If we would write the edges labels respecting the the adjacency matrix format,
we would have the matrix `A_labels`.
Intuitively, the edges are labeled in left-to-right top-to-bottom fashion.

>>> A_labels = [[edges_labels[(i,j)] if (i,j) in edges_labels else '' for j in range(4)]
...             for i in range(4)]
>>> A_labels = np.matrix(A_labels)
>>> A_labels
matrix([['', '0', '', ''],
        ['1', '', '2', '3'],
        ['', '4', '', '5'],
        ['', '6', '7', '']], dtype='<U21')

For consistency, any state :math:`\ket\psi \in \mathcal{H}^{2|E|}`
is such that :math:`\ket\psi = \sum_{i = 0}^{2|E| - 1} \psi_i \ket{i}`
where :math:`\ket{i}` is the computational basis state
associated to the :math:`i`-th edge.
In our example, the state

>>> psi = np.matrix([1/np.sqrt(2), 0, 1j/np.sqrt(2), 0, 0, 0, 0, 0]).T
>>> psi
matrix([[0.70710678+0.j        ],
        [0.        +0.j        ],
        [0.        +0.70710678j],
        [0.        +0.j        ],
        [0.        +0.j        ],
        [0.        +0.j        ],
        [0.        +0.j        ],
        [0.        +0.j        ]])

corresponds to the walker being at vertex 0
and the coin pointing to vertex 1 with
associated amplitude of :math:`\frac{1}{\sqrt 2}`, and
to the walker being at vertex 1
and the coin pointing to vertex 2 with
associated amplitude of :math:`\frac{\text{i}}{\sqrt 2}`.


References
----------
.. [1] Portugal, Renato. "Quantum walks and search algorithms".
    Vol. 19. New York: Springer, 2013.
