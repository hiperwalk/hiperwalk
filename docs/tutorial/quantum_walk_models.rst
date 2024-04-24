Quantum Walk Models
===================

There are numerous known quantum walk models. Hiperwalk offers a
unified interface for all quantum walks via the abstract
class :class:`hiperwalk.QuantumWalk`. This class cannot be instantiated
directly, but it can be inherited from. All its methods and attributes
will be available to the child class. However, the abstract methods must
be overridden by the child class as they are model-dependent.

Currently, two models are available: the Coined model
(:class:`hiperwalk.Coined`) and the Continuous model
(:class:`hiperwalk.ContinuousTime`). Users are encouraged
to implement new models and add them to the Hiperwalk
package (see the :ref:`docs_development` section
for more information).

Creating a Quantum Walk
-----------------------

.. testsetup::

   from sys import path as sys_path
   sys_path.append("../..")
   import numpy as np
   import hiperwalk as hpw

For creating a quantum walk,
we must first define the graph in which the quantum walk will
take place. This can be accomplished by passing a :class:`hiperwalk.Graph`
object to the Quantum Walk constructor.

For example, consider the cycle graph with 11 vertices.

>>> cycle = hpw.Cycle(11)
>>> cycle #doctest: +SKIP
<hiperwalk.graph.graph.Graph object at 0x7f657268c0d0>

Since ``cycle`` is an instance of :class:`hiperwalk.Graph`,
we can pass ``cycle`` to the quantum walk constructor.

Coined Model
''''''''''''
A coined quantum walk can be created by passing an instance of
:class:`hiperwalk.Graph` or :class:`hiperwalk.Multigraph`.
To create a coined quantum walk on the cycle, we execute

>>> coined = hpw.Coined(graph=cycle)
>>> coined #doctest: +SKIP
<hiperwalk.quantum_walk.coined_walk.Coined object at 0x7f655b0cd900>

The Hilbert space of the coined quantum walk has dimension
:math:`2|E|`, i.e. the number of arcs.

>>> coined.hilbert_space_dimension() == 2*cycle.number_of_edges()
True

Continuous-time Model
'''''''''''''''''''''
A coined quantum walk can be created by passing an instance of
:class:`hiperwalk.Graph` or :class:`hiperwalk.WeightedGraph`.
To create a continuous-time quantum walk on the cycle,
we execute an analogous command.

>>> continuous = hpw.ContinuousTime(graph=cycle)
>>> continuous #doctest: +SKIP
<hiperwalk.quantum_walk.continuous_time.ContinuousTime object at 0x7098fe80eef0>

The Hilbert space of the continuous-time quantum walk has dimension
:math:`|V|`, i.e. the number of vertices.

>>> continuous.hilbert_space_dimension() == cycle.number_of_vertices()
True

Creating a State
----------------

Hiperwalk offers three easy ways of creating a state.
The user can create a state of the computational basis
(:meth:`hiperwalk.QuantumWalk.ket`),
a uniform superposition (:meth:`hiperwalk.QuantumWalk.uniform_state`,
or an arbitrary superposition (:meth:`hiperwalk.QuantumWalk.state`).

State of the computational basis
''''''''''''''''''''''''''''''''
Any state of the computational basis can be created using the
:meth:`hiperwalk.QuantumWalk.ket` method
as long as the correct label is passed.

Coined Model
````````````
In the coined quantum walk model,
the label of a state within the computational basis corresponds
to an arc. You can use either the arc notation, which involves
specifying the arc's tail and head,
or the arc number (an integer).
Please refer to the :class:`hiperwalk.Graph` class for
correct arc labeling guidelines,
as the arc number varies according to the order of neighbors.

>>> state = coined.ket((5, 6))
>>> state
array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0.])
>>> state2 = coined.ket(10)
>>> np.all(state == state2)
True

Continuous-time Model
`````````````````````
In the continuous-time model,
the labels correspond directly to the labels of the vertices.

>>> continuous.ket(5)
array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])

Uniform superposition
---------------------

To create a uniform superposition,
you can use the :meth:`hiperwalk.QuantumWalk.uniform_state` method
which is applicable to any model.

>>> coined.uniform_state()
array([0.21320072, 0.21320072, 0.21320072, 0.21320072, 0.21320072,
       0.21320072, 0.21320072, 0.21320072, 0.21320072, 0.21320072,
       0.21320072, 0.21320072, 0.21320072, 0.21320072, 0.21320072,
       0.21320072, 0.21320072, 0.21320072, 0.21320072, 0.21320072,
       0.21320072, 0.21320072])
>>> continuous.uniform_state()
array([0.30151134, 0.30151134, 0.30151134, 0.30151134, 0.30151134,
       0.30151134, 0.30151134, 0.30151134, 0.30151134, 0.30151134,
       0.30151134])


Arbitrary state
---------------
Creating a generic state with the :meth:`hiperwalk.QuantumWalk.state`
method can be a bit challenging. It expects a list consisting
of ``[amplitude, label]`` entries, where each entry represents an amplitude
and a label of the computational basis.

Since :meth:`hiperwalk.QuantumWalk.state` must return a valid state,
the amplitudes are renormalized when needed.

Coined Model
''''''''''''
In the coined model,
the labels of the computational basis are represented by
either numbers or arcs (i.e. ``(tail, head)``).
An example using numeric labels is

>>> coined.state([[0.5, 0],
...               [0.5, 2],
...               [0.5, 4],
...               [0.5, 6]])
array([0.5, 0. , 0.5, 0. , 0.5, 0. , 0.5, 0. , 0. , 0. , 0. , 0. , 0. ,
       0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ])

and using the equivalent arc notation is

>>> coined.state([[0.5, (0, 1)],
...               [0.5, (1, 2)],
...               [0.5, (2, 3)],
...               [0.5, (3, 4)]])
array([0.5, 0. , 0.5, 0. , 0.5, 0. , 0.5, 0. , 0. , 0. , 0. , 0. , 0. ,
       0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ])

.. note::
   Do not forget the parenthesis while using the arc notation
   for generating a state.

If we try to create a non-normalized state,
the amplitudes are renormalized.

>>> coined.state([[1, (0, 1)],
...               [1, (1, 2)],
...               [1, (2, 3)],
...               [1, (3, 4)]])
array([0.5, 0. , 0.5, 0. , 0.5, 0. , 0.5, 0. , 0. , 0. , 0. , 0. , 0. ,
       0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ])

Continuous-time Model
'''''''''''''''''''''
For the continuous-time model,
the labels of the computational basis correspond to
the labels of the vertices:

>>> continuous.state([[0.5, 0],
...                   [0.5, 1],
...                   [0.5, 2],
...                   [0.5, 3]])
array([0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0. ])

If we try to create a non-normalized state,
the amplitudes are renormalized.

>>> continuous.state([[1, 0],
...                   [1, 1],
...                   [1, 2],
...                   [1, 3]])
array([0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0. ])


Simulation of Propagation
-------------------------

Once a quantum walk is defined,
it is linked to an appropriate evolution operator.
The user has the flexibility to modify this operator either during
the quantum walk's creation or at any time afterward.
Once the evolution operator is set,
the user triggers the simulation process,
deciding which intermediate states are of particular interest.

Configuring the evolution operator
''''''''''''''''''''''''''''''''''
To set up the evolution operator, users should refer to
the :meth:`hiperwalk.QuantumWalk.set_evolution` method.
Note that the parameters for this method depend on the model being used.

Regardless of the method employed,
:meth:`hiperwalk.QuantumWalk.set_evolution` is invoked when
the quantum walk is instantiated.
Consequently, the  constructors accept any parameter that
is valid for the ``set_evolution`` method.
To illustrate this point,
let us examine the explicit evolution operator of two coined walks,
which can be derived using the
:meth:`hiperwalk.QuantumWalk.get_evolution` method.

>>> U = coined.get_evolution()
>>> coined.set_evolution(shift='flipflop', coin='grover')
>>> U2 = coined.get_evolution()
>>> (U != U2).nnz == 0 # efficient way of comparing sparse arrays
False
>>> coined2 = hpw.Coined(graph=cycle, shift='flipflop', coin='grover')
>>> U3 = coined.get_evolution()
>>> (U2 != U3).nnz == 0
True

Coined Model
````````````
The :meth:`hiperwalk.Coined.set_evolution` method
accepts three key arguments:
``shift``, ``coin``, and ``marked``,
which are the arguments of
:meth:`hiperwalk.Coined.set_shift`,
:meth:`hiperwalk.Coined.set_coin`, and
:meth:`hiperwalk.Coined.set_marked`,
respectively.

The ``shift`` key can either take a string value
(``'persistent'`` or ``'flipflop'``), or the explicit operator.

The ``coin`` key can accept four types of inputs:

* An explicit coin.
* A string specifying the coin name, which will be applied to all vertices.
* A list of strings of size equal to the number of vertices :math:`|V|`
  specifying the coin names where the :math:`i`-th coin will be applied to
  the :math:`i`-th vertex.
* A dictionary with the coin name as the key and
  a list of vertices as values.
  The coin referred to by the key will be applied to the vertices
  listed as its values
  If the list of vertices is empty ``[]``,
  that particular coin will be applied to all the remaining vertices.

There are eight possible coin names:
``'fourier'``, ``'grover'``, ``'hadamard'``, ``'identity'``, and
their respective variants prefixed with ``'minus_'``.

The following are examples of how you could generate a coin that applies
the Grover operator to all vertices.

>>> coined.set_coin(coin='grover')
>>> C1 = coined.get_coin()
>>> coined.set_coin(coin=['grover'] * 11)
>>> C2 = coined.get_coin()
>>> coined.set_coin(coin={'grover' : list(range(11))})
>>> C3 = coined.get_coin()
>>> (C1 != C2).nnz == 0
True
>>> (C2 != C3).nnz == 0
True

The following are valid ways of generating a coin that applies
Grover to even vertices and Hadamard to odd vertices.

>>> coined.set_coin(coin=['grover' if i % 2 == 0 else 'hadamard'
...                       for i in range(11)])
>>> C1 = coined.get_coin()
>>> coined.set_coin(coin={'grover': list(range(0, 11, 2)),
...                       'hadamard': []})
>>> C2 = coined.get_coin()
>>> (C1 != C2).nnz == 0
True

The ``marked`` key can accept two types of inputs:

* A list of the marked vertices: In this case,
  the vertices are simply set as marked,
  but the coin operator remains unchanged.
* A dictionary with the coin name as key and
  the list of vertices as values:
  This operates similarly to the dictionary accepted by the
  :meth:`hiperwalk.Coined.set_coin` method.
  The vertices are set as marked and
  *the coin operator is modified* accordingly.

Here are examples of how to create a coin that applies the Grover
operator to even vertices and the Hadamard operator to odd vertices:

>>> coined.set_coin(coin={'grover': list(range(0, 11, 2)),
...                       'minus_identity': []})
>>> coined.set_marked(marked=list(range(1, 11, 2)))
>>> C1 = coined.get_coin()
>>> M1 = coined.get_marked()
>>> coined.set_coin(coin='grover')
>>> coined.set_marked(marked={'minus_identity': list(range(1, 11, 2))})
>>> C2 = coined.get_coin()
>>> M2 = coined.get_marked()
>>> (C1 != C2).nnz == 0
True
>>> np.all(M1 == M2)
True

All these keys can be integrated into a single call to the
:meth:`hiperwalk.Coined.set_evolution` method when creating
an instance of the object.

Continuous-time Model
`````````````````````
The dynamics of the continuous-time quantum walk is
fully defined by the Hamiltonian.
The Hamiltonian is given by

.. math::

   H = -\gamma C - \sum_{m \in M} \ket m \bra m

where :math:`C` is either the adjacency matrix or
the laplacian matrix of the graph,
and :math:`M` is the set of marked vertices.
Therefore, three parameters are needed to describe the Hamiltonian:
* ``gamma``: the value of gamma.
* ``type``: the type of :math:`C`: adjacency or laplacian matrix.
* ``marked``: the list of marked vertices.
These parameters can be specified by the
:meth:`hiperwalk.ContinuousTime.set_hamiltonian` or by the
:meth:`hiperwalk.ContinuousTime.set_evolution` method.

On the other hand,
the evolution operator is defined as

.. math::

   U = e^{-\text{i} t H}.

Note that the continuous-time evolution operator is time-dependent.
The ``time`` may be specified using the constructor, by the
:meth:`hiperwalk.ContinuousTime.set_time` method or by the
:meth:`hiperwalk.ContinuousTime.set_evolution`.
``time`` accepts float values,
but if it is omitted, it is set to 1.
In addition, :math:`U` is calculated by a partial sum of
the Taylor series expansion.
The number of terms in the expansion can be specified in
the :meth:`hiperwalk.ContinuousTime.set_evolution` method
by the ``terms`` key or in
the :meth:`hiperwalk.ContinuousTime.set_terms` method.

>>> continuous.set_evolution(gamma=0.35, type='adjacency',
...                          time=0.5, terms=21)
>>> U = continuous.get_evolution()
>>> U
array([[ 9.69608676e-01-2.35604916e-16j, -7.40128936e-15+1.72333955e-01j,
        -1.51567821e-02+4.22911229e-13j,  2.17423162e-11-8.86411289e-04j,
         3.88400285e-05-9.93511542e-10j, -3.97187194e-08+1.36079115e-06j,
        -3.97187194e-08+1.36079115e-06j,  3.88400285e-05-9.93511542e-10j,
         2.17423162e-11-8.86411289e-04j, -1.51567821e-02+4.22911229e-13j,
        -7.40128936e-15+1.72333955e-01j],
       [-7.40128936e-15+1.72333955e-01j,  9.69608676e-01-2.35604916e-16j,
        -7.40128936e-15+1.72333955e-01j, -1.51567821e-02+4.22911229e-13j,
         2.17423162e-11-8.86411289e-04j,  3.88400285e-05-9.93511542e-10j,
        -3.97187194e-08+1.36079115e-06j, -3.97187194e-08+1.36079115e-06j,
         3.88400285e-05-9.93511542e-10j,  2.17423162e-11-8.86411289e-04j,
        -1.51567821e-02+4.22911229e-13j],
       [-1.51567821e-02+4.22911229e-13j, -7.40128936e-15+1.72333955e-01j,
         9.69608676e-01-2.35604916e-16j, -7.40128936e-15+1.72333955e-01j,
        -1.51567821e-02+4.22911229e-13j,  2.17423162e-11-8.86411289e-04j,
         3.88400285e-05-9.93511542e-10j, -3.97187194e-08+1.36079115e-06j,
        -3.97187194e-08+1.36079115e-06j,  3.88400285e-05-9.93511542e-10j,
         2.17423162e-11-8.86411289e-04j],
       [ 2.17423162e-11-8.86411289e-04j, -1.51567821e-02+4.22911229e-13j,
        -7.40128936e-15+1.72333955e-01j,  9.69608676e-01-2.35604916e-16j,
        -7.40128936e-15+1.72333955e-01j, -1.51567821e-02+4.22911229e-13j,
         2.17423162e-11-8.86411289e-04j,  3.88400285e-05-9.93511542e-10j,
        -3.97187194e-08+1.36079115e-06j, -3.97187194e-08+1.36079115e-06j,
         3.88400285e-05-9.93511542e-10j],
       [ 3.88400285e-05-9.93511542e-10j,  2.17423162e-11-8.86411289e-04j,
        -1.51567821e-02+4.22911229e-13j, -7.40128936e-15+1.72333955e-01j,
         9.69608676e-01-2.35604916e-16j, -7.40128936e-15+1.72333955e-01j,
        -1.51567821e-02+4.22911229e-13j,  2.17423162e-11-8.86411289e-04j,
         3.88400285e-05-9.93511542e-10j, -3.97187194e-08+1.36079115e-06j,
        -3.97187194e-08+1.36079115e-06j],
       [-3.97187194e-08+1.36079115e-06j,  3.88400285e-05-9.93511542e-10j,
         2.17423162e-11-8.86411289e-04j, -1.51567821e-02+4.22911229e-13j,
        -7.40128936e-15+1.72333955e-01j,  9.69608676e-01-2.35604916e-16j,
        -7.40128936e-15+1.72333955e-01j, -1.51567821e-02+4.22911229e-13j,
         2.17423162e-11-8.86411289e-04j,  3.88400285e-05-9.93511542e-10j,
        -3.97187194e-08+1.36079115e-06j],
       [-3.97187194e-08+1.36079115e-06j, -3.97187194e-08+1.36079115e-06j,
         3.88400285e-05-9.93511542e-10j,  2.17423162e-11-8.86411289e-04j,
        -1.51567821e-02+4.22911229e-13j, -7.40128936e-15+1.72333955e-01j,
         9.69608676e-01-2.35604916e-16j, -7.40128936e-15+1.72333955e-01j,
        -1.51567821e-02+4.22911229e-13j,  2.17423162e-11-8.86411289e-04j,
         3.88400285e-05-9.93511542e-10j],
       [ 3.88400285e-05-9.93511542e-10j, -3.97187194e-08+1.36079115e-06j,
        -3.97187194e-08+1.36079115e-06j,  3.88400285e-05-9.93511542e-10j,
         2.17423162e-11-8.86411289e-04j, -1.51567821e-02+4.22911229e-13j,
        -7.40128936e-15+1.72333955e-01j,  9.69608676e-01-2.35604916e-16j,
        -7.40128936e-15+1.72333955e-01j, -1.51567821e-02+4.22911229e-13j,
         2.17423162e-11-8.86411289e-04j],
       [ 2.17423162e-11-8.86411289e-04j,  3.88400285e-05-9.93511542e-10j,
        -3.97187194e-08+1.36079115e-06j, -3.97187194e-08+1.36079115e-06j,
         3.88400285e-05-9.93511542e-10j,  2.17423162e-11-8.86411289e-04j,
        -1.51567821e-02+4.22911229e-13j, -7.40128936e-15+1.72333955e-01j,
         9.69608676e-01-2.35604916e-16j, -7.40128936e-15+1.72333955e-01j,
        -1.51567821e-02+4.22911229e-13j],
       [-1.51567821e-02+4.22911229e-13j,  2.17423162e-11-8.86411289e-04j,
         3.88400285e-05-9.93511542e-10j, -3.97187194e-08+1.36079115e-06j,
        -3.97187194e-08+1.36079115e-06j,  3.88400285e-05-9.93511542e-10j,
         2.17423162e-11-8.86411289e-04j, -1.51567821e-02+4.22911229e-13j,
        -7.40128936e-15+1.72333955e-01j,  9.69608676e-01-2.35604916e-16j,
        -7.40128936e-15+1.72333955e-01j],
       [-7.40128936e-15+1.72333955e-01j, -1.51567821e-02+4.22911229e-13j,
         2.17423162e-11-8.86411289e-04j,  3.88400285e-05-9.93511542e-10j,
        -3.97187194e-08+1.36079115e-06j, -3.97187194e-08+1.36079115e-06j,
         3.88400285e-05-9.93511542e-10j,  2.17423162e-11-8.86411289e-04j,
        -1.51567821e-02+4.22911229e-13j, -7.40128936e-15+1.72333955e-01j,
         9.69608676e-01-2.35604916e-16j]])

Simulation Invocation
'''''''''''''''''''''

Once the evolution operator is set,
the :meth:`hiperwalk.QuantumWalk.simulate` method
needs to be called in order to carry out the simulation.
This method requires two arguments:
``range`` and ``state``.
The ``range`` parameter specifies the number of times that
the evolution operator is going to be applied to the ``state``.
``range`` also specifies when the simulation should
stop and which intermediate states need to be stored.
The simulation returns a list of states such that
the ``i``-th entry corresponds to the ``i``-th saved state.

Coined Model
````````````
There are three argument types for ``range``.

* integer: ``end``.
  The simulation saves the states from the ``0``-th to the
  ``end - 1``-th application of the evolution operator.

  >>> states = coined.simulate(range=10,
  ...                          state=coined.ket(0))
  >>> len(states)
  10
  >>> len(states[0]) == coined.hilbert_space_dimension()
  True

* 2-tuple of integer: ``(start, end)``.
  Save every state from the ``start``-th to time ``end - 1``-th
  application of the evolution operator.
  For example,
  if ``range=(2, 10)``, returns the states corresponding to
  the application of the evolution operator with exponents
  ``[2, 3, 4, 5, 6, 7, 8, 9]``.

  >>> states = coined.simulate(range=(2, 10),
  ...                          state=coined.ket(0))
  >>> len(states)
  8
  >>> len(states[0]) == coined.hilbert_space_dimension()
  True

  A 2-tuple of integer is the simplest way to obtain a single state.
  To obtain the state corresponding to the ``i``-th application of
  the evolution operator, use ``range(i, i + 1)``.

  >>> psi = coined.simulate(range=(10, 11),
  ...                       state=coined.ket(0))
  >>> # extract the desired state from the list of states
  >>> psi = psi[0]
  >>>
  >>> # check result
  >>> U = coined.get_evolution().todense()
  >>> phi = np.linalg.matrix_power(U, 10) @ coined.ket(0)
  >>> np.allclose(psi, phi)
  True

* 3-tuple of integer: ``(start, end, step)``.
  Save every state from time ``start`` to time ``end``
  separated by ``step`` applications of the evolution operator.
  For example,
  if ``range=(1, 10, 2)``, returns the states corresponding to
  the application of the evolution operator with exponents
  ``[1, 3, 5, 7, 9]``.

  >>> states = coined.simulate(range=(1, 10, 2),
  ...                          state=coined.ket(0))
  >>> len(states)
  5
  >>> len(states[0]) == coined.hilbert_space_dimension()
  True

Continuous-time Model
`````````````````````

Recall that,
in the continuous-time quantum walk model,
the evolution operator is time-dependent.
The evolution operator can be created
by passing a float value ``t`` to the
:meth:`hiperwalk.ContinuousTime.set_time` method.

>>> continuous.set_time(0.3)

For this reason,
we remove the responsability of dealing with float numbers from
:meth:`hiperwalk.ContinuousTime.simulate`.
And its ``range`` parameter describes
the number of times that
the evolution operator is going to be applied to the ``state``
(analogous to the coined model).
In this sense,
``t`` is interpreted as a single ``step``.

Analogous to the coined model,
there are three argument types for ``range``.

* integer: ``end``.
  The simulation saves the states from the ``0``-th to the
  ``end - 1``-th application of the evolution operator.
  The resulting states correspond to the timestamps
  ``0``, ``t``, ..., ``(end - 1)*t``.
  In the following example, the timestamps are
  ``[0, 0.3, ..., 2.7]``.

  >>> cont_states = continuous.simulate(range=10,
  ...                                   state=continuous.ket(0))
  >>> len(cont_states)
  10
  >>> len(cont_states[0]) == continuous.hilbert_space_dimension()
  True
  >>>
  >>> # cont_states correspond to timestamps
  >>> 0.3*np.array(range(10))
  array([0. , 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7])

* 2-tuple of integer: ``(start, end)``.
  Save every state from the initial state to the
  state after the ``(end - 1)``-th application of the evolution operator.
  That is, the stored states correspond to timestamps
  ``start*t``, ``(start + 1)*t``, ..., ``(end - 1)*t``.
  In the following example,
  the stored states correspond to timestamps
  ``[0.6, 0.9, ..., 2.7]``.

  >>> cont_states = continuous.simulate(range=(2, 10),
  ...                                   state=continuous.ket(0))
  >>> len(cont_states)
  8
  >>> len(cont_states[0]) == continuous.hilbert_space_dimension()
  True
  >>>
  >>> # cont_states correspond to timestamps
  >>> 0.3*np.array(range(2, 10))
  array([0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7])

  A 2-tuple of integer can be used to obtain a single state
  without updating the evolution operator,
  as long as the timestamp is a multiple of ``t``.
  More specifically, to obtain the state at timestamp ``i*t``,
  use ``range=(i, i + 1)``.
  The following example return the state at timestamp ``3``.

  >>> psi = continuous.simulate(range=(10, 11),
  ...                           state=continuous.ket(0))
  >>> # extract the single state from the list of states
  >>> psi = psi[0]
  >>>
  >>> # verify
  >>> U = continuous.get_evolution()
  >>> phi = np.linalg.matrix_power(U, 10) @ continuous.ket(0)
  >>> np.allclose(psi, phi)
  True

* 3-tuple of integer: ``(start, end, step)``.
  Save every state from the ``start``-th to
  the ``(end - 1)``-th application of the evolution operator.
  The saved states are separated by ``step``
  applications of the evolution operator.
  In the following example,
  the stored states correspond to timestamps
  ``[0.3, 0.9, 1.5, 2.1, 2.7]``, respectively.

  >>> cont_states = continuous.simulate(range=(1, 10, 2),
  ...                                   state=continuous.ket(0))
  >>> # single state returned
  >>> len(cont_states)
  5
  >>> len(cont_states[0]) == continuous.hilbert_space_dimension()
  True
  >>>
  >>> # cont_states correspond to timestamps
  >>> 0.3*np.array(range(1, 10, 2))
  array([0.3, 0.9, 1.5, 2.1, 2.7])

Calculating Probability
-----------------------

There are two ways of calculating probabilities:
:meth:`hiperwalk.QuantumWalk.probability`, and
:meth:`hiperwalk.QuantumWalk.probability_distribution`.

The
:meth:`hiperwalk.QuantumWalk.probability` method computes
the probability of the walker being found on a
subset of the vertices for each state.

>>> probs = coined.probability(states, [0, 1, 2])
>>> len(probs) == len(states)
True
>>> np.all([0 <= p and p <= 1  for p in probs])
True

The
:meth:`hiperwalk.QuantumWalk.probability_distribution` method
calculates the probability of each vertex.
Basically, the probability of vertex ``v`` is
the sum of the probabilities of each entry
corresponding to arcs with tail ``v``.

>>> prob_dist = coined.probability_distribution(states)
>>> len(prob_dist) == len(states)
True
>>> len(prob_dist[0]) != len(states[0])
True
>>> # Since each vertex on a cycle has degree 2, the following is True
>>> len(prob_dist[0]) == len(states[0]) / 2
True

.. note::
   For the Continuous model,
   :meth:`hiperwalk.ContinuousTime.probability` and
   :meth:`hiperwalk.ContinuousTime.probability_distribution` yield
   the same result.

Having obtained a probability distribution, the user may find it helpful to
visualize this data graphically to gain further insights. Graphical
representation can make complex data more understandable, reveal underlying
patterns, and support more effective data analysis.

For more information about how to create plots and customize them to best
represent your data, please refer to the following section. This will cover
the specifics of data visualization, including various plotting techniques
and customization options.
