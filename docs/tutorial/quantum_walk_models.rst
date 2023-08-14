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
   cycle = hpw.Cycle(11)

For creating a quantum walk,
we must first define the graph in which the quantum walk will 
take place. This can be accomplished by passing a :class:`hiperwalk.Graph` 
object to the Quantum Walk constructor.

For example, consider that ``cycle`` is an instance of
:class:`hiperwalk.Cycle` with 11 vertices.

>>> cycle #doctest: +SKIP
<hiperwalk.graph.cycle.Cycle object at 0x7f657268c0d0>

Since :class:`hiperwalk.Cycle` is a child of :class:`hiperwalk.Graph`,
we can pass ``cycle`` to the quantum walk constructor.


To create a coined quantum walk, we execute

>>> coined = hpw.Coined(graph=cycle)
>>> coined #doctest: +SKIP
<hiperwalk.quantum_walk.coined_walk.Coined object at 0x7f655b0cd900>

The Hilbert space of the coined quantum walk has dimension
:math:`2|E|`, i.e. the number of arcs.

>>> coined.hilbert_space_dimension() == cycle.number_of_arcs()
True

To create a continuous-time quantum walk,
we need to pass both the graph and the ``gamma`` parameter.
The ``gamma`` parameter is required because there is no universally
accepted default value for ``gamma`` in the literature.

>>> continuous = hpw.ContinuousTime(graph=cycle, gamma=0.35)
>>> continuous #doctest: +SKIP
<hiperwalk.quantum_walk.continuous_walk.ContinuousTime object at 0x7f655b0cd8d0>

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
````````````````````````````````
Any state of the computational basis can be created using the
:meth:`hiperwalk.QuantumWalk.ket` method
as long as the correct label is passed.

In the coined quantum walk model,
the label of a state within the computational basis corresponds 
to an arc. You can use either the arc notation, which involves 
specifying the arc's tail and head,
or the arc number (an integer). Please refer to the Graph class 
for correct arc labeling guidelines.

>>> state = coined.ket(5, 6)
>>> state
array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0.])
>>> state2 = coined.ket(10)
>>> np.all(state == state2)
True

An easy way to convert between arc notation and the numerical label is by using
the :meth:`hiperwalk.Graph.arc` and
:meth:`hiperwalk.Graph.arc_number` methods.

>>> arc = cycle.arc(10)
>>> arc
(5, 6)
>>> cycle.arc_number(arc[0], arc[1])
10
>>>
>>> cycle.arc(cycle.arc_number(5, 6))
(5, 6)

In the continuous-time model,
the labels correspond directly to the labels of the vertices.

>>> continuous.ket(5)
array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])

Uniform superposition
`````````````````````

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
```````````````
Creating a generic state with the :meth:`hiperwalk.QuantumWalk.state` 
method can be a bit challenging. It expects a list consisting 
of ``[amplitude, label]`` entries, where each entry represents an amplitude
and a label of the computational basis.

In the coined model,
the labels are either numerical or a ``(tail, head)`` in  the arc notation.
An example using numeric labels is

>>> coined.state([0.5, 0],
...              [0.5, 2],
...              [0.5, 4],
...              [0.5, 6])
array([0.5, 0. , 0.5, 0. , 0.5, 0. , 0.5, 0. , 0. , 0. , 0. , 0. , 0. ,
       0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ])

and using the equivalent arc notation is

>>> coined.state([0.5, (0, 1)],
...              [0.5, (1, 2)],
...              [0.5, (2, 3)],
...              [0.5, (3, 4)])
array([0.5, 0. , 0.5, 0. , 0.5, 0. , 0.5, 0. , 0. , 0. , 0. , 0. , 0. ,
       0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ])

.. note::
   Do not forget the parenthesis while using the arc notation
   for generating a state.

For the continuous-time model,
the labels correspond to the labels of the vertices:

>>> continuous.state([0.5, 0],
...                  [0.5, 1],
...                  [0.5, 2],
...                  [0.5, 3])
array([0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0. ])

Since :meth:`hiperwalk.QuantumWalk.state` must return a valid state,
the amplitudes are renormalized when needed.

>>> continuous.state([1, 0],
...                  [1, 1],
...                  [1, 2],
...                  [1, 3])
array([0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0. ])


Simulation of Propagation
-------------------------

Once a quantum walk is defined, it is linked to an appropriate evolution operator. 
The user has the flexibility to modify this operator either during 
the quantum walk's creation or at any time afterward. Once the evolution 
operator is set, the user triggers the simulation process, 
deciding which intermediate states are of particular interest.

Configuring the evolution operator
``````````````````````````````````
To set up the evolution operator, users should refer to 
the :meth:`hiperwalk.QuantumWalk.set_evolution` method. 
Note that the parameters for this method depend on the model being used.

Regardless of the method employed, :meth:`hiperwalk.QuantumWalk.set_evolution` 
is invoked when the quantum walk is instantiated. Consequently, the 
constructors accept any parameter that is valid for the ``set_evolution`` method. 
To illustrate this point, let us examine the explicit evolution operator 
of two coined walks, which can be derived using 
the :meth:`hiperwalk.QuantumWalk.get_evolution` method from the QuantumWalk class.

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
''''''''''''
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
'''''''''''''''''''''
The dynamics of the continuous-time quantum walk is
fully defined by the Hamiltonian.
As a result, calling :meth:`hiperwalk.ContinuousTime.set_evolution`
is the same as calling :meth:`hiperwalk.ContinuousTime.set_hamiltonian`
in the same class.
The Hamiltonian is given by

.. math::

   H = -\gamma A - \sum_{m \in M} \ket m \bra m

where :math:`A` is the graph adjacency matrix and
:math:`M` is the set of marked vertices.
Therefore, the ``set_hamiltonian`` method accepts two arguments:
* ``gamma``: the value of gamma.
* ``marked``: the list of marked vertices.
For example,

>>> continuous2 = hpw.ContinuousTime(graph=cycle, gamma=0.35, marked=0)
>>> continuous2 #doctest: +SKIP
<hiperwalk.quantum_walk.continuous_walk.ContinuousTime object at 0x7ffad2de9510>

The evolution operator is defined as

.. math::

   U = e^{-\text{i} t H}.

Since the continuous-time evolution operator is time-dependent,
it should be generated as needed, based on the most recent timestamp. 

>>> U = continuous.get_evolution(time=1)
>>> continuous.set_marked(marked=0)
>>> U2 = continuous.get_evolution(time=1)
>>> np.any(U != U2)
True

Simulation Invocation
`````````````````````

Once the evolution operator is set,
the :meth:`hiperwalk.QuantumWalk.simulate` method 
needs to be called in order to carry out the simulation. 
This method primarily requires two arguments: 
``time`` and ``initial_state``.
The ``time`` argument specifies when the simulation should 
stop and which intermediate states need to be stored. 
The evolution operator will be applied to the ``initial_state``
as many times as necessary.
The simulation returns a list of states such that
the ``i``-th entry corresponds to the ``i``-th saved state.

Coined Model
''''''''''''
In the coined quantum walk model,
the ``time`` is discrete.
Thus, only integer entries are accepted.
There are three argument types for ``time``.

* integer: ``stop``.
  The final simulation time.

  >>> states = coined.simulate(time=10,
  ...                          initial_state=coined.ket(0))
  >>> len(states)
  1
  >>> len(states[0]) == coined.hilbert_space_dimension()
  True
  >>> U = coined.get_evolution().todense()
  >>> state = np.linalg.matrix_power(U, 10) @ coined.ket(0)
  >>> np.allclose(state, states[0])
  True

* 2-tuple of integer: ``(stop, step)``.
  Save every state from time ``0`` to time ``stop``
  separated by ``step`` applications of the evolution operator.
  For example,
  if ``time=(10, 2)``, returns the states obtained at times
  ``[0, 2, 4, 6, 8, 10]``.

  >>> states = coined.simulate(time=(10, 2),
  ...                          initial_state=coined.ket(0))
  >>> # single state returned
  >>> len(states)
  6
  >>> len(states[0]) == coined.hilbert_space_dimension()
  True

* 3-tuple of integer: ``(start, stop, step)``.
  Save every state from time ``start`` to time ``stop``
  separated by ``step`` application of the evolution operator.
  For example,
  if ``time=(1, 10, 2)``, returns the states at times
  ``[1, 3, 5, 7, 9]``.

  >>> states = coined.simulate(time=(1, 10, 2),
  ...                          initial_state=coined.ket(0))
  >>> # single state returned
  >>> len(states)
  5
  >>> len(states[0]) == coined.hilbert_space_dimension()
  True

Continuous-time Model
'''''''''''''''''''''
In the continuous-time quantum walk model,
the ``time`` parameter is continuous, which means it can accept 
float values. The operation is similar to the coined model, but here, 
the ``step`` parameter is used to rescale all values. 
This means the ``step`` parameter adjusts the scale of time, 
accommodating the continuous nature of time in this model.

* float : ``stop``. Unchanged.
* 2-tuple of float : ``(stop, step)``.
  The evolution operator ``continuous.get_evolution(time=step)`` is
  considered a single step and the ``time`` parameter is converted to
  ``(stop/step, 1)``.
  The value ``stop/step`` is rounded up if it is within
  a ``1e-05`` value of the next integer
  and rounded down otherwise.

* 3-tuple of float : ``(start, stop, step)``.
  The evolution operator ``continuous.get_evolution(time=step)`` is
  considered a single step and the ``time`` is converted to
  ``(start/step, stop/step, 1)``.
  The values ``start/step`` and ``stop/step`` are subjected to rounding. 
  If it's within a 1e-05 value of the next integer, it's rounded up; 
  otherwise, it's rounded down. This ensures a more accurate representation 
  of time within the model's continuous framework.

For example, if ``time=(10, 0.51)`` --
which is equivalent to ``time=(0, 10, 0.51)`` --
it is converted to ``(19, 1)``.
Thus, the states corresponding to timestamps
``[0.   , 0.501, 1.002, 1.503, ..., 9.018, 9.519]`` wil be stored.
On the other hand, if ``time=(10, 0.5000001)``,
it is converted to ``(20, 1)``, which
results in the states corresponding to the timestamps
``[ 0.       ,  0.5000001,  1.0000002,  ...,  9.5000019, 10.000002 ])``.

Calculating Probability
-----------------------

There are two ways of calculating probabilities:
:meth:`hiperwalk.QuantumWalk.probability` and
:meth:`hiperwalk.QuantumWalk.probability_distribution`.
:meth:`hiperwalk.QuantumWalk.probability` computes
the probability of each state entry.

>>> probs = coined.probability(states)
>>> len(probs) == len(states)
True
>>> len(probs[0]) == len(states[0])
True

:meth:`hiperwalk.QuantumWalk.probability_distribution`
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
