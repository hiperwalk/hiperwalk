Quantum Walk Models
===================

There is a plethora of Quantum Walk models.
Hiperwalk offers a common interface for all quantum walks.
This interface is the abstract class :class:`hiperwalk.QuantumWalk` --
a class that cannot be instantiated.
It is possible, however to inherit from :class:`hiperwalk.QuantumWalk`.
All its methods and attributes will be available for the child class.
Even though the abstract methods must be overriden by the child class
since they are model dependent.

Currently, there are two models available.
The Coined model (:class:`hiperwalk.CoinedWalk`) and
the Continuous model (:class:`hiperwalk.ContinuousWalk`).
The user is encouraged to implement a new model and
add it to the Hiperwalk package
(see :ref:`docs_development` Section).

Creating a Quantum Walk
-----------------------

.. testsetup::

   from sys import path as sys_path
   sys_path.append("../..")
   import numpy as np
   import hiperwalk as hpw 
   cycle = hpw.Cycle(11)

For creating a Quantum Walk,
we must define in which graph the Quantum Walk occurs.
This can be done by passing a :class:`hiperwalk.Graph` object to
the Quantum Walk constructor.

For example, suppose that ``cycle`` is an instance of
:class:`hiperwalk.Cycle` with 11 vertices.

>>> cycle #doctest: +SKIP
<hiperwalk.graph.cycle.Cycle object at 0x7f657268c0d0>

Since :class:`hiperwalk.Cycle` is a child of :class:`hiperwalk.Graph`,
we can also pass ``cycle`` to the quantum walk constructor.


For creating the Coined Quantum Walk we execute

>>> coined = hpw.CoinedWalk(graph=cycle)
>>> coined #doctest: +SKIP
<hiperwalk.quantum_walk.coined_walk.CoinedWalk object at 0x7f655b0cd900>

The Hilbert space of the Coined Quantum Walk has dimension
:math`2|E|`:, i.e. the number of arcs.
>>> coined.hilbert_space_dimension() == cycle.number_of_arcs()
True

For creating the Continuous Walk,
we must pass both the graph and the ``gamma`` parameter.
The ``gamma`` is required because
no default ``gamma`` value exists in the literature.

>>> continuous = hpw.ContinuousWalk(graph=cycle, gamma=0.35)
>>> continuous #doctest: +SKIP
<hiperwalk.quantum_walk.continuous_walk.ContinuousWalk object at 0x7f655b0cd8d0>

The Hilbert space of the Continuous Quantum Walk has dimension
:math`|V|`:, i.e. the number of verties.
>>> coined.hilbert_space_dimension() == cycle.number_of_vertices()
True

Creating a State
----------------

Hiperwalk offers three easy ways of creating a state.
The user can create a state of the computational basis
(:meth:`hiperwalk.QuantumWalk.ket`),
a uniform superposition (:meth:`hiperwalk.QuantumWalk.uniform_state`,
or a general superposition (:meth:`hiperwalk.QuantumWalk.state`).

State of the computational basis
````````````````````````````````
Any state of the computational basis may be created using the
:meth:`hiperwalk.QuantumWalk.ket` method
as long as the correct label is passed.

For the Coined Quantum Walk model,
the label of a computational basis state is given by its arc.
We may use either the arc notation (passing the arc's tail and head),
or the arc label (refer to the Graph class for appropriate arc labelling).

>>> state = coined.ket(5, 6)
>>> state
array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0.])
>>> state2 = coined.ket(10)
>>> np.all(state == state2)
True

An easy way to convert between arc notation and arc label is by using
the :meth:`hiperwalk.Graph.arc` and
:meth:`hiperwalk.Graph.arc_label` methods.

>>> arc = cycle.arc(10)
>>> arc
(5, 6)
>>> cycle.arc_label(arc[0], arc[1])
10
>>>
>>> cycle.arc(cycle.arc_label(5, 6))
(5, 6)

For the Continuous model,
the label are the vertices labels.

>>> continuous.ket(5)
array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])

Uniform superposition
`````````````````````

To greate the uniform superposition,
the :meth:`hiperwalk.QuantumWalk.uniform_state` method
works for whichever model.

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


General state
`````````````
Creating a general state with the
:meth:`hiperwalk.QuantumWalk.state` method may be a bit tricky.
A list of amplitudes and computational basis labels is expected,
i.e. multiple ``[amplitude, label]`` entries.

For the Coined model,
the labels are either the arc label or the arc notation ``(tail, head)``.
Using the arc labels we obtain

>>> coined.state([0.5, 0],
...              [0.5, 2],
...              [0.5, 4],
...              [0.5, 6])
array([0.5, 0. , 0.5, 0. , 0.5, 0. , 0.5, 0. , 0. , 0. , 0. , 0. , 0. ,
       0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ])

While using the equivalent arc notation we obtain

>>> coined.state([0.5, (0, 1)],
...              [0.5, (1, 2)],
...              [0.5, (2, 3)],
...              [0.5, (3, 4)])
array([0.5, 0. , 0.5, 0. , 0.5, 0. , 0.5, 0. , 0. , 0. , 0. , 0. , 0. ,
       0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ])

.. note::
   Do not forget the parenthesis while using the arc notation
   for generating a state.

For the Continuous model,
the labels are the vertices labels.

>>> continuous.state([0.5, 0],
...                  [0.5, 1],
...                  [0.5, 2],
...                  [0.5, 3])
array([0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0. ])

Since :meth:`hiperwalk.QuantumWalk.state` must return a valid state,
the amplitudes are normalized if needed.

>>> continuous.state([1, 0],
...                  [1, 1],
...                  [1, 2],
...                  [1, 3])
array([0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0. ])

Simulating
----------

Details about the time parameter

change initial_condition to initial_state everywhere

Calculating Probability
-----------------------

probability vs probability distribution
