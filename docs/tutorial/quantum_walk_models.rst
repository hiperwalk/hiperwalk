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

>>> continuous.hilbert_space_dimension() == cycle.number_of_vertices()
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

Once a quantum walk is created,
a valid evolution operator is associated with it.
The user may change the evolution operator
upon the quantum walk creation or afterwards.
After defining the evolution operator,
the user invokes the simulation process,
determining which intermediate states are of interest.

Configuring the evolution operator
``````````````````````````````````
To configure the evolution operator,
check the :meth:`hiperwalk.QuantumWalk.set_evolution` method.
This method parameters are model-dependent.

Regardless of the method,
:meth:`hiperwalk.QuantumWalk.set_evolution` is invoked upon the
Quantum Walk instantiation.
Hence, the constructors accept any parameter valid for ``set_evolution``.
To illustrate this,
let us analyze the explicit evolution operator of two Coined Walks
(which can be obtained by :meth:`hiperwalk.QuantumWalk.get_evolution`).

>>> U = coined.get_evolution()
>>> coined.set_evolution(shift='flipflop', coin='grover')
>>> U2 = coined.get_evolution()
>>> (U != U2).nnz == 0 # efficient way of comparing sparse arrays
False
>>> coined2 = hpw.CoinedWalk(graph=cycle, shift='flipflop', coin='grover')
>>> U3 = coined.get_evolution()
>>> (U2 != U3).nnz == 0
True

Coined Model
''''''''''''
The :meth:`hiperwalk.CoinedWalk.set_evolution`
accepts three key arguments:
``shift``, ``coin``, and ``marked``.
Respectively,
they are the arguments of
:meth:`hiperwalk.CoinedWalk.set_shift`,
:meth:`hiperwalk.CoinedWalk.set_coin`, and
:meth:`hiperwalk.CoinedWalk.set_marked`.

The ``shift`` key must have either a string value
(``'persistent'`` or ``'flipflop'``) or
the explicit operator.

The ``coin`` key accepts four types of entries.

* The explicit coin.
* A string with the coin name to be applied to all vertices.
* A list of strings of size :math:`|V|` with the coin names
  where the :math:`i`-th coin will be applied to the :math:`i`-th vertex.
* A dictionary with the coin name as key and
  the list of vertices as values.
  The coin depicted as key will be applied to
  the vertices depicted as values.
  If the list of vertices is the empty list ``[]``,
  that coin is going to be applied to all remaining vertices.

There are eight possible coin names:
``'fourier'``, ``'grover'``, ``'hadamard'``, ``'identity'``, and
its variants with the ``'minus_'`` prefix to it.

The following are equivalent ways of generating a coin
that applies Grover to all vertices.

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

The following are valid ways of generating a con that applies
Grover to even vertices and Hadamard to odd vertices.

>>> coined.set_coin(coin=['grover' if i % 2 == 0 else 'hadamard'
...                       for i in range(11)])
>>> C1 = coined.get_coin()
>>> coined.set_coin(coin={'grover': list(range(0, 11, 2)),
...                       'hadamard': []})
>>> C2 = coined.get_coin()
>>> (C1 != C2).nnz == 0
True

The ``marked`` key accepts two types of entries.

* A list of the marked vertices.
  The vertices are just set as marked,
  but the coin operator remains unchanged.
* A dictionary with the coin name as key and
  the list of vertices as values.
  This is analogous to the dictionary accepted by
  :meth:`hiperwalk.CoinedWalk.set_coin`.
  The vertices are set as marked and
  *the coin operator is changed* accordingly.

The following are two ways of generating the same evolution operator
with the same set of marked vertices.

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

We may combine all these keys in a single
:meth:`hiperwalk.CoinedWalk.set_evolution` call
or object instantiation.

Continuous Model
''''''''''''''''
The dynamics of the Continuous Quantum Walk is
completely described by the Hamiltonian.
Hence, :meth:`hiperwalk.ContinuousWalk.set_evolution`
is equivalent to :meth:`hiperwalk.ContinuousWalk.set_hamiltonian`.
The Hamiltonian is given by

.. math::

   H = -\gamma A - \sum_{m \in M} \ket m \bra m

where :math:`A` is the graph adjacency matrix and
:math:`M` is the set of marked vertices.
Hence ``set_hamiltonian`` accepts two arguments.
* ``gamma``: the value of gamma.
* ``marked``: the list of marked vertices.
For example,

>>> continuous2 = hpw.ContinuousWalk(graph=cycle, gamma=0.35, marked=0)
>>> continuous2 #doctest: +SKIP
<hiperwalk.quantum_walk.continuous_walk.ContinuousWalk object at 0x7ffad2de9510>

The evolution operator is calculated by

.. math::

   U = e^{-\text{i} t H}.

Since the Continuous Walk evolution operator is time-dependent,
it must be generated by demand given the last timestamp.

>>> U = continuous.get_evolution(time=1)
>>> continuous.set_marked(marked=0)
>>> U2 = continuous.get_evolution(time=1)
>>> np.any(U != U2)
True

Simulation Invocation
`````````````````````

After setting the evolution operator,
the :meth:`hiperwalk.QuantumWalk.simulate` method must be invoked
to perform the simulation.
There are two key arguments for this method:
``time`` and ``initial_state``.
The ``time`` describes when the simulation stops
and which intermediate states must be saved.
The evolution operator will be applied to the ``initial_state``
as many times as needed.
The simulation returns a list of states such that
the ``i``-th entry corresponds to the ``i``-th saved state.

Coined Model
''''''''''''
In the Coined Walk model,
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

Continuous Model
''''''''''''''''
In the Continuous Walk model,
the ``time`` is continuous.
Thus, float entries are accepted.
It works analogous to the Coined Model,
but ``step`` is used to rescale all values.

* float : ``stop``. Unchanged.
* 2-tuple of float : ``(stop, step)``.
  The evolution operator ``continuous.get_evolution(time=step)`` is
  considered a single step and the ``time`` is converted to
  ``(stop/step, 1)``.
  The value ``stop/step`` is rounded up if it is within
  a ``1e-05`` value of the next integer
  and rounded down otherwise.

* 3-tuple of float : ``(start, stop, step)``.
  The evolution operator ``continuous.get_evolution(time=step)`` is
  considered a single step and the ``time`` is converted to
  ``(start/step, stop/step, 1)``.
  The values ``start/step`` and ``stop/step`` are rounded up
  if it is within a ``1e-05`` value of the next integer
  and rounded down otherwise.

Calculating Probability
-----------------------

probability vs probability distribution
