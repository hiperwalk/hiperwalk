.. _docs_tutorial:

Tutorial
========

Every Hiperwalk-based code follows up to five steps.

#. Import Hiperwalk.
#. Create the graph.
#. Create the quantum walk based on the previous graph.
#. Simulate the walk.
#. Exhibit the results.

A simple example is the coined walk on the line.
Do not struggle with the details for now.
We just illustrate the steps.

.. testsetup::

   from sys import path as sys_path
   sys_path.append("../..")

Import Hiperwalk
----------------

>>> import hiperwalk as hpw

Create the graph
----------------

Here we create a line with 11 vertices.
The result is an object of the :class:`hiperwalk.Line` class.

>>> N = 11
>>> line = hpw.Line(N)
>>> line #doctest: +SKIP
<hiperwalk.graph.line.Line object at 0x7ff59f1900d0>

Create the quantum walk based on the previous graph
---------------------------------------------------

We create a coined quantum walk on the line with
11 vertices by passing the created graph as an
argument to the quantum walk constructor.
This results in an object of the :class:`hiperwalk.CoinedWalk` class.

>>> qw = hpw.CoinedWalk(line)
>>> qw #doctest: +SKIP
<hiperwalk.quantum_walk.coined_walk.CoinedWalk object at 0x7f2691de9840>

Simulate the walk
-----------------

Before simulating the walk,
To simulate the walk we need to specify the initial state.
One way to create the initial state is by using the
:meth:`hiperwalk.CoinedWalk.ket` method,
which creates a valid state of the computational basis.

>>> vertex = N // 2
>>> initial_state = qw.ket(vertex, vertex + 1)
>>> initial_state
array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0.])

This state corresponds to the walker being on
vertex 5 pointing to vertex 6
(the labels of the vertices go from 0 to 10).

To simulate the walk we must specify the number of steps
(number of applications of the evolution operator)
and the initial state.
By specifying only the final time,
the result is the final state.
If everything was installed properly,
the :meth:`hiperwalk.CoinedWalk.simulate` method automatically uses
high-performance computing to perform the matrix-vector multiplications.

>>> final_state = qw.simulate(time=N//2,
...                           initial_state=initial_state)



Exhibit the results
-------------------

The results exhibition may be a simple print

>>> final_state
array([[ 0.1767767 ,  0.        ,  0.        , -0.1767767 ,  0.35355339,
         0.        ,  0.        ,  0.        , -0.35355339,  0.        ,
         0.        ,  0.        ,  0.35355339,  0.        ,  0.        ,
         0.70710678,  0.1767767 ,  0.        ,  0.        ,  0.1767767 ]])
         

or a more sofisticated output.
Frequently, we are interested in the probability of the walker being
found on each vertex.
This can be done via the
:meth:`hiperwalk.CoinedWalk.probability_distribution` method
by passing the final state as argument.

>>> probability = qw.probability_distribution(final_state)
>>> probability
array([[0.03125, 0.     , 0.15625, 0.     , 0.125  , 0.     , 0.125  ,
        0.     , 0.53125, 0.     , 0.03125]])

It is also possible to plot the probability distribution
with a simple command.

>>> hpw.plot_probability_distribution(probability) #doctest: +SKIP

Resulting in the following plot.

.. figure:: probability_distribution.png
    :alt: Plot of the probability distribution.

    Probability distribution of a quantum walk on a line.

Next Steps
----------

The remainder of the tutorial is subdivided into the following sections.

.. toctree::
    :maxdepth: 1

    graphs.rst
    quantum_walk_models.rst
    plotting_customization.rst
