.. _docs_tutorial_plotting:

========
Plotting
========

After calculating the probability distribution of one or multiple states,
the user may wish to plot the distribution.
This can be done by invoking the standalone
:meth:`hiperwalk.plot_probability_distribution` function.
To plot the probability distribution given the probabilities,
simply pass the probabilities as arguments to the
:meth:`hiperwalk.plot_probability_distribution` function.

>>> import hiperwalk as hpw
>>> # create graph, quantum walk, simulate
>>> # compute the probability distribution
>>> hpw.plot_probability_distribution(prob_dist) #doctest: +SKIP

This will generate ``len(probs)`` images where
the ``i``-th image corresponds to the ``i``-th probability.
On a Jupyter notebook, the images will be shown in sequence simultaneously.
On a terminal, the first image will be shown and
the program will wait for the user to close the image
-- by pressing the ``q`` key for example --
before showing the next one.

Customization
=============

Albeit plotting was simple,
configuring the plot to behave as the user wishes may be a bit tricky.
We antecipate that the plotting was built on top of
`Matplotlib <https://matplotlib.org/>`_ and
`NetworkX <https://networkx.org/>`_.
Hence, every key argument accepted by these libraries is
accepted by Hiperwalk!
Of course, this depends on the plot type.
It does not make sense to demand ``node_color='red'`` if
a bar plot is being request.

Plot Types
----------

#. TODO
#. TODO

Hiperwalk-Specific Arguments
----------------------------
show
filename
animate
interval
rescale
