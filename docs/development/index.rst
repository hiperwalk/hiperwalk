.. _docs_development:

===========
Development
===========

HiperWalk is built on top of some Python libraries.
Before developing for HiperWalk,
we must install these libraries.
HiperWalk requires
`numpy <https://numpy.org/>`_,
`scipy <https://scipy.org/>`_,
`networkx <https://networkx.org/>`_, and
`matplotlib <https://matplotlib.org/>`_.
To install these libraries run

.. code-block:: shell

   pip3 install numpy
   pip3 install scipy
   pip3 install networkx
   pip3 install matplotlib

The ``main`` branch is where the new features are developed.
To contribute, clone the repository.

.. code-block:: shell

    git clone https://github.com/hiperwalk/hiperwalk.git

Then, make the desired alterations.
And make a pull request.

Testing
=======

Tests are located into the ``hiperwalk/tests/`` directory.
If you performed the complete installation (with HPC support),
execute

.. code-block:: shell

    ./run_all.sh

If you installed the standalone version (with no HPC support),
execute

.. code-block:: shell

    ./run_nonhpc.sh

Documentation
=============

It is very likely that there are new features in the ``main`` branch.
These new features are documentated online in the **latest** version.

Install Requirements
--------------------

To generate the current (under development) documentation locally,
it is necessary to install all the HiperWalk requirements
(see :ref:`HiperWalk section in the installation guide
<docs_install_hiperwalk>`).

Supposing that all HiperWalk requirements are installed
(see *TODO*),
the following commands install the remaining documentation only requirements.

.. code-block:: shell

   sudo apt install python3-sphinx
   sudo apt install graphviz
   pip3 install numpydoc
   pip3 install sphinx-autodoc-typehints
   pip3 install pydata-sphinx-theme

Generate Documentation
----------------------

Inside the ``hiperwalk/docs/`` directory, execute


.. code-block:: shell

   ./go

View Generated Documentation
----------------------------

If everything went right,
the recently compiled documentation is available locally.
To view (and test) it,
open the file ``hiperwalk/docs/build/html/index.html``
in the browser of your reference --
e.g. by double clicking it.

Todo
====
* Releases notes.
* The current version was tested with Ubuntu 20.04.
  Hiperwalk failed to be configured in Ubuntu 22.04.
* The current version only works with GTK 3.0.
  Implementation using GTK 4.0 are postponed to the next release.
