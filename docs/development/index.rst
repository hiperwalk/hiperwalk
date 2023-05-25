===========
Development
===========

The ``main`` branch is where the new features are developed.
To contribute, clone the repository.

.. code-block:: shell

    git clone https://github.com/hiperwalk/hiperwalk.git

Then, make the desired alterations.
And make a pull request.

.. todo::

   Continuous model must be implemented.

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
