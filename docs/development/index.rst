.. _docs_development:

===========
Development
===========

This page describes how to download Hiperwalk 
from GitHub, how to test it, and how to generate 
its documentation. Before that, if you wish to 
contribute or extend Hiperwalk's capabilities by 
implementing new quantum walk models, please contact 
the developers (see the next section).

Bug Report and Contact
======================

To send a bug report or
contribute to Hiperwalk or contact the developers, 
write an email to hiperwalk@gmail.com.

Downloading
===========

HiperWalk is built upon several Python libraries. 
Before developing for HiperWalk, these libraries need to be installed. 
The required libraries include
`numpy <https://numpy.org/>`_,
`scipy <https://scipy.org/>`_,
`networkx <https://networkx.org/>`_, and
`matplotlib <https://matplotlib.org/>`_.
To install these libraries, use the following commands:

.. code-block:: shell

   pip3 install numpy
   pip3 install scipy
   pip3 install networkx
   pip3 install matplotlib

The ``main`` branch is the main development area for new features. 
To contribute, first clone the repository.

.. code-block:: shell

    git clone https://github.com/hiperwalk/hiperwalk.git

Next, make the desired modifications and submit a pull request.

Testing
=======

Tests are located in the ``hiperwalk/tests/`` directory.
If a complete installation (including HPC support) was performed, 
execute:

.. code-block:: shell

    ./run_all.sh

If you installed the standalone version (without HPC support), 
execute:

.. code-block:: shell

    ./run_nonhpc.sh

Documentation
=============

There are likely to be new features in the ``main`` branch. 
These features are documented online in the  **latest** version.

Install Requirements
--------------------

To generate the current, under-development documentation locally, 
all HiperWalk requirements must be installed
(see :ref:`HiperWalk section in the installation guide
<docs_hpc_enabled>`).

Assuming that all HiperWalk requirements are installed (see *TODO*), 
the following commands install the remaining documentation-specific requirements.

.. code-block:: shell

   sudo apt install python3-sphinx
   sudo apt install graphviz
   pip3 install numpydoc
   pip3 install sphinx-autodoc-typehints
   pip3 install pydata-sphinx-theme

Generate Documentation
----------------------

Within the ``hiperwalk/docs/`` directory, execute:


.. code-block:: shell

   ./go

View Generated Documentation
----------------------------

If the process was successful, the newly compiled documentation 
is now available for local access. To view (and test) it, open the 
file  ``hiperwalk/docs/build/html/index.html``
in your preferred browser -- for instance, by double-clicking on it.

Todo
====
* More examples 
* Other quantum walk models
* Releases notes
* The current version only works with GTK 3.0. Implementation using GTK 4.0 is postponed to the next release.

