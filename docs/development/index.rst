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

To send a bug report, `add a new issue on Hiperwalk's github
<https://github.com/hiperwalk/hiperwalk/issues/new>`_.

A good way to start contributing to Hiperwalk is to solve any of the
`open issues <https://github.com/hiperwalk/hiperwalk/issues>`_.

Alternatively, to send a bug report,
contribute to Hiperwalk or contact the developers, 
write an email to hiperwalk@gmail.com.

Downloading
===========

The ``main`` branch is the main development area for new features. 
To contribute, first clone the repository.

.. code-block:: shell

    git clone https://github.com/hiperwalk/hiperwalk.git

Hiperwalk is built upon several Python libraries. 
Before developing for Hiperwalk, these libraries need to be installed.
These libraries include
`numpy <https://numpy.org/>`_,
`scipy <https://scipy.org/>`_,
`networkx <https://networkx.org/>`_, and
`matplotlib <https://matplotlib.org/>`_.
It is recommended to install these libraries in a new `virtual environment
<https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments>`_.

Suppose Hiperwalk was installed in the ``~/hiperwalk`` directory.
To create a new virtual environment on this directory,
issue the command

.. code-block:: shell

    python3 -m venv ~/hiperwalk/.venv

To activate the virtual environment use the command

.. code-block:: shell

   source ~/hiperwalk/.venv/bin/activate

With the appropriate virtual environment activated,
the required libraries can be installed.
To install the required libraries, use the following commands:

.. code-block:: shell

   pip3 install numpy
   pip3 install scipy
   pip3 install networkx
   pip3 install matplotlib

Next, make the desired modifications and submit a pull request.
We strongly recommend reading `best practices for pull requests
<https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/best-practices-for-pull-request>`_.

Testing
=======

Tests are located in the ``hiperwalk/tests/`` directory.
If a complete installation (including GPU support) was performed, 
execute:

.. warning::

   Do not forget to activate the virtual environment
   before running the tests.

.. code-block:: shell

    ./run_all.sh

If you installed the standalone version (without any HPC support), 
execute:

.. code-block:: shell

    ./run_hpc_none.sh

Documentation
=============

There are likely to be new features in the ``main`` branch. 
These features are documented online in the  **latest** version.

Install Requirements
--------------------

To generate the current, under-development documentation locally, 
all Hiperwalk requirements must be installed
(see :ref:`Hiperwalk section in the installation guide
<docs_hpc_enabled>`).

Assuming that all Hiperwalk requirements are installed, 
the following commands install the remaining
documentation-specific requirements.

.. code-block:: shell

   sudo apt install python3-sphinx
   sudo apt install graphviz

.. warning::

   Do not forget to activate the virtual environment
   before installing the following Python packages.

.. code-block:: shell

   pip3 install numpydoc
   pip3 install sphinx-autodoc-typehints
   pip3 install pydata-sphinx-theme

Generate Documentation
----------------------

Within the ``hiperwalk/docs/`` directory, execute:

.. warning::

   Do not forget to activate the virtual environment
   before generating the documentation.

.. code-block:: shell

   ./go

View Generated Documentation
----------------------------

If the process was successful, the newly compiled documentation 
is now available for local access. To view (and test) it, open the 
file  ``hiperwalk/docs/build/html/index.html``
in your preferred browser -- for instance, by double-clicking on it.
