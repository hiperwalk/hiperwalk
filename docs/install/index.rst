=======
Install
=======

Hiperwalk is built on top of some Python libraries.
By just installing the Python libraries,
Hiperwalk does not make use of High-Performance Computing (HPC).
If an installating with HPC support is desired,
you can jump to :ref:`docs_install_hpc_prerequisites`
before installing Hiperwalk.

In this page we describe the process of installing Hiperwalk in a
freshly installed Ubuntu 20.04 operating system.
We will describe the process of identifying the GPU, and
installing the GPU drivers, neblina-core, pyneblina, and
the necessary Python libraries.

.. warning::

   Other distributions than Ubuntu 20.04 are currently not supported.

   Support for Ubuntu 22.04 is under development.
   Contributions are welcomed.

.. _docs_install_hiperwalk:

Hiperwalk
=========

Hiperwalk can be easily installed via pip.
First of all, install pip.

.. code-block:: shell

   sudo apt install python3-pip

The following command
install Hiperwalk and all its Python dependencies.
Namely
`numpy <https://numpy.org/>`_,
`scipy <https://scipy.org/>`_,
`networkx <https://networkx.org/>`_, and
`matplotlib <https://matplotlib.org/>`_.

.. warning::

   If you have an old version of these packages,
   it will be probably updated.
   If you do not wish this to happen, we recommend you to
   `create a virtual environment
   <https://docs.python.org/3/library/venv.html>`_.

.. code-block:: shell

   pip3 install hiperwalk

To test if it the installation was successful,
you can execute any code in
`repository examples directory
<https://`https://github.com/hiperwalk/hiperwalk/tree/2.0.x/examples>`_
or go to the :ref:`docs_tutorial`.

.. _docs_install_hpc_prerequisites:

HPC Prerequisites
=================

Beforehand, it is recommended to update and upgrade the Ubuntu packages.
Execute

.. code-block:: shell

   sudo apt update
   sudo apt upgrade

   
Then,
execute the following commands to install the prerequisites.

.. code-block:: shell

   sudo apt install git
   sudo apt install g++
   sudo apt install cmake
   sudo apt install libgtest-dev
   sudo apt install python3-distutils
   sudo apt install python3-pip
   pip3 install pytest


The newly installed programs are needed for the following purposes.

* git: download neblina-core, pyneblina, hiperwalk;
* g++: compiling neblina-core;
* cmake: compiling neblina-core;
* libgtest-dev: test if neblina-core installation is correct;
* python3-distutils: install pyneblina;
* python3-pip: install Python libraries;
* pytest: test pyneblina.

It is **recommend** but not necessary to install FFmpeg.
It is used for generating animations.

.. code-block:: shell

   sudo apt install ffmpeg

GPU Driver
----------

You may follow this
`tutorial for installing NVIDIA drivers <https://www.linuxcapable.com/install-nvidia-drivers-on-ubuntu-linux/>`_
Here we list the core steps.

First we have to identify the GPU by running.

.. code-block:: shell

   lspci | grep -e VGA

We can then check if the outputted
`GPU is CUDA compatible <https://developer.nvidia.com/cuda-gpus>`_.
If that's the case, execute

.. code-block:: shell

   ubuntu-drivers devices

This is going to list the available drivers for your GPU.
We recommend to install the driver tagged with ``recommended`` at the end.
The driver name's `probably` has format ``nvidia-driver-XXX``
where ``XXX`` is some number.
For the remaining installation, substitute ``XXX`` accordingly when necessary.
For installing the GPU driver, execute

.. code-block:: shell

   sudo apt install nvidia-driver-XXX

Finally, **reboot you computer**.
After reboot,
if the installation was successful,
running the command

.. code-block::

   nvidia-smi

should print GPU Information,
e.g. name, driver version, CUDA version, etc.
Alternatively,
you can verify whether the **NVIDIA Settings** application is available by
pressing the ``Super`` keyboard key and typing ``nvidia settings``.

NVIDIA Toolkit
--------------

After installing the GPU drivers correctly,
it is necessary to install the NVIDIA Toolkit.
So the neblina-core can use CUDA.
Execute

.. code-block:: shell

   sudo apt install nvidia-cuda-toolkit

To check if the NVIDIA Toolkit was installed correctly,
check if the ``nvcc`` compiler was installed.
This can be done simply by running the command

.. code-block:: shell

   nvcc --version


Installing neblina-core And pyneblina
=====================================

For HPC support,
Hiperwalk uses 
`neblina-core <https://github.com/paulomotta/neblina-core>`_,
and `pyneblina <https://github.com/paulomotta/pyneblina>`_.
Although a computer with a **GPU compatible with CUDA** is necessary.


We compile the information in
`Paulo Motta's blog <https://paulomotta.pro.br/wp/2021/05/01/pyneblina-and-neblina-core/>`_,
`neblina-core github <https://github.com/paulomotta/neblina-core>`_,
and `pyneblina github <https://github.com/paulomotta/pyneblina>`_.

It is **strongly recommended** that neblina-core and pyneblina
are installed (i.e. cloned) in the same directory.
In this guide we will install both projects into the home directory.
In Linux, the tilde (``~``) is an alias for the home directory.

neblina-core
------------

First, clone the repository in the home directory.

.. code-block:: shell

   cd ~
   git clone https://github.com/paulomotta/neblina-core.git

Then,
enter the neblina-core directory to compile and install the code.

.. code-block:: shell

   cd neblina-core
   cmake .
   make
   sudo make install
   sudo ldconfig

The ``ldconfig`` commands creates a link for the newly installed neblina-core.
Making it available to be used by the pyneblina.

To check if neblina-core was installed successfully,
execute the ``vector_test`` and ``matrix_test`` tests.

.. code-block:: shell

   ./vector_test
   ./matrix_test

pyneblina
---------

Before installing pyneblina,
make sure that neblina-core was installed successfully.
Then, **reboot** your computer
to make sure the ``ldconfig`` command takes effect.

For installing pyneblina, first clone the repository into
**the same directory neblina-core was cloned**.
In this guide, neblina-core was cloned into the home directory.
Thus,

.. code-block:: shell

   cd ~
   git clone https://github.com/paulomotta/pyneblina.git

Then enter the newly created ``pyneblina`` directory to install it.

.. code-block:: shell

   cd pyneblina
   sudo python3 setup.py install

To verify if the installation completed successfully,
run the test.

.. code-block:: shell

   python3 test.py
