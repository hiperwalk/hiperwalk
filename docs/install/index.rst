=======
Install
=======

Hiperwalk relies on a number of Python libraries. 
However, installing these Python libraries alone does not enable 
Hiperwalk to leverage High-Performance Computing (HPC). 
If you desire to install Hiperwalk with HPC support, please refer
to :ref:`docs_install_hpc_prerequisites` before proceeding 
with the Hiperwalk installation.

On this page, we outline the process for installing Hiperwalk on 
a newly installed Ubuntu 20.04 operating system. The steps will 
cover identifying the GPU, installing the GPU drivers, 
neblina-core, pyneblina, and all necessary Python libraries.

.. warning::

   Other distributions than Ubuntu 20.04 are currently not supported.

   Support for Ubuntu 22.04 is under development.
   Contributions are welcomed.

.. _docs_install_hiperwalk:

Hiperwalk
=========

Hiperwalk can be conveniently installed using pip. 
To begin, ensure that pip is installed on your system.

.. code-block:: shell

   sudo apt install python3-pip

The following command will install Hiperwalk as well as all its 
Python dependencies, which include
`numpy <https://numpy.org/>`_,
`scipy <https://scipy.org/>`_,
`networkx <https://networkx.org/>`_, and
`matplotlib <https://matplotlib.org/>`_.

.. warning::

    If you have older versions of these packages, they will likely be 
    updated. If you prefer not to have them updated, we recommend 
    `creating a virtual environment 
    <https://docs.python.org/3/library/venv.html>`_.   

.. code-block:: shell

   pip3 install hiperwalk

To verify the success of the installation, 
you can execute any code found in the
`examples directory of the repository
<https://`https://github.com/hiperwalk/hiperwalk/tree/2.0.x/examples>`_
or proceed to the :ref:`docs_tutorial`.

.. _docs_install_hpc_prerequisites:

HPC Prerequisites
=================

Before proceeding, it's advisable to update and upgrade your 
Ubuntu packages. Execute the following commands:

.. code-block:: shell

   sudo apt update
   sudo apt upgrade

   
Next, run the following commands to install the prerequisites:

.. code-block:: shell

   sudo apt install git
   sudo apt install g++
   sudo apt install cmake
   sudo apt install libgtest-dev
   sudo apt install python3-distutils
   sudo apt install python3-pip
   pip3 install pytest


These newly installed programs serve the following purposes:

* git: used to download neblina-core, pyneblina, and hiperwalk;
* g++: used for compiling neblina-core;
* cmake: essential for compiling neblina-core;
* libgtest-dev: verifies the successful installation of neblina-core;
* python3-distutils: aids in the installation of pyneblina;
* python3-pip: necessary for installing Python libraries;
* pytest: helps test pyneblina.

Although it's not essential, we **recommend** installing FFmpeg, 
which is used for generating animations.

.. code-block:: shell

   sudo apt install ffmpeg

GPU Driver
----------

To install the GPU driver, you can follow this
`tutorial for installing NVIDIA drivers <https://www.linuxcapable.com/install-nvidia-drivers-on-ubuntu-linux/>`_
Below, we have outlined the essential steps.

First, you'll need to identify your GPU by running the following command:

.. code-block:: shell

   lspci | grep -e VGA

You can then verify if the outputted
`GPU is CUDA compatible <https://developer.nvidia.com/cuda-gpus>`_.
If it is, execute the following command:

.. code-block:: shell

   ubuntu-drivers devices

This will list the available drivers for your GPU. We recommend 
installing the driver tagged with ``recommended`` at the end.
The driver's name typically follows the format ``nvidia-driver-XXX``
where ``XXX`` is a specific number.
For the subsequent steps in the installation process, substitute ``XXX`` 
as required. To install the GPU driver, execute the following command:

.. code-block:: shell

   sudo apt install nvidia-driver-XXX

Finally, **reboot you computer**.
After rebooting, if the installation was successful, 
running the following command:

.. code-block::

   nvidia-smi

should display GPU information such as the name, driver version, 
CUDA version, and so on. Alternatively, you can verify the 
availability of the **NVIDIA Settings** application by
pressing the ``Super`` key on your keyboard and 
typing ``nvidia settings``.

NVIDIA Toolkit
--------------

Once the GPU drivers have been successfully installed, it's 
necessary to install the NVIDIA Toolkit, allowing neblina-core 
to use CUDA. To do this, execute the following command:

.. code-block:: shell

   sudo apt install nvidia-cuda-toolkit

To verify the correct installation of the NVIDIA Toolkit, 
you can check if the ``nvcc`` compiler has been installed. 
This can be simply done by running the following command:

.. code-block:: shell

   nvcc --version


Installing neblina-core and pyneblina
=====================================

For HPC support,
Hiperwalk uses 
`neblina-core <https://github.com/paulomotta/neblina-core>`_,
and `pyneblina <https://github.com/paulomotta/pyneblina>`_.
Note that a computer with a **GPU compatible with CUDA** is required
for this.

The information in this guide is compiled from
`Paulo Motta's blog <https://paulomotta.pro.br/wp/2021/05/01/pyneblina-and-neblina-core/>`_,
`neblina-core github <https://github.com/paulomotta/neblina-core>`_,
and `pyneblina github <https://github.com/paulomotta/pyneblina>`_.

It is **strongly recommended** that neblina-core and pyneblina
are installed (i.e. cloned) in the same directory.
In this guide, we will install both projects into the home directory.
In Linux, the tilde (``~``) serves as an alias for the home directory.

neblina-core
------------

Firstly, clone the repository in the home directory.

.. code-block:: shell

   cd ~
   git clone https://github.com/paulomotta/neblina-core.git

Next, navigate to the neblina-core directory to compile and 
install the code.

.. code-block:: shell

   cd neblina-core
   cmake .
   make
   sudo make install
   sudo ldconfig

The ``ldconfig`` command creates a link for the newly installed neblina-core,
making it accessible for use by pyneblina.

To verify the successful installation of neblina-core,
execute the ``vector_test`` and ``matrix_test`` tests.

.. code-block:: shell

   ./vector_test
   ./matrix_test

pyneblina
---------

Before installing pyneblina,
ensure that neblina-core has been successfully installed.
Then, **reboot** your computer
to ensure that the ``ldconfig`` command takes effect.

To install pyneblina, first clone the repository into
**the same directory neblina-core was cloned**.
In this guide, we cloned neblina-core into the home directory.
Thus, execute:

.. code-block:: shell

   cd ~
   git clone https://github.com/paulomotta/pyneblina.git

Next, navigate to the newly created ``pyneblina`` directory to install it.

.. code-block:: shell

   cd pyneblina
   sudo python3 setup.py install

To verify whether the installation was successful, run the following test:

.. code-block:: shell

   python3 test.py
