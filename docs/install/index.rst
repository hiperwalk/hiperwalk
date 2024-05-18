=======
Install
=======

Hiperwalk offers multiple installation paths depending on your requirements. 
You can install Hiperwalk without High-Performance 
Computing (HPC) support, or opt for a Docker-based installation 
that includes HPC capabilities. 

Choose your installation method:

- :ref:`docs_basic_installation` without HPC support  

- :ref:`docs_hpc_enabled`

  - Docker Installation

  - Local Installation (for advanced users)


.. _docs_basic_installation:

------------------
Basic Installation
------------------

Hiperwalk can be installed without HPC support. 
This installation relies on certain Python libraries.

Installation Steps
==================

Hiperwalk can be conveniently installed using pip.
To begin, ensure that pip is installed on your system.

.. code-block:: shell

   sudo apt install python3-pip

Hiperwalk has several Python dependencies, including
`numpy <https://numpy.org/>`_,
`scipy <https://scipy.org/>`_,
`networkx <https://networkx.org/>`_, and
`matplotlib <https://matplotlib.org/>`_.

.. warning::

    If you have older versions of these packages, they will likely be
    updated. If you prefer not to have them updated, we recommend
    `creating a virtual environment
    <https://docs.python.org/3/library/venv.html>`_.

The following command will install Hiperwalk and its
dependencies:

.. code-block:: shell

   pip install hiperwalk

To verify the success of the installation, 
you can execute any example code available in the
`GitHub repository
<https://github.com/hiperwalk/hiperwalk/tree/master/examples>`_
or any Jupyter notebook in the
:ref:`docs_examples` section. Alternatively,
you can proceed to the :ref:`docs_tutorial` section.

To update an older version of the hiperwalk package:

.. code-block:: shell

   pip install hiperwalk --upgrade

To uninstall the hiperwalk package:

.. code-block:: shell

   pip uninstall hiperwalk


.. _docs_hpc_enabled:

------------------------
HPC-Enabled Installation
------------------------

Hiperwalk supports HPC through either a Docker installation 
or a local installation. HPC capabilities can be leveraged 
using parallelism on multicore CPUs and, optionally, NVIDIA GPUs.

The following instructions will guide you through setting 
up CPU-based HPC, with optional GPU support.

.. _docs_gpu_prerequisites:

GPU Prerequisites
=================

Skip this section if GPU support is not needed.
To install the GPU driver, you can follow this
`tutorial for installing NVIDIA drivers 
<https://www.linuxcapable.com/install-nvidia-drivers-on-ubuntu-linux/>`_.
Below, we have outlined the essential steps.

.. warning::

	If you have any concerns about the following commands, 
	especially for installing the NVIDIA driver, 
	please contact your local support team for assistance.

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
running the command

.. code-block::

   nvidia-smi

should display GPU information such as the name, driver version,
CUDA version, and so on. Alternatively, you can verify the
availability of the **NVIDIA Settings** application by
pressing the ``Super`` key on your keyboard and
typing ``nvidia settings``.


Docker Installation
===================

Using Hiperwalk on Docker offers numerous benefits. 
Docker, a form of containerization, automatically includes Hiperwalk, 
its prerequisites, and all HPC-enabling software. 
It provides a lightweight, portable, and scalable environment, 
ensuring seamless deployment across different systems. 
Docker simplifies dependency management, updates, and configuration replication, 
enhancing consistency and reliability. 


Installation Steps
------------------

Single time configuration of Docker.

Start by updating the package lists:

.. code-block:: shell

	sudo apt-get update

Step 1. Add Docker's official GPG key:

.. code-block:: shell

	sudo apt-get install ca-certificates curl
	sudo install -m 0755 -d /etc/apt/keyrings
	sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
	sudo chmod a+r /etc/apt/keyrings/docker.asc

Step 2. Add the repository to Apt sources:

.. code-block:: shell

    echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

Update the package lists again:

.. code-block:: shell

    sudo apt-get update

Step 3. Install Docker:

.. code-block:: shell

	sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

Step 4. Add the user to Docker group:

.. code-block:: shell

	sudo groupadd docker
	sudo usermod -aG docker $USER

Step 5. Log out of your session and then log back in.

Step 6. Test your Docker installation:

.. code-block:: shell

	docker run hello-world

NVIDIA Container
----------------

Skip this section if GPU support is not needed.
If the prerequisites are not installed, refer to
:ref:`docs_gpu_prerequisites`.

Single time configuration of NVDIA container toolkit.

Step 1. Configure the repository:

.. code-block:: shell

    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

Step 2. Update and install:

.. code-block:: shell

	sudo apt-get update
	sudo apt-get install -y nvidia-docker2

Step 3. Configure Docker to use NVIDIA in rootless mode:

.. code-block:: shell

	nvidia-ctk runtime configure --runtime=docker --config=$HOME/.config/docker/daemon.json 
	systemctl --user restart docker
	sudo nvidia-ctk config --set nvidia-container-cli.no-cgroups --in-place

It may be necessary to edit the following file and 
change the value of ``no-cgroups`` to ``false``:

.. code-block:: shell

	sudo vi /etc/nvidia-container-runtime/config.toml 

Then, execute:

.. code-block:: shell

	sudo systemctl restart docker

Step 4. Test the GPU access:

.. code-block:: shell

	docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu20.04 nvidia-smi

Run Docker
----------

Create a folder where you want to save your notebooks. 
This folder will store all the examples you create in
the browser.

Open a terminal and navigate to the folder you just created.

To run Hiperwalk in Docker with CPU-only support, 
execute the following command:

.. code-block:: shell

	docker run --rm -v $(pwd):/home/jovyan/work -p 8888:8888 hiperwalk/hiperwalk:2.0.b0 

To run Hiperwalk in Docker with GPU and CPU support, 
execute the following command:

.. code-block:: shell

	docker run --rm --gpus all -v $(pwd):/home/jovyan/work -p 8888:8888 hiperwalk/hiperwalk:2.0.b0 

Open your web browser and go to the URL that appears on the screen, 
starting with ``127.0.0.1``. Alternatively, you can directly click 
on the hyperlink.

In the Jupyter environment, access the ``work/`` folder. 
All your data and notebooks will be saved in the folder
you have created above
(external to the Docker container).




Local Installation
==================

This section and the ones that follow are intended 
for developers of the Hiperwalk package. 
Before proceeding, it is advisable to update and 
upgrade your Ubuntu packages. 
Execute the following commands:

.. code-block:: shell

   sudo apt update
   sudo apt upgrade

The steps described here will cover identifying the GPU, 
installing the GPU drivers, hiperblas-core, 
hiperblas-opencl-bridge, pyhiperblas, and
all necessary Python libraries.
Next, run the following commands to install the prerequisites:

.. code-block:: shell

   sudo apt install git
   sudo apt install g++
   sudo apt install cmake
   sudo apt install libgtest-dev
   sudo apt install python3-distutils
   sudo apt install python3-pip
   pip install pytest


These newly installed programs serve the following purposes:

* git: used to download hiperblas-core, hiperblas-opencl-bridge,
  pyhiperblas, and hiperwalk;
* g++: used for compiling hiperblas-core, and hiperblas-opencl-bridge;
* cmake: essential for compiling hiperblas-core, hiperblas-opencl-bridge;
* libgtest-dev: verifies the successful installation of
  hiperblas-core, and hiperblas-opencl-bridge;
* python3-distutils: aids in the installation of pyhiperblas;
* python3-pip: necessary for installing Python libraries;
* pytest: helps test pyhiperblas.

Although it's not essential, we **recommend** installing FFmpeg,
which is used for generating animations.

.. code-block:: shell

   sudo apt install ffmpeg

NVIDIA Toolkit
--------------

Skip this section if the :ref:`docs_gpu_prerequisites` are not installed.

Once the GPU drivers have been successfully installed, it's
necessary to install the NVIDIA Toolkit, allowing hiperblas-core
to use CUDA.
To do this, access
`CUDA toolkit Downloads
<https://developer.nvidia.com/cuda-downloads>`_
and select the options of
*Operating System*, *Architecture*, *Distribution*, and
*Version*,  according to your machine,
and the desired *Installer Type*.
Then, follow the instructions of the **Base Installer** section.

To verify the correct installation of the NVIDIA Toolkit,
you can check if the ``nvcc`` compiler has been installed.
This can be simply done by running the following command:

.. code-block:: shell

   nvcc --version


Hiperblas
---------

For HPC support,
Hiperwalk uses
`hiperblas-core <https://github.com/hiperblas/hiperblas-core>`_,
`hiperblas-opencl-bridge
<https://github.com/hiperblas/hiperblas-opencl-bridge>`_,
and `pyhiperblas <https://github.com/hiperblas/pyhiperblas>`_.

The information in this guide is compiled from
`Paulo Motta's blog
<https://paulomotta.pro.br/wp/2021/05/01/pyhiperblas-and-hiperblas-core/>`_,
`hiperblas-core github <https://github.com/hiperblas/hiperblas-core>`_,
and `pyhiperblas github <https://github.com/hiperblas/pyhiperblas>`_.

It is **strongly recommended** that hiperblas-core,
hiperblas-opencl-bridge, and pyhiperblas
are installed (i.e. cloned) in the same directory.
In this guide, we will install both projects into the home directory.
In Linux, the tilde (``~``) serves as an alias for the home directory.

hiperblas-core
**************

Firstly, clone the repository in the home directory.

.. code-block:: shell

   cd ~
   git clone https://github.com/hiperblas/hiperblas-core.git

Next, navigate to the hiperblas-core directory to compile and
install the code.

.. code-block:: shell

   cd ~/hiperblas-core
   cmake .
   make
   sudo make install
   sudo ldconfig

The ``ldconfig`` command creates a link for the newly installed hiperblas-core,
making it accessible for use by pyhiperblas.
Before moving forward, **reboot** your computer to
ensure that the ``ldconfig`` command takes effect.

After rebboting,
run the following ``ln`` command to create
a symbolic link to another directory.

.. code-block:: shell

   sudo ln -s /usr/local/lib /usr/local/lib64

To verify the successful installation of hiperblas-core,
execute the ``vector_test`` and ``matrix_test`` tests.

.. code-block:: shell

   cd ~/hiperblas-core
   ./vector_test
   ./matrix_test

hiperblas-opencl-bridge
***********************

Skip this section if the :ref:`docs_gpu_prerequisites` are not installed.

The installation of the hiperblas-opencl-bridge is very similar to
the installation of hiperblas-core.
To install hiperblas-opencl-bridge,
first clone the repository into
**the same directory hiperblas-core was cloned**.
In this guide, we cloned hiperblas-core into the home directory.

.. code-block:: shell

   cd ~
   git clone https://github.com/hiperblas/hiperblas-opencl-bridge.git

Now, enter the new ``hiperblas-opencl-bridge`` directory to compile and
install the code.

.. code-block:: shell

   cd hiperblas-opencl-bridge
   cmake .
   make
   sudo make install

To verify the succesful installation of hiperblas-opencl-bridge,
execute the tests

.. code-block:: shell

   ./vector_test
   ./matrix_test

pyhiperblas
***********

To install pyhiperblas, first clone the repository into
**the same directory hiperblas-core was cloned**.
In this guide, we cloned hiperblas-core into the home directory.
Thus, execute:

.. code-block:: shell

   cd ~
   git clone https://github.com/hiperblas/pyhiperblas.git

Before installing ``pyhiperblas``,
install ``numpy`` using the ``sudo`` command.

.. code-block:: shell

    sudo pip install numpy

Next, navigate to the newly created ``pyhiperblas`` directory to install it.

.. code-block:: shell

   cd pyhiperblas
   sudo python3 setup.py install

To verify whether the installation was successful, run the following test:

.. code-block:: shell

   python3 test.py

Hiperwalk
---------

To finish the local hiperwalk installation,
issue the same commands of the
:ref:`docs_basic_installation` section.
