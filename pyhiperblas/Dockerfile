# Use the jupyter/minimal-notebook base image
FROM jupyter/minimal-notebook

LABEL maintainer="prmottajr@gmail.com"
LABEL version="2.0.b18-gpu"
LABEL description="This is custom Docker Image for the Hiperwalk Quantum Walks Simulator."

USER root

ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies for CUDA Toolkit 12 and OpenCL headers
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    gnupg \
    dirmngr \
    ocl-icd-opencl-dev \
    && rm -rf /var/lib/apt/lists/*

# Add the CUDA Toolkit 12 repository
RUN echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64 /" > /etc/apt/sources.list.d/cuda.list

# Import the NVIDIA public key using gpg
RUN gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys A4B469963BF863CC && \
    gpg --export --armor A4B469963BF863CC | apt-key add -

# Update package lists and install CUDA Toolkit 12
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    cuda-libraries-12-0 \
    cuda-toolkit-12-0 \
    clinfo \
    libnvidia-compute-550 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configure CUDA environment variables
ENV LD_LIBRARY_PATH=/home/bidu/libs/lib
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

RUN apt update

RUN apt install -y git gcc g++ libgtest-dev cmake vim libunwind8 python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    apt clean
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++

COPY hiperblas-core/src /tmp/hiperblas-core/src
COPY hiperblas-core/include /tmp/hiperblas-core/include
COPY hiperblas-core/test /tmp/hiperblas-core/test
COPY hiperblas-core/CMakeLists.txt /tmp/hiperblas-core/CMakeLists.txt

RUN ln -s /usr/local/lib /usr/local/lib64

RUN cd /tmp && \
    cd hiperblas-core && \
    cmake . && \
    make && \
    make install && \
    ldconfig \
    && ./vector_test \
    && ./matrix_test \
    && ./sparse_matrix_test \
    && ./neblina_test

COPY hiperblas-opencl-bridge/src /tmp/hiperblas-opencl-bridge/src
COPY hiperblas-opencl-bridge/include /tmp/hiperblas-opencl-bridge/include
COPY hiperblas-opencl-bridge/test /tmp/hiperblas-opencl-bridge/test
COPY hiperblas-opencl-bridge/CMakeLists.txt /tmp/hiperblas-opencl-bridge/CMakeLists.txt

RUN cd /tmp && \
    cd hiperblas-opencl-bridge && \
    cmake . && \
    make && \
    make install && \
    ldconfig

RUN pip3 install pytest
RUN pip3 install numpy==1.24.2
RUN pip3 install scipy==1.11.1
RUN pip3 install networkx==3.3
RUN pip3 install matplotlib==3.8.4

COPY pyhiperblas/setup.py /tmp/pyhiperblas/setup.py
COPY pyhiperblas/test.py /tmp/pyhiperblas/test.py
COPY pyhiperblas/neblina_wrapper.c /tmp/pyhiperblas/neblina_wrapper.c

RUN cd /tmp/pyhiperblas && \
    python3 setup.py install
 
# RUN pip install -i https://test.pypi.org/simple/ hiperwalk==2.0b10
RUN pip install hiperwalk==2.0b18

RUN fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

USER ${NB_UID}
