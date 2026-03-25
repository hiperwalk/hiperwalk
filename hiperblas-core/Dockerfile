FROM ubuntu:20.04

LABEL maintainer="prmottajr@gmail.com"
LABEL version="1.1.0"
LABEL description="This is custom Docker Image for the hiperblas-core library."

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update

RUN apt install -y git gcc g++ libgtest-dev cmake vim libunwind8 && \
    rm -rf /var/lib/apt/lists/* && \
    apt clean

RUN cd /tmp && \
    git clone https://github.com/paulomotta/hiperblas-core && \
    cd hiperblas-core && \
    cmake . && \
    make && \
    make install && \
    ldconfig 

RUN ln -s /usr/local/lib /usr/local/lib64

# COPY DrMemory-Linux-2.5.0 /tmp/DrMemory-Linux-2.5.0

# COPY src /tmp/hiperblas-core/src
# COPY include /tmp/hiperblas-core/include
# COPY test /tmp/hiperblas-core/test
# COPY CMakeLists.txt /tmp/hiperblas-core/CMakeLists.txt

# RUN cd /tmp && \
#     cd hiperblas-core && \
#     cmake . && \
#     make && \
#     make install && \
#     ldconfig 

VOLUME ["/src"]