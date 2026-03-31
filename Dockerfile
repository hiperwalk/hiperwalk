# ============================================================
# BASE
# ============================================================
FROM jupyter/minimal-notebook AS base

USER root

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

# usa o usuário padrão da imagem
ENV HOME=/home/jovyan
WORKDIR ${HOME}

# ============================================================
# DEV (build do hiperblas)
# ============================================================
FROM base AS dev

USER root

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential \
        cmake \
        libgtest-dev \
        pkg-config \
        vim \
        less \
        gdb \
    && rm -rf /var/lib/apt/lists/*

# clone
RUN git clone --branch bidu --single-branch \
    https://github.com/hiperwalk/hiperwalk.git \
    ${HOME}/hiperwalk

#COPY ./hiperwalk ${HOME}/hiperwalk

WORKDIR ${HOME}/hiperwalk/hiperblas-core

# PREFIX
ENV PREFIX=${HOME}/local
ENV LD_LIBRARY_PATH=${PREFIX}/lib:$LD_LIBRARY_PATH

RUN mkdir -p ${PREFIX}/lib ${PREFIX}/include ${PREFIX}/bin

# build
RUN rm -rf CMakeCache.txt CMakeFiles && \
    cmake . \
      -DCMAKE_INSTALL_PREFIX=${PREFIX} \
      -DCMAKE_INSTALL_LIBDIR=lib && \
    make -j$(nproc) && \
    make install

# copia libs para pyhiperblas
#RUN cp ${PREFIX}/lib/*.so ${HOME}/hiperwalk/pyhiperblas/

ENV CPATH=${PREFIX}/include
ENV LIBRARY_PATH=${PREFIX}/lib
ENV LD_LIBRARY_PATH=${PREFIX}/lib:$LD_LIBRARY_PATH

# python (ainda como root)
WORKDIR ${HOME}/hiperwalk

RUN python -m pip install --no-cache-dir numpy scipy pytest
RUN python -m pip install -e pyhiperblas
RUN python -m pip install -e .

# remove node_modules (evita problema futuro)
RUN rm -rf ${HOME}/hiperwalk/node_modules

# garante permissão correta
RUN chown -R jovyan:users ${HOME}

# ============================================================
# NOTEBOOK (runtime final)
# ============================================================
FROM base AS notebook

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 time && \
    rm -rf /var/lib/apt/lists/*

# copia build pronto
COPY --from=dev /home/jovyan/local /home/jovyan/local
COPY --from=dev /home/jovyan/hiperwalk /home/jovyan/hiperwalk

# env
ENV PREFIX=/home/jovyan/local
ENV HIPERBLAS_PREFIX=${PREFIX}
ENV HIPERBLAS_PLUGIN=${PREFIX}/lib/libhiperblas-cpu-bridge.so
ENV LD_LIBRARY_PATH=${PREFIX}/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH=/home/jovyan/hiperwalk/pyhiperblas:$PYTHONPATH

# python runtime
RUN python -m pip install --no-cache-dir numpy scipy networkx matplotlib
RUN python -m pip install -e /home/jovyan/hiperwalk

# 🔥 remove LSP (resolve seus erros)
RUN python -m pip uninstall -y jupyter-lsp

# 🔥 garante que NÃO sobra node_modules
RUN rm -rf /home/jovyan/hiperwalk/node_modules

# 🔥 permissões finais corretas
RUN chown -R jovyan:users /home/jovyan

USER jovyan
WORKDIR /home/jovyan/hiperwalk

CMD ["start-notebook.py"]
