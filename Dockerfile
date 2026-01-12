
# ============================================================
# Base comum (Jupyter + Python)
# ============================================================
FROM jupyter/minimal-notebook AS base

LABEL description="Hiperwalk Quantum Walks Simulator"

ARG USER_NAME=bidu
ARG USER_ID=1001
ARG GROUP_ID=1001

USER root

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g ${GROUP_ID} ${USER_NAME} || true && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USER_NAME}

ENV HOME=/home/${USER_NAME}
WORKDIR ${HOME}

# ============================================================
# DEV (imagem completa – compila tudo)
# ============================================================
FROM base AS dev

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        pkg-config \
        gdb \
        libgtest-dev \
        vim \
        less \
        procps \
        time \
    && rm -rf /var/lib/apt/lists/*

RUN git clone \
    --branch bidu \
    --single-branch \
    https://github.com/hiperwalk/hiperwalk.git \
    ${HOME}/hiperwalk

WORKDIR ${HOME}/hiperwalk

RUN cd hiperblas-core && \
    mkdir -p build && cd build && \
    cmake .. \
      -DCMAKE_INSTALL_PREFIX=${HOME}/hiperblas \
      -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install

# após make install do hiperblas
RUN cp ${HOME}/hiperblas/lib/*.so ${HOME}/hiperwalk/pyhiperblas/


#ENV HIPERBLAS_PREFIX=${HOME}/hiperblas
#ENV LD_LIBRARY_PATH=${HIPERBLAS_PREFIX}/lib
ENV HIPERBLAS_PREFIX=/opt/hiperblas

RUN pip install --no-cache-dir numpy scipy pytest
RUN cd pyhiperblas && pip install -e .
RUN pip install -e .

USER ${USER_NAME}
CMD ["/bin/bash"]

# ============================================================
# NOTEBOOK-SLIM (só runtime)
# ============================================================
FROM base AS notebook

# copia apenas o resultado pronto
COPY --from=dev /home/bidu/hiperblas /home/bidu/hiperblas
COPY --from=dev /home/bidu/hiperwalk /home/bidu/hiperwalk

ENV HIPERBLAS_PREFIX=/home/bidu/hiperblas
ENV LD_LIBRARY_PATH=${HIPERBLAS_PREFIX}/lib

RUN pip install --no-cache-dir numpy scipy
RUN cd /home/bidu/hiperwalk/pyhiperblas && pip install .
RUN pip install /home/bidu/hiperwalk

USER ${USER_NAME}
CMD ["start-notebook.sh"]


