# ============================================================
# Hiperwalk + Hiperblas – imagem CPU
# ============================================================

FROM jupyter/minimal-notebook

LABEL description="Custom Docker Image for the Hiperwalk Quantum Walks Simulator"

# -----------------------------
# Build arguments
# -----------------------------
ARG USER_NAME=bidu
ARG USER_ID=1001
ARG GROUP_ID=1001

USER root

# -----------------------------
# Sistema base + toolchain
# -----------------------------
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        pkg-config \
        ca-certificates \
        gdb \
        libgtest-dev \
        vim \
        less \
        procps \
        time \
        x11-apps \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Usuário
# -----------------------------
RUN groupadd -g ${GROUP_ID} ${USER_NAME} || true && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USER_NAME}

ENV HOME=/home/${USER_NAME}
WORKDIR ${HOME}

# -----------------------------
# Hiperblas (C core)
# -----------------------------
RUN git clone \
    --branch bidu \
    --single-branch \
    https://github.com/hiperwalk/hiperwalk.git \
    ${HOME}/hiperwalk && \
    chown -R ${USER_NAME}:${USER_NAME} ${HOME}/hiperwalk

WORKDIR ${HOME}/hiperwalk

RUN cd hiperblas-core && \
    mkdir -p build && cd build && \
    cmake .. \
      -DCMAKE_INSTALL_PREFIX=${HOME}/hiperblas \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install

WORKDIR ${HOME}/hiperwalk

ENV HIPERBLAS_PREFIX=${HOME}/hiperblas
ENV LD_LIBRARY_PATH=${HIPERBLAS_PREFIX}/lib

# -----------------------------
# Python deps (usando pip do notebook)
# -----------------------------
RUN pip install --no-cache-dir numpy scipy pytest

RUN cd pyhiperblas && pip install -e .

RUN python -c "import hiperblas; print('hiperblas OK:', hiperblas.__file__)"

# -----------------------------
# hiperwalk
# -----------------------------

WORKDIR ${HOME}/hiperwalk
RUN pip install -e .

RUN python -c "import hiperwalk; print('hiperwalk OK:', hiperwalk.__file__)"

# -----------------------------
# Usuário final
# -----------------------------
USER ${USER_NAME}

CMD ["/bin/bash"]

