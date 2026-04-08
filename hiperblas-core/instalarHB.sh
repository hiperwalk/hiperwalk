#!/bin/bash
set -e
cd "$(dirname "$0")"

echo ">> instalarHB.sh (modo correto com build isolado)"

prefix=${1:-$HOME/local}
buildDir=build

echo ">> prefix=$prefix"

# ==============================
# Ambiente
# ==============================

export    LD_LIBRARY_PATH="$prefix/lib:$LD_LIBRARY_PATH"
export       LIBRARY_PATH="$prefix/lib:$LIBRARY_PATH"
export CPLUS_INCLUDE_PATH="$prefix/include:$CPLUS_INCLUDE_PATH"
export               PATH="$prefix/bin:$PATH"

# ==============================
# Limpeza
# ==============================

echo ">> Limpando build antigo"
rm -rf "$buildDir"

# ==============================
# Build isolado
# ==============================

cmake -B "$buildDir" -S . \
  -DCMAKE_INSTALL_PREFIX="$prefix" \
  -DCMAKE_PREFIX_PATH="$prefix" \
  -DENABLE_TESTS=ON

cmake --build "$buildDir" -j"$(nproc)"

echo ">> Instalando"
cmake --install "$buildDir"


# ==============================
# Pós-instalação
# ==============================

echo ">> Verificando libs"
ls -lh "$prefix/lib" || true

echo ">> Dependências da lib:"
ldd "$prefix/lib/libhiperblas-core.so" || true

echo ">> OK"
