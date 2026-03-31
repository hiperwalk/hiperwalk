#!/bin/bash
set -e

echo ">> instalarHB.sh (modo correto com build isolado)"

prefix=${1:-$HOME/local}
buildDir=build
buildDir=.

echo ">> prefix=$prefix"

# ==============================
# Ambiente
# ==============================

export LD_LIBRARY_PATH="$prefix/lib:$LD_LIBRARY_PATH"
export PATH="$prefix/bin:$PATH"

# GoogleTest (usuário)
export GTest_ROOT="$HOME/local"

# ==============================
# Limpeza correta
# ==============================

echo ">> Limpando build antigo"
#rm -rf "$buildDir"

# 🔴 IMPORTANTE: limpar lixo da raiz
rm -rf CMakeFiles CMakeCache.txt Makefile

# ==============================
# Build isolado
# ==============================

mkdir -p "$buildDir"
cd "$buildDir"

echo ">> Rodando CMake"
cmake \
  -DCMAKE_INSTALL_PREFIX="$prefix" \
  -DCMAKE_PREFIX_PATH="$HOME/local" \
  .

echo ">> Compilando"
make -j"$(nproc)"

echo ">> Instalando"
make install

# ==============================
# Pós-instalação
# ==============================

echo ">> Verificando libs"
ls -lh "$prefix/lib" || true

echo ">> Dependências da lib:"
ldd "$prefix/lib/libhiperblas-core.so" || true

echo ">> OK"
