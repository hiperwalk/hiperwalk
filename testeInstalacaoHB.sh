#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Configuração geral
# ============================================================

ROOT_DIR="$(pwd)"
BASE_DIR="$ROOT_DIR/testeInstalacaoHB"
BASE_DIR="$ROOT_DIR/"
PREFIX="$BASE_DIR/hiperblas"

SRC_DIR="$ROOT_DIR/hiperblas-core"
BUILD_DIR="$SRC_DIR/build"

echo "==> ROOT_DIR   = $ROOT_DIR"
echo "==> BASE_DIR   = $BASE_DIR"
echo "==> PREFIX     = $PREFIX"
echo "==> SRC_DIR    = $SRC_DIR"
echo "==> BUILD_DIR  = $BUILD_DIR"

# ============================================================
# Limpeza TOTAL (evita conflitos de CMakeCache)
# ============================================================

echo
echo "==> Limpando cache antigo do CMake (source)"
rm -f  "$SRC_DIR/CMakeCache.txt"
rm -rf "$SRC_DIR/CMakeFiles"

echo
echo "==> Limpando build anterior"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

echo
echo "==> Limpando instalação anterior"
rm -rf "$PREFIX"
mkdir -p "$PREFIX/lib" "$PREFIX/include"

# ============================================================
# Configuração CMake (OUT-OF-SOURCE, SEMPRE)
# ============================================================

echo
echo "==> Configurando CMake"

cd "$BUILD_DIR"

cmake "$SRC_DIR" \
  -DCMAKE_BUILD_TYPE=Realease \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_INSTALL_RPATH="$PREFIX/lib" \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON

# ============================================================
# Build
# ============================================================

echo
echo "==> Compilando"
make -j"$(nproc)"

# ============================================================
# Instalação LOCAL
# ============================================================

echo
echo "==> Instalando em $PREFIX"
make install

# ============================================================
# Ambiente de execução
# ============================================================

export LD_LIBRARY_PATH="$PREFIX/lib"

echo
echo "==> LD_LIBRARY_PATH configurado:"
echo "    $LD_LIBRARY_PATH"

# ============================================================
# Diagnóstico de bibliotecas
# ============================================================

echo
echo "==> Verificando dependências do plugin:"
ldd "$PREFIX/lib/libhiperblas-cpu-bridge.so" || true

echo
echo "==> Verificando símbolos esperados:"
nm -D "$PREFIX/lib/libhiperblas-core.so" | grep addVectorF || true

# ============================================================
# Testes (reproduz o SEGFAULT)
# ============================================================

echo
echo "==> Rodando testes com CTest"
ctest --output-on-failure || true

# ============================================================
# Instruções para debug manual
# ============================================================

echo
echo "============================================================"
echo " Para debug interativo:"
echo
echo " cd $BUILD_DIR"
echo " export LD_LIBRARY_PATH=$PREFIX/lib"
echo " gdb ./sparse_matrix_test"
echo
echo " Dentro do gdb:"
echo "   run"
echo "   bt"
echo "============================================================"


