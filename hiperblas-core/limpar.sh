#!/usr/bin/env bash
set -e

# Diretório de build (ajuste se não for 'build')
BUILD_DIR="build"
BUILD_DIR="."

echo "==> Limpando arquivos de compilação..."

if [ -d "$BUILD_DIR" ]; then
    cmake --build "$BUILD_DIR" --target clean || true
    rm -rf "$BUILD_DIR"/*.gcda "$BUILD_DIR"/*.gcno
    rm -f  "$BUILD_DIR"/coverage.info
    rm -rf "$BUILD_DIR"/coverage-html
    echo "Arquivos removidos dentro de $BUILD_DIR"
else
    echo "Diretório $BUILD_DIR não encontrado. Nada para limpar."
fi

echo "==> Limpando artefatos no diretório de testes..."
rm -f test/*.out test/*.log test/*.txt || true
rm -f test/libhiperblas-core.so || true

echo "==> Limpeza completa."

