#!/bin/bash
set -e

relativeDir="${1:-./}"
SCRATCH="${SCRATCH:-$HOME}"
PREFIX="$HOME/local"

# --------------------------------------------------
# 🔧 Funções utilitárias
# --------------------------------------------------
add_ld_path () {
    case ":$LD_LIBRARY_PATH:" in
        *":$1:"*) ;;
        *) export LD_LIBRARY_PATH="$1:$LD_LIBRARY_PATH" ;;
    esac
}

instalar_se_faltar () {
    for pkg in "$@"; do
        dpkg -s "$pkg" &> /dev/null || {
            echo ">> Instalando $pkg"
            sudo apt-get install -y "$pkg"
        }
    done
}

# --------------------------------------------------
# 🚀 Setup inicial
# --------------------------------------------------
setUpInicial() {
    echo ">> Setup inicial"

    sudo -v
    ( while true; do sudo -n true; sleep 60; done ) 2>/dev/null &

    echo ">> Atualizando pacotes (uma vez só)"
    sudo apt-get update -y

    # limpar builds antigos
    find . -name build -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name CMakeFiles -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name CMakeCache.txt -delete 2>/dev/null || true

    # cmake
    if command -v cmake >/dev/null 2>&1; then
        if which cmake | grep -q snap; then
            echo ">> Removendo cmake do snap"
            sudo snap remove cmake
            instalar_se_faltar cmake
        else
            echo ">> cmake OK"
        fi
    else
        instalar_se_faltar cmake
    fi

    # toolchain
    instalar_se_faltar build-essential python3-pip git pkg-config

    # pytest
    echo ">> Verificando pytest"
    python3 -c "import pytest" &> /dev/null || {
        echo ">> Instalando pytest"
        python3 -m pip install --user pytest --break-system-packages
    }

    # teste compilador
    echo 'int main(){return 0;}' > a.cpp
    g++ a.cpp -o a.out && ./a.out
    rm -f a.cpp a.out

    # GoogleTest
    if [ ! -f "$PREFIX/lib/libgtest.a" ]; then
        echo ">> Instalando GoogleTest"
        mkdir -p "$PREFIX/src"
        cd "$PREFIX/src"

        [ ! -d googletest ] && git clone https://github.com/google/googletest.git

        cd googletest
        rm -rf build
        mkdir build && cd build

        cmake -DCMAKE_INSTALL_PREFIX="$PREFIX" ..
        make -j"$(nproc)"
        make install
    else
        echo ">> GoogleTest já instalado"
    fi

    # variáveis ambiente
    export CPLUS_INCLUDE_PATH="$PREFIX/include:$CPLUS_INCLUDE_PATH"
    export LIBRARY_PATH="$PREFIX/lib:$LIBRARY_PATH"
    export CMAKE_PREFIX_PATH="$PREFIX:$CMAKE_PREFIX_PATH"

    add_ld_path "$PREFIX/lib"
    add_ld_path "$HOME/hiperblas/lib"
    add_ld_path "$SCRATCH/hiperblas/lib"

    # PYTHONPATH (ESSENCIAL pro hiperwalk)
    export PYTHONPATH="$relativeDir:$PYTHONPATH"

    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    echo "PYTHONPATH=$PYTHONPATH"
}

# --------------------------------------------------
# ⚙️ HiperBLAS core
# --------------------------------------------------
instalarHB_core() {
    echo ">> Instalando hiperblas-core"
    cd "$relativeDir/hiperblas-core"

    mkdir -p build
    cd build

    cmake .. -DCMAKE_INSTALL_PREFIX="$PREFIX"
    make -j$(nproc)
    make install
}

# --------------------------------------------------
# 🐍 Python bindings
# --------------------------------------------------
instalarPyHB() {
    echo ">> Instalando pyhiperblas"
    cd "$relativeDir/pyhiperblas"


    export HIPERBLAS_PREFIX="$HOME/local"

    export C_INCLUDE_PATH="$HIPERBLAS_PREFIX/include:$C_INCLUDE_PATH"
    export LIBRARY_PATH="$HIPERBLAS_PREFIX/lib:$LIBRARY_PATH"
    export LD_LIBRARY_PATH="$HIPERBLAS_PREFIX/lib:$LD_LIBRARY_PATH"

    python3 -m pip install --user -e . --break-system-packages
}

# --------------------------------------------------
# 📦 Hiperwalk
# --------------------------------------------------
instalarHW() {
    echo ">> Instalando hiperwalk"
    cd "$relativeDir"

    python3 -m pip install --user -e . --break-system-packages
}

# --------------------------------------------------
# 🧪 Experimentos
# --------------------------------------------------
experimentosIniciais() {

    comandoA='./hiperblas-core/build/sparseMatrixTest | grep "state_index\|_simul_vec_\|500\|new" || true'
    comandoB='python3 pyhiperblas/sparse_matrix_testDiagGridExp.py | grep "state_index\|_simul_vec_\|500\|new" || true'
    comandoC='python3 examples/coined/diagonal-grid.py | grep "state_index\|_simul_vec_\|500\|new" || true'

    echo $comandoA
    echo $comandoB
    echo $comandoC
    echo Os 3 programas acima usam os mesmos dados: diagonal-grid, n = 3
    read -p ">> Enter para rodar experimento"

    echo ">> Teste C"
    echo $comandoA
    read -p ">> Enter para rodar experimento"
    bash -c "$comandoA"

    echo ">> Teste Python"
    echo $comandoB
    read -p ">> Enter para rodar experimento"
    bash -c "$comandoB"

    echo ">> Exemplo"
    echo $comandoC
    read -p ">> Enter para rodar experimento"
    bash -c "$comandoC"
}

# --------------------------------------------------
# ▶️ Execução
# --------------------------------------------------
main() {

    cdWork=$PWD

    setUpInicial

    cd $cdWork
    read -p ">> Enter para instalar hiperblas-core"
    instalarHB_core

    cd $cdWork
    read -p ">> Enter para instalar pyhiperblas"
    instalarPyHB

    cd $cdWork
    read -p ">> Enter para instalar hiperwalk"
    instalarHW

    cd $cdWork
    read -p ">> Enter para rodar experimentos"
    experimentosIniciais

    echo ">> Finalizado com sucesso"
}

main

