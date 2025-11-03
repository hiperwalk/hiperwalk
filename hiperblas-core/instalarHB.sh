#!/bin/bash
set -e  # para parar se algo falhar

# ==============================
# Configuração
# ==============================

pwd 

# Diretório de trabalho (onde está o código-fonte)
workDir=/prj/prjedlg/bidu/OneDrive/aLncc/passeiosQuantHiago
workDir=/mnt/c/Users/bidu/OneDrive/aLncc/passeiosQuantOut25
workDir=$HOME/OneDrive/aLncc/passeiosQuantOut25A
workDir=./

# Prefixo de instalação (padrão: $HOME/hiperblas)
prefix=${1:-$HOME/hiperblas}

# Subdiretórios
hwLIB=$prefix/lib
hwINC=$prefix/include
hwBIN=$prefix/bin
echo prefix=$prefix, hwLIB=$hwLIB;
#
# ==============================
# Variáveis de ambiente
# ==============================

echo ">> Configurando variáveis de ambiente locais"
export PATH="$hwBIN:$PATH"
export LD_LIBRARY_PATH="$hwLIB:$LD_LIBRARY_PATH"

echo
echo ">> Para manter estas variáveis em sessões futuras, adicione ao seu ~/.bashrc ou ~/.zshrc:"
#echo "export PATH=\"$hwBIN:\$PATH\""
#echo "export LD_LIBRARY_PATH=\"$hwLIB:\$LD_LIBRARY_PATH\""
echo ".......LD_LIBRARY_PATH=$LD_LIBRARY_PATH"+++; 


# Variáveis para build/link
export GTEST_INCLUDE_DIR=$hwINC
export GTEST_LIBRARIES=$hwLIB

# ==============================
# Instalação
# ==============================

echo ">> Instalando Hiperwalk em: $prefix"

# Limpa instalação anterior
#echo ">> Limpando instalação antiga em $prefix"
#rm -rf "$prefix"

# Recria diretórios
echo ">> Criando diretórios $prefix"
rm -rf "$hwLIB" "$hwINC" "$hwBIN"
mkdir -p "$hwLIB" "$hwINC" "$hwBIN"

# Compilação e instalação
(
  #cd "$prefix"
#pwd; exit
  #cd "$workDir/hiperblas-core"
  make clean || true
  rm -rf CMakeCache.txt CMakeFiles
  comando="cmake \
    -DGTEST_INCLUDE_DIR="$GTEST_INCLUDE_DIR" \
    -DCMAKE_INSTALL_PREFIX="$prefix" \
    ."
  #echo $comando; 
  #read -p "cmake ..";
  echo $comando; eval $comando

  comando="make -j$(nproc)"
  echo $comando; eval $comando
  comando="make install"
  echo $comando; eval $comando
)

# Link lib64 → lib (se necessário)
if [ ! -d "$prefix/lib64" ]; then
  echo ">> Criando link simbólico lib64 → lib"
  ln -s "$hwLIB" "$prefix/lib64"
fi


# ==============================
# Validação
# ==============================

echo ">> Bibliotecas instaladas em $hwLIB:"
ls -l "$hwLIB" || true
echo
echo ">> Link simbólico $prefix/lib64:"
ls -l "$prefix/lib64" || true
