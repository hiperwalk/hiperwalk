relativeDir=${1:-"./"}
SCRATCH="${SCRATCH:-$HOME}"
comandoInstallC="if [ ! -f "/snap/bin/cmake" ] ; then sudo snap install cmake --classic; fi"
echo $comandoInstallC
eval $comandoInstallC
comandoInstallC="export PATH=/snap/bin:\"$PATH\""
echo $comandoInstallC
eval $comandoInstallC
echo PATH=$PATH; which cmake; #exit
comandoInstallG="sudo apt install -y libgtest-dev; (cd /usr/src/gtest; sudo cmake .; sudo make -j$(nproc); sudo cp lib/*.a /usr/lib)"
comandoInstallG="mkdir -p $HOME/local/src; (cd $HOME/local/src; git clone https://github.com/google/googletest.git; cd googletest; mkdir build && cd build; cmake -DCMAKE_INSTALL_PREFIX=$HOME/local ..; make -j4; make install;) "

PREFIX="$HOME/local"
if [ ! -f "$PREFIX/lib/libgtest.a" ]; then
   echo $comandoInstallG
   eval $comandoInstallG
else
   echo GTEST instalado em: $PREFIX
fi

# xGoogleTest paths
comandoSetG="
export CPLUS_INCLUDE_PATH="$HOME/local/include:$CPLUS_INCLUDE_PATH"
export LIBRARY_PATH="$HOME/local/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$HOME/local/lib:$LD_LIBRARY_PATH"
"
# uso do gtest
# g++ -std=c++17 test.cpp -lgtest -lgtest_main -pthread -o test
## Ou especificando o prefixo manualmente:
#g++ -I$HOME/local/include -L$HOME/local/lib test.cpp -lgtest -lgtest_main -pthread -o test


export PYTHONUNBUFFERED=1
#export LD_LIBRARY_PATH=/home/bidu/hiperblas/lib:$LD_LIBRARY_PATH
echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH
comandoLD="export LD_LIBRARY_PATH=$HOME/hiperblas/lib:$SCRATCH/hiperblas/lib:$LD_LIBRARY_PATH"
comandoLD="echo \":\$LD_LIBRARY_PATH:\" | grep -q \":\$HOME/hiperblas/lib:\" || export LD_LIBRARY_PATH=\"\$HOME/hiperblas/lib:\$LD_LIBRARY_PATH\" "
echo $comandoLD;  eval $comandoLD;
comandoLD="echo \":\$LD_LIBRARY_PATH:\" | grep -q \":\$SCRATCH/hiperblas/lib:\" || export LD_LIBRARY_PATH=\"\$SCRATCH/hiperblas/lib:\$LD_LIBRARY_PATH\" "
echo $comandoLD;  eval $comandoLD;
echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH
read -p " aguardando um enter para seguir ou um ctrl+C para interromper ...."
comandoHB="(pwd; cd $relativeDir/hiperblas-core; pwd; make install   )"; 
comandoHB="(pwd; cd $relativeDir/hiperblas-core; pwd; ./instalarHB.sh     )"; 
echo $comandoHB
comandoPyHB="(cd $relativeDir/pyhiperblas;    pwd; ./instalarPyHB.sh . )";
comandoPyHB="(cd $relativeDir/pyhiperblas; python3 -m pip install --user -e . --break-system-packages --no-deps --no-cache )"
comandoPyHB="(pwd; cd $relativeDir/pyhiperblas;    pwd; pip install  .  )";
echo $comandoPyHB
comandoHW="(pwd; cd $relativeDir;  pip install -e .  )"; # só funciona se o backend suportar instalação editável.
comandoHW="(pwd; cd $relativeDir;  pip install    .  )"; #  se o backend NAO suportar instalação editável.
echo $comandoHW;
comandoR1="(pwd; cd $relativeDir; pwd;  OMP_NUM_THREADS=1 stdbuf -oL time python3 -u examples/coined/diagonal-grid.py 2>&1) " ; 
comandoR2="(pwd; cd $relativeDir; pwd;  OMP_NUM_THREADS=1 stdbuf -oL time python3 -u examples/coined/hypercube.py     2>&1) " ;
echo " os comando acima serão executados"
echo " os comandos são possiveis comandos para exeucão de exemplos, não serão executados"
echo $comandoR1; echo $comandoR2
read -p " aguardando um enter para seguir ou um ctrl+C para interromper ...."

eval $comandoHB
eval $comandoPyHB
eval $comandoHW

echo "foram executados os seguintes comandos:"
echo $comandoLD; echo $comandoHB; echo $comandoPyHB; echo $comandoHW;
echo "linha de execuão possivel:"
echo $comandoR1
echo $comandoR2
