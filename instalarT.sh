relativeDir=./
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/home/bidu/hiperblas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/hiperblas/lib:$LD_LIBRARY_PATH
echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH
comandoA="(cd $relativeDir/hiperblas-core/;. ./instalarHB.sh)"; echo $comando;
eval $comandoA
comandoB="(cd $relativeDir/pyhiperblas/; ./instalarPyHB.sh $PWD)"; echo $comando;
eval $comandoB
comandoC="(cd $relativeDir; pip install -e .  --break-system-packages)"; echo $comando;
eval $comandoC
comandoD="(cd $relativeDir; OMP_NUM_THREADS=1 stdbuf -oL time python3 -u examples/coined/diagonal-grid.py 2>&1) " ; echo $comando
comandoD="(cd $relativeDir; OMP_NUM_THREADS=1 stdbuf -oL time python3 -u examples/coined/hiperblasCPLX.py 2>&1) " ; echo $comando
eval $comandoD
