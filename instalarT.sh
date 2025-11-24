relativeDir=${1:-"./"}
export PYTHONUNBUFFERED=1
#export LD_LIBRARY_PATH=/home/bidu/hiperblas/lib:$LD_LIBRARY_PATH
echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH
comandoA="export LD_LIBRARY_PATH=$HOME/hiperblas/lib:$SCRATCH/hiperblas/lib:$LD_LIBRARY_PATH"
echo $comandoA; 
comandoB="(cd $relativeDir/hiperblas-core; . ./instalarHB.sh     )"; echo $comandoB;
comandoC="(cd $relativeDir/pyhiperblas;    . ./instalarPyHB.sh . )"; echo $comandoC;
#comandoC="(cd $relativeDir/pyhiperblas; python3 -m pip install --user -e --break-system-packages --no-deps --no-cache )"
comandoD="(cd $relativeDir; pip install -e .  --break-system-packages)"; echo $comandoD;
#comandoE="(cd $relativeDir; OMP_NUM_THREADS=1 stdbuf -oL time python3 -u examples/coined/diagonal-grid.py 2>&1) " ; echo $comandoE
comandoE="(cd $relativeDir; OMP_NUM_THREADS=1 stdbuf -oL time python3 -u examples/coined/hypercube.py 2>&1) " ; echo $comandoE
echo " os comando acima serão executados"
read -p " aguardando um enter para segui ou um ctrl+C para interromper ...."
eval $comandoA; 
eval $comandoB
eval $comandoC
eval $comandoD
#eval $comandoE
