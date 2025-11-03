relativeDir=../
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/home/bidu/hiperblas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/hiperblas/lib:$LD_LIBRARY_PATH
echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH
comando="(cd $relativeDir/hiperblas-core/;. ./instalarHB.sh)"; echo $comando;
comando="(cd $relativeDir/pyhiperblas/; ./instalarPyHB.sh)"; echo $comando;
comando="(cd $relativeDir/hiperwalk/; pip install -e .  --break-system-packages)"; echo $comando;
comando="(cd $relativeDir/hiperwalk; OMP_NUM_THREADS=1 stdbuf -oL time python3 -u examples/coined/diagonal-grid.py 2>&1) " ; echo $comando
export OMP_NUM_THREADS=4
read -p "aguardando um ok para continuar"
comando="(cd $relativeDir/hiperblas-core/;. ./instalarHB.sh)"
echo $comando;
read -p "aguardando um ok para continuar"
eval $comando
comando="python3 -c 'import importlib.util; print(\"hiperblas ->\", importlib.util.find_spec(\"hiperblas\"))'"
echo $comando;
read -p "aguardando um ok para continuar"
eval $comando
pwd
comando="(cd $relativeDir/pyhiperblas/; ./instalarPyHB.sh)"
echo $comando;
read -p "aguardando um ok para continuar"
eval $comando
comando="(cd $relativeDir/hiperwalk/; pip install -e .  --break-system-packages)"
echo $comando;
read -p "aguardando um ok para continuar"
eval $comando
comando="(cd $relativeDir/hiperwalk; OMP_NUM_THREADS=1 stdbuf -oL time python3 -u examples/coined/diagonal-grid.py 2>&1)  |grep -v \"BD\""
comando="(cd $relativeDir/hiperwalk; OMP_NUM_THREADS=1 stdbuf -oL time python3 -u examples/coined/diagonal-grid.py 2>&1) " 
echo $comando;
read -p "aguardando um ok para continuar"
eval $comando
comando="(cd $relativeDir/hiperwalk; python3 examples/coined/hypercube.py |grep -v \"em mat\|-core\" ) "
#echo $comando; eval $comando
