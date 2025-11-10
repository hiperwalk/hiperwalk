#https://hiperwalk.org/docs/stable/install/index.html
function setingsSD(){
  module load gcc/14.2.0_sequana
  module load  autodock-gpu/4.2.6_opencl_sequana
  module load cmake/3.30.3_sequana
}
comando="export LD_LIBRARY_PATH=$HOME/hiperblas/lib:$LD_LIBRARY_PATH"
echo $comando; eval $comando
echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH
setingsSD
