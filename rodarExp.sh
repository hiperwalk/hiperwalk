#!/bin/bash
expDir="./examples/coined"
expFileModel="$expDir/aHypercubeExp"
expFileModel="hypercube"
expFileModel="hypercube"; DIMs=( 18 16 14 12 ); numThreads=(16 8 4 1)
expFileModel="diagonal-grid"; DIMs=(100 200 400 800); numThreads=(16 8 4 1); 
expFileModel="grovers-algorithm"; DIMs=(047); numThreads=(1)
expFileModel="grovers-algorithm"; DIMs=(600 800); numThreads=(16 8 4 1)
expFileModel="grovers-algorithm"; DIMs=(047); numThreads=(16 8 4 1)
expFileModel=${1:-"grovers-algorithm"}; DIMs=(047 084 148 256 440 600); numThreads=(16 8 4 1)
expFileModel=${1:-"grovers-algorithm"}; DIMs=(07); numThreads=(1)
COINs=(\"F\")
COINs=(\"G\" \"F\")

function simulations () {
   for C in "${COINs[@]}"; do
     for D in "${DIMs[@]}"; do
	T=1
	d=$((10#$D))
	#d=$D; if [[ $D -lt 10 ]]; then d=0${D}; fi
        nome="a${expFileModel}_${D}_coin${C}_SP_0${T}T"
	comando="sed 's/=aDim/=$d/; s/=aCoin/=$C/; s/=aHPCoPTION/=None/' $expDir/${expFileModel}Stencil.py > $expDir/${nome}.py"
	echo $comando; eval $comando
	comando="(cd ../hiperwalk; OMP_NUM_THREADS=$T stdbuf -oL time python3 -u $expDir/${nome}.py 2>&1) |grep \": print_vectorT\|_simul_vec_out=\|Tempo\|lge\|initial\|algebra\|Arcs\|elapsed\" --color=always |tee telA${nome}.txt " 
	echo $comando; eval $comando

	for T in "${numThreads[@]}" ; do 
            if [[ $T -lt 10 ]]; then T=0${T}; fi 
            nome="a${expFileModel}_${D}_coin${C}_HB_${T}T";
	    comando="sed 's/=aDim/=$d/; s/=aCoin/=$C/; s/=aHPCoPTION/=\"cpu\"/' $expDir/${expFileModel}Stencil.py > $expDir/${nome}.py"
	    echo $comando; eval $comando
	    comando="(cd ../hiperwalk; OMP_NUM_THREADS=$T stdbuf -oL time python3 -u $expDir/${nome}.py 2>&1) |grep \": print_vectorT\|_simul_vec_out=\|Tempo\|lge\|initial\|algebra\|Arcs\|elapsed\" --color=always |tee telA${nome}.txt " 
	    echo $comando; eval $comando
	 done # for T
       done # for D
    done # for C
} # function simulations () {

function filtros() {
	comando="grep "algebra" telA$nome*"
	echo $comando; eval $comando
	return
	grep "algebra" telAaHypercubeExp_*_coinG_HB_*T.txt

	grep "algebra" telAaHypercubeExp_*_coinF_SP_*T.txt
	grep "algebra" telAaHypercubeExp_*_coinF_HB_*T.txt
} # function filtros() {

simulations 
filtros
