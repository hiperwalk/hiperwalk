
expDir="./examples/coined"
expFileModel="$expDir/aHypercubeExp"
expFileModel="aHypercubeExp"

DIMs=( 12 14 16 18 )
DIMs=(8 10)
numThreads=(16 1)

function simulations () {
   for C in \"G\" \"F\"; do
     for D in "${DIMs[@]}"; do
	T=1
	d=$D; if [[ $D -lt 10 ]]; then d=0${D}; fi
        nome="${expFileModel}_${d}_coin${C}_SP_0${T}T"
	comando="sed 's/aDIM/$D/; s/aCoin/$C/; s/aHPCoPTION/None/' ./examples/coined/hypercubeStencil.py > $expDir/${nome}.py"
	echo $comando; eval $comando
	comando="(cd ../hiperwalk; OMP_NUM_THREADS=$T stdbuf -oL time python3 -u $expDir/${nome}.py 2>&1) |grep \": print_vectorT\|_simul_vec_out=\|Tempo\|lge\|initial\|algebra\|Arcs\|elapsed\" --color=always |tee telA${nome}.txt " 
	echo $comando; eval $comando

	for T in "${numThreads[@]}" ; do 
            if [[ $T -lt 10 ]]; then T=0${T}; fi 
            nome="${expFileModel}_${d}_coin${C}_HB_${T}T";
	    comando="sed 's/aDIM/$D/; s/aCoin/$C/; s/aHPCoPTION/\"cpu\"/' ./examples/coined/hypercubeStencil.py > $expDir/${nome}.py"
	    echo $comando; eval $comando
	    comando="(cd ../hiperwalk; OMP_NUM_THREADS=$T stdbuf -oL time python3 -u $expDir/${nome}.py 2>&1) |grep \": print_vectorT\|_simul_vec_out=\|Tempo\|lge\|initial\|algebra\|Arcs\|elapsed\" --color=always |tee telA${nome}.txt " 
	    echo $comando; eval $comando
	 done # for T
       done # for D
    done # for C
} # function simulations () {

function filtros() {
	grep "algebra" telAaHypercubeExp_*_coinG_SP_*T.txt
	grep "algebra" telAaHypercubeExp_*_coinG_HB_*T.txt

	grep "algebra" telAaHypercubeExp_*_coinF_SP_*T.txt
	grep "algebra" telAaHypercubeExp_*_coinF_HB_*T.txt
} # function filtros() {

simulations 
filtros
