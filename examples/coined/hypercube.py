import hiperwalk as hpw
import numpy as np

#nbl.init_engine(nbl.CPU,0)

import time
import sys
#sys.stdout.reconfigure(line_buffering=False, write_through=False)
sys.stdout.reconfigure(line_buffering=True)

aDim=3; aNumSteps=3; aCoin="G"; aHPCoPTION=None
aDim=3; aNumSteps=3; aCoin="G"; aHPCoPTION="cpu"
aDim=3; aNumSteps=3; aCoin="F"; aHPCoPTION=None
aDim=3; aNumSteps=3; aCoin="F"; aHPCoPTION="cpu"
#aNumSteps=15; aDim=10

dim          =aDim        # 10
coin         =aCoin       # "G" Grover para Real e  "F"  Fourier para Complex
myHPC_option =aHPCoPTION  # None   "cpu"    "gpu"
myNumSteps   =aNumSteps

coinT  = "Grover  coin,    real" if coin=="G"            else "Fourier coin, complex"
algebra="SciPy"                  if myHPC_option == None else "HiperBlas"
startStep=1;endStep=startStep+myNumSteps;step=1
aRange=(startStep,endStep,step)

from warnings import warn
def main():
    
    hpw.set_hpc(myHPC_option)

    inicioG = time.perf_counter()
    g = hpw.Hypercube(dim)
    fimG    = time.perf_counter()

    inicioC = time.perf_counter()
    qw = hpw.Coined(g, coin=coin) 
    fimC    = time.perf_counter()

    initialState = qw.state([[1, i] for i in range(dim)])

    inicioS = time.perf_counter()
    for r in range(1): #50*1000*1000):
       states = qw.simulate(aRange, state=initialState)
    fimS    = time.perf_counter()

    print(f"Hypercube: Tempo decorrido: {fimG - inicioG:.6f} segundos", file=sys.stderr)
    print(f"computeU : Tempo decorrido: {fimC - inicioC:.6f} segundos", file=sys.stderr)
    print(f"Iteracoes: Tempo decorrido: {fimS - inicioS:.6f} segundos", file=sys.stderr)
    print(f"Tempo total      decorrido: {fimS - inicioG:.6f} segundos", file=sys.stderr)


    U = qw.get_evolution(); num_arcs=U.shape[0];  densidade=U.nnz/(num_arcs*num_arcs)
    import os
    nome=os.path.splitext(os.path.basename(__file__))[0] # sem extensão
    print(
    f"{nome:14s}, "
    f"dim = {dim:4d}, "
    f"numStep = {endStep - startStep:4d}, "
    f"{coinT}, "
    f"numArcs = {num_arcs:10d}, "
    f"nnz = {U.nnz:12d}, "
    f"densidade = {densidade:.5e}, "
    f"algebra = {algebra:>10s}, "
    f"OMP_NUM_THREADS = {os.getenv('OMP_NUM_THREADS') or 'ND':>3s}, "
    f"tempo computeU = {fimC - inicioC:.5e}, "
    f"tempo Iteracoes = {(fimS - inicioS) / (endStep - startStep + 1):.5e}, "
    f"tempo total = {(fimS - inicioG) :.5e}")
    print('\n')
    probs = qw.probability_distribution(states)

    np.set_printoptions(linewidth=820, threshold=240)
    print("probs =\n", probs)
    print("np.sum(prob) = ", np.sum(probs))

    return
    hpw.plot_probability_distribution(probs, graph=g)


    #print(probs)
    #plt.savefig("grafico.png")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("Erro:", e, file=sys.stderr)
        traceback.print_exc()
