import hiperwalk as hpw
import numpy as np
#hiperwalkimport hiperblas as nbl

#nbl.init_engine(nbl.CPU,0)

import time
import sys
#sys.stdout.reconfigure(line_buffering=False, write_through=False)
sys.stdout.reconfigure(line_buffering=True)

dim = 16 + 6 - 3   ; startStep=1; endStep=startStep+500; step=1 #10*300//1-1

aRange=(startStep,endStep,step)

coin="G" # Grover  para Real
coin="F" # Fourier para Complex  para Real

myOption=None
myOption="cpu"

dim      =aDIM        # 10  
coin     =aCoin       # "G" Grover para Real e  "F"  Fourier para Complex
myOption =aHPCoPTION  # None   "cpu"    "gpu"

coinT= "Grover  coin,    real" if coin=="G" else "Fourier coin, complex"

num_vert = 1 << dim; num_arcs = dim*num_vert

from warnings import warn
def main():
    hpw.set_hpc(myOption)
    print(" hpw.get_hpc() = ",  hpw.get_hpc())
    algebra="SciPy"
    if  hpw.get_hpc() == "cpu" :
        algebra="HiperBlas"
    
    inicioG = time.perf_counter()
    g = hpw.Hypercube(dim)
    fimG    = time.perf_counter()
#    print(f"Hypercube: Tempo decorrido: {fimG - inicioG:.6f} segundos", file=sys.stderr)

    inicioC = time.perf_counter()
    qw = hpw.Coined(g, coin=coin) 
    fimC    = time.perf_counter()
#    print(f"computeU : Tempo decorrido: {fimC - inicioC:.6f} segundos", file=sys.stderr)
    U = qw.get_evolution(); densidade=num_arcs/U.nnz

    initialState = qw.state([[1, i] for i in range(dim)])
    np.set_printoptions(threshold=10)
    print("initialState = ", np.array(initialState),  end="; ")
    print("state.l2Norm=", np.linalg.norm(initialState));

    inicioS = time.perf_counter()
    for r in range(1): #50*1000*1000):
       states = qw.simulate(range=aRange, state=initialState)
    fimS    = time.perf_counter()

    print(f"Hypercube: Tempo decorrido: {fimG - inicioG:.6f} segundos", file=sys.stderr)
    print(f"computeU : Tempo decorrido: {fimC - inicioC:.6f} segundos", file=sys.stderr)
    print(f"Iteracoes: Tempo decorrido: {fimS - inicioS:.6f} segundos", file=sys.stderr)

    import os

    print(
    f"Hypercube, dim = {dim:4d}, "
    f"numStep = {endStep - startStep:4d}, "
    f"{coinT}, "
    f"numArcs = {num_arcs:10d}, "
    f"nnz = {U.nnz:12d}, "
    f"densidade = {densidade:.5e}, "
    f"algebra = {algebra:>10s}, "
    f"OMP_NUM_THREADS = {os.getenv('OMP_NUM_THREADS') or 'ND':>3s}, "
    f"tempo computeU = {fimC - inicioC:.5e}, "
    f"tempo Iteracoes = {(fimS - inicioS) / (endStep - startStep):.5e}")


    print('\n')
    return
    probs = qw.probability_distribution(states)

    #hpw.plot_probability_distribution(probs, graph=g)
    #print(probs)
    #plt.savefig("grafico.png")

main()
