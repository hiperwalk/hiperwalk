import hiperwalk as hpw
import numpy as np
import networkx as nx

import time
import sys
#sys.stdout.reconfigure(line_buffering=False, write_through=False)
sys.stdout.reconfigure(line_buffering=True)

N = 47
numSteps=1000
coin="F" # Fourier para Complex  para Real
coin="G" # Grover  para Real
myOption=None
myOption="cpu"

aDim=N; aNumSteps=numSteps; aCoin=coin; aHPCoPTION=myOption

myDim    =aDim        # 10
myCoin   =aCoin       # "G" Grover para Real e  "F"  Fourier para Complex
myOption =aHPCoPTION  # None   "cpu"    "gpu"
myNumSteps=aNumSteps
step=1; startStep=1; endStep=startStep+myNumSteps; 
coinT  = "Grover  coin,    real" if myCoin=="G"      else "Fourier coin, complex"
algebra= "SciPy"                 if myOption == None else "HiperBlas"

from warnings import warn
def main():

    hpw.set_hpc(myOption)
    N = myDim
    inicioG = time.perf_counter()
    K_N = nx.complete_graph(N)
    A = nx.adjacency_matrix(K_N)+np.eye(N)
    graph = hpw.Graph(A)
    fimG    = time.perf_counter()

    inicioC = time.perf_counter()
    qw = hpw.Coined(graph, shift='flipflop', coin=myCoin, marked={'-G': [0]})
    fimC    = time.perf_counter()
    t_final = round(4*np.pi*np.sqrt(N)/4) + 1

    aRange=(startStep,endStep,step)
    #states = qw.simulate(range=t_final, state=qw.uniform_state())
    inicioS = time.perf_counter()
    states = qw.simulate(range=aRange, state=qw.uniform_state())
    fimS    = time.perf_counter()
    print(f"Hypercube: Tempo decorrido: {fimG - inicioG:.6f} segundos", file=sys.stderr)
    print(f"computeU : Tempo decorrido: {fimC - inicioC:.6f} segundos", file=sys.stderr)
    print(f"Iteracoes: Tempo decorrido: {fimS - inicioS:.6f} segundos", file=sys.stderr)
    print(f"Tempo total      decorrido: {fimS - inicioG:.6f} segundos", file=sys.stderr)
    U = qw.get_evolution(); num_arcs=U.shape[0]; densidade=U.nnz/(num_arcs*num_arcs)

    import os
    nome=os.path.splitext(os.path.basename(__file__))[0] # sem extensão
    nome = os.path.basename(__file__)  # Apenas o nome do arquivo
    print(
    f"{nome:14s}, "
    f"dim = {myDim:4d}, "
    f"numStep = {myNumSteps:4d}, "
    f"{coinT}, "
    f"numArcs = {num_arcs:10d}, "
    f"nnz = {U.nnz:12d}, "
    f"densidade = {densidade:.5e}, "
    f"algebra = {algebra:>10s}, "
    f"OMP_NUM_THREADS = {os.getenv('OMP_NUM_THREADS') or 'ND':>3s}, "
    f"tempo computeU = {fimC - inicioC:.5e}, "
    f"tempo Iteracoes = {(fimS - inicioS) / (endStep - startStep):.5e}, "
    f"tempo total = {(fimS - inicioG) :.5e}")
    print('\n')
    return
    marked_prob = qw.success_probability(states)
    hpw.plot_success_probability(t_final, marked_prob)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("Erro:", e, file=sys.stderr)
        traceback.print_exc()

