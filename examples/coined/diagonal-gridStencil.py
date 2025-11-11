import hiperwalk as hpw
import numpy as np
#hiperwalkimport hiperblas as nbl

#nbl.init_engine(nbl.CPU,0)

import time
import sys
#sys.stdout.reconfigure(line_buffering=False, write_through=False)
sys.stdout.reconfigure(line_buffering=True)

numSteps=2000
dim = 16 + 6 - 3   ; startStep=1; endStep=startStep+numSteps; step=1 #10*300//1-1

aRange=(startStep,endStep,step)

coin="G" # Grover  para Real
coin="F" # Fourier para Complex  para Real

myOption=None
myOption="cpu"

aDim = 10
aCoin = "G"
aHPCoPTION = None

dim      =aDim        # 10  
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
    grid = hpw.Grid(dim, diagonal=True, periodic=False)
    fimG    = time.perf_counter()

    inicioC = time.perf_counter()
    dtqw = hpw.Coined(grid, shift='persistent', coin='grover')
    fimC = time.perf_counter()

    center = np.array([dim // 2, dim // 2])
    psi0 = dtqw.state([[0.5, (center, center + (1, 1))],
                   [-0.5, (center, center + (1, -1))],
                   [-0.5, (center, center + (-1, 1))],
                   [0.5, (center, center + (-1, -1))]])
    inicioS = time.perf_counter()
    for r in range(1): #50*1000*1000):
        psi_final = dtqw.simulate(range=aRange, state=psi0)
    #KOR psi_final = dtqw.simulate(range=aRange, state=psi0)
    fimS = time.perf_counter()
    print(f"Hypercube: Tempo decorrido: {fimG - inicioG:.6f} segundos", file=sys.stderr)
    print(f"computeU : Tempo decorrido: {fimC - inicioC:.6f} segundos", file=sys.stderr)
    print(f"Iteracoes: Tempo decorrido: {fimS - inicioS:.6f} segundos", file=sys.stderr)
    print(f"Tempo total      decorrido: {fimS - inicioG:.6f} segundos", file=sys.stderr)
    U = dtqw.get_evolution(); num_arcs=U.shape[0]; densidade=U.nnz/(num_arcs*num_arcs)

    import os
    nome=os.path.splitext(os.path.basename(__file__))[0] # sem extensão
    nome = os.path.basename(__file__)  # Apenas o nome do arquivo
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
    f"tempo Iteracoes = {(fimS - inicioS) / (endStep - startStep):.5e}, ",
    f"tempo total = {(fimS - inicioG) :.5e}")

    return

    psi_final = dtqw.simulate(range=(1, 29 + 1), state=psi0)
    prob = dtqw.probability_distribution(psi_final)
    #hpw.plot_probability_distribution(probs, graph=grid)
    #print(probs)
    #plt.savefig("grafico.png")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("Erro:", e, file=sys.stderr)
        traceback.print_exc()


#main()
