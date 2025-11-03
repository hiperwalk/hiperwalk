import hiperwalk as hpw
import numpy as np
#hiperwalkimport hiperblas as nbl

#nbl.init_engine(nbl.CPU,0)

import time
import sys
#sys.stdout.reconfigure(line_buffering=False, write_through=False)
sys.stdout.reconfigure(line_buffering=True)

myOption=None
myOption="cpu"

dim = 16 + 6 - 4 - 4 - 4 - 4 ; start=1; end=start+5; step=1 #10*300//1-1
dim = 16 + 6 - 00; start=1; end=start+5; step=1 #10*300//1-1
dim = 3 + 9 +6 +1; start=1; end=start+1+1-0; step=1 #10*300//1-1
dim = 16 + 6 - 4 - 4 ; start=1; end=start+5*20*10; step=1 #10*300//1-1
dim = 5 ; start=1; end=start+1+1-0; step=1 #10*300//1-1
dim = 16 + 6 - 4 - 2  ; start=1; end=start+2000; step=1 #10*300//1-1
aRange=(start,end,step)

grafo="G" # Grover  para Real
grafo="F" # Fourier para Complex

from warnings import warn
def main():

    hpw.set_hpc(myOption)
    print(f"graph=hpw.Hypercube({dim}),  aRange = {aRange}, get_hpc() = { hpw.get_hpc()}" )

    inicioG = time.perf_counter()
    g = hpw.Hypercube(dim)
    fimG    = time.perf_counter()
    print(f"Hypercube: Tempo decorrido: {fimG - inicioG:.6f} segundos", file=sys.stderr)

    inicioC = time.perf_counter()
    qw = hpw.Coined(g, coin=grafo) 
    fimC    = time.perf_counter()
    print(f"computeU : Tempo decorrido: {fimC - inicioC:.6f} segundos", file=sys.stderr)
    #return

    initialState = qw.state([[1, i] for i in range(dim)])
    np.set_printoptions(threshold=10)
    print("initialState = ", np.array(initialState),  end="; ")
    print("state.l2Norm=", np.linalg.norm(initialState));

    inicioS = time.perf_counter()
    for r in range(1): #50*1000*1000):
       states = qw.simulate(range=aRange, state=initialState)
    fimS    = time.perf_counter()

    print(f"Hypercube: tempo decorrido: {fimG - inicioG:.6f} segundos", file=sys.stderr)
    print(f"computeU : tempo decorrido: {fimC - inicioC:.6f} segundos", file=sys.stderr)
    print(f"Iteracoes: Tempo decorrido: {fimS - inicioS:.6f} segundos", file=sys.stderr)

    print('\n\n\n')
    print(len(states))
    probs = qw.probability_distribution(states)

    #hpw.plot_probability_distribution(probs, graph=g)
    print(probs)
    #plt.savefig("grafico.png")

main()
