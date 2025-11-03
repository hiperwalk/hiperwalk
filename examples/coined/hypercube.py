import hiperwalk as hpw
#import hiperblas as nbl

#nbl.init_engine(nbl.CPU,0)

import time
import sys
#sys.stdout.reconfigure(line_buffering=False, write_through=False)

myOption="cpu"
myOption=None
hpw.set_hpc(myOption)
print("A, get_hpc()=", hpw.get_hpc() )


dim = 3 + 1 -1 -0; start=1; end=start+1; step=1 #10*300//1-1
dim = 16 + 6 - 4; start=1; end=start+20*10; step=1 #10*300//1-1
aRange=(start,end,step)

print(f"graph=hpw.Hypercube({dim}),  aRange = {aRange}, get_hpc() = { hpw.get_hpc()}" )

inicioG = time.perf_counter()
g = hpw.Hypercube(dim)
fimG = time.perf_counter()
print(f"Hypercube: Tempo decorrido: {fimG - inicioG:.6f} segundos")
inicioC = time.perf_counter()
qw = hpw.Coined(g)
fimC = time.perf_counter()
print(f"computeU : Tempo decorrido: {fimC - inicioC:.6f} segundos")
print("B, get_hpc()=", hpw.get_hpc() ); 
aState  = qw.state([[1, i] for i in range(dim)])

inicioS = time.perf_counter()
for r in range(1): #50*1000*1000):
    states = qw.simulate(range=aRange, state=aState)
fimS = time.perf_counter()
print(f"Hypercube: tempo decorrido: {fimG - inicioG:.6f} segundos")
print(f"computeU : tempo decorrido: {fimC - inicioC:.6f} segundos")
print(f"Iteracoes: Tempo decorrido: {fimS - inicioS:.6f} segundos")

#states = qw.simulate(range=(1, 1 + 1), state=state)
exit()

print('\n\n\n')
print(len(states))
exit()
probs = qw.probability_distribution(states)

#hpw.plot_probability_distribution(probs, graph=g)
print(probs)
#plt.savefig("grafico.png")
