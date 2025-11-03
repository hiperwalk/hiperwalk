import numpy as np

import sys
#sys.stdout.reconfigure(line_buffering=False, write_through=False)
import time

import hiperwalk as hpw
myOption=None
myOption="cpu"
hpw.set_hpc(myOption) 
print("\nA, get_hpc()=", hpw.get_hpc() )

dim = 32*5*2*2*2
dim = 3*1

step=1
start=1; end=start+5+10000; #step=1 #10*300//1-1
start=1; end=start+1+1; #step=1 #10*300//1-1
aRange=(start,end,step)

grid = hpw.Grid(dim, diagonal=True, periodic=True)
grid = hpw.Grid(dim, diagonal=True, periodic=False)

print(f"graph=hpw.grid({dim}),  aRange = {aRange}, get_hpc() = { hpw.get_hpc()}" )

inicioA = time.perf_counter()
dtqw = hpw.Coined(grid, shift='persistent', coin='grover')
fimA = time.perf_counter()
print(f"computeU : Tempo decorrido: {fimA - inicioA:.6f} segundos");
#exit()
center = np.array([dim // 2, dim // 2])
psi0 = dtqw.state([[0.5, (center, center + (1, 1))],
                   [-0.5, (center, center + (1, -1))],
                   [-0.5, (center, center + (-1, 1))],
                   [0.5, (center, center + (-1, -1))]])
inicio = time.perf_counter()
for r in range(1): #50*1000*1000):
    psi_final = dtqw.simulate(range=aRange, state=psi0)
#KOR psi_final = dtqw.simulate(range=aRange, state=psi0)
fim = time.perf_counter()
print("get_hpc()=", hpw.get_hpc() )
print(f"computeU : Tempo decorrido: {fimA - inicioA:.6f} segundos")
print(f"Iteracoes: Tempo decorrido: {fim - inicio:.6f} segundos")
#psi_final = dtqw.simulate(range=(1, 29 + 1), state=psi0)
#prob = dtqw.probability_distribution(psi_final)
#hpw.plot_probability_distribution(prob, graph=grid)



U = dtqw.get_evolution()
print('--------------------------------------------------------------')
print(hex(id(U.indices)))
print(hex(id(U.indptr)))
print("& U.data = " , hex(id(U.data)))
print("data[0] = " , U.data[0])

data = U.data
# Endereço base e tamanho de cada elemento
addr_base = data.__array_interface__['data'][0]
itemsize = data.itemsize
print(f"Endereço base de U.data (U.data[0]): {hex(addr_base)}")
for i in range(min(2, data.size)):  # mostra os dois primeiros
    print(f"&U.data[{i}] = {hex(addr_base + i * itemsize)}")

indices = U.indices
# Endereço base e tamanho de cada elemento
addr_base = indices.__array_interface__['data'][0]
itemsize = indices.itemsize
print(f"Endereço base de U.indices (U.indices[0]): {hex(addr_base)}")
for i in range(min(2, indices.size)):  # mostra os dois primeiros
    print(f"&U.indices[{i}] = {hex(addr_base + i * itemsize)}")


print('--------- TIPOS VETORES INTERNOS CSR ---------------------')
print(type(U.indptr[0]))
print(type(U.indices[0]))
print(type(U.data[0]))
