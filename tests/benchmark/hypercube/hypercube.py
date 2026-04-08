import sys
from time import perf_counter
sys.path.append('../../../')
import hiperwalk as hpw

dim = int(sys.argv[1])
coin = sys.argv[2]
uniform = bool(int(sys.argv[3]))
hpc = sys.argv[4]

if hpc != "none":
    hpw.set_hpc(hpc)

start = perf_counter()
g = hpw.Hypercube(dim)
end = perf_counter()
print("create graph: " + str(end - start))

num_vert = g.number_of_vertices()

start = perf_counter()
qw = hpw.Coined(g, coin=coin)
end = perf_counter()
print("create QW: " + str(end - start))

state = qw.ket(0) if uniform == 0 else qw.uniform_state()
num_ite = int(num_vert**0.5)
start = perf_counter()
res = qw.simulate(state=state,
            range=(num_ite, num_ite + 1))
end = perf_counter()
print("simulation: " + str(end - start))
