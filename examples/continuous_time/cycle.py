import hiperwalk as hpw
import os;
import numpy as np

hb="n"
hb="y"
myHPC_option="cpu" if hb == "y" else None
hpw.set_hpc(myHPC_option)
print('num nucleos    =', os.cpu_count());
print('OMP_NUM_THREADS=', os.environ.get('OMP_NUM_THREADS'));
print('MKL_NUM_THREADS=', os.environ.get('MKL_NUM_THREADS'))

N = 101*10*2
N = 101
N = 101*2*3


#print('number of qubits = n =', n, '  number of vertices = N =', N)
print(' number of vertices, N =', N)
cycle = hpw.Cycle(N)
ctqw = hpw.ContinuousTime(cycle, gamma=0.35)
psi0 = ctqw.ket(N // 2)
aRange=(N//2-10, N//2 + 1)
aRange=(1,200*5,1)
aRange=(1,10+1,1)
psi_final = ctqw.simulate(range_=aRange, state=psi0)
prob = ctqw.probability_distribution(psi_final)
#hpw.plot_probability_distribution(prob)
np.set_printoptions(linewidth=820, threshold=240)
print("probs =\n", prob)
print("np.sum(prob) = ", np.sum(prob))
