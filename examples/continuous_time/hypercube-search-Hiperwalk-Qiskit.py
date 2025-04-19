#!/usr/bin/env python
# coding: utf-8

# ## SEARCH ON HYPERCUBE USING CTQW

# ### Exact calculations with Hiperwalk (http://hiperwalk.org)
# 
# ### Approximate calculations with Qiskit circuits (http://qiskit.org)

# In[1]:


#!pip install hiperwalk
#!pip install qiskit
#!pip install qiskit-aer
#!pip show qiskit qiskit-aer ### qiskit Version: 2.0.0 qiskit-aer Version: 0.17.0


# In[2]:


import numpy as np
import math
import cmath
import scipy as sp
import scipy.special
import matplotlib.pylab as plt

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit import *
from qiskit.circuit import ControlledGate
from qiskit.circuit.library import RYGate
from qiskit.visualization import plot_histogram

from qiskit_aer import *  # updated simulator

import hiperwalk as hpw # for exact calculation using Hiperwalk (http://hiperwalk.org)


# ### Dimension of the hypercube $(n)$

# In[3]:


n = 8

N = 2**n
print('number of qubits = n =', n, '  number of vertices = N =', N)


# ### Results with the exact Hamiltonian using Hiperwalk

# In[4]:


cube = hpw.Hypercube(n)

def S1(n):
    N = 2**n
    # From Eq. (53) in the paper "Efficient Circuit Implementations 
    # of Continuous-Time Quantum Walks for Quantum Search" MDPI Entropy
    return 1/(2*N) * sum(scipy.special.binom(n, k)/k for k in range(1, n+1))
    
qw = hpw.ContinuousTime(cube, gamma=S1(n), time=1, marked = {0})

psi0 = qw.uniform_state()

steps = int(np.pi*np.sqrt(2**n))
states = qw.simulate(range=steps, state=psi0)
succ_prob = qw.success_probability(states)
exact_n = [ (i,succ_prob[i]) for i in range(len(succ_prob))]
hpw.plot_success_probability(steps, succ_prob, marker='.', figsize=(5,3.4), color='orange')


# ### Circuit using Qiskit

# In[5]:


# The following definitions come from the paper "Efficient Circuit Implementations 
# of Continuous-Time Quantum Walks for Quantum Search", Subsection 5.1: 
# "Spatial search on hypercube", MDPI Entropy 2025

def thetap(k, n, sign):
    # From Eq. (60) in the paper
    return -sign * 2 * math.atan(1 / math.sqrt(1 + 2**k)) - math.pi / 2

# mcRy implements Ry with j empty controls
# j - number of controls
# n - total number of qubits
def mcRy(circ,theta,j,n):
    if j==0:
        circ.ry(theta,q[n-1])
    else:
        for i in range(n-j,n):
            circ.x(q[i])
        circ.mcry(theta,
                  q_controls=[n-1+n-j-i for i in range(n-j,n)],
                  q_target=n-j-1,
                  use_basis_gates=False,
                  mode='noancilla')
        for i in range(n-j,n):
            circ.x(q[i])
QuantumCircuit.mcRy = mcRy

def mcRy_dagger(circ,theta,j,n):
    circ.mcRy(2*math.pi-theta,j,n)
QuantumCircuit.mcRy_dagger = mcRy_dagger

# mcRz implements Rz with n-1 empty controls
# n - total number of qubits
def mcRz(circ,theta,n):
    j = n-1  # j - number of controls
    if j==0:
        circ.rz(2*theta,q[n-1])
    else:
        for i in range(n-j-1,n):
            circ.x(q[i])
        circ.mcrz(2*theta,
                  q_controls=[n-1+n-j-i for i in range(n-j,n)],
                  q_target=n-j-1,
                  use_basis_gates=False)
        for i in range(n-j-1,n):
            circ.x(q[i])
QuantumCircuit.mcRz = mcRz

# Rket0 is the circuit for U such that U|0> = |psi>
def Rket0(circ,n,sign):
    circ.h(q[n-1])
    circ.mcRy(thetap(1,n,sign),0,n)
    for i in range(1,n):
        circ.h(q[n-i-1])
        circ.mcRy(thetap(i+1,n,sign),i,n)
QuantumCircuit.Rket0 = Rket0

def Rket0_dagger(circ,n,sign):
    for i in range(n-1,0,-1):
        circ.mcRy(-thetap(i+1,n,sign),i,n)
        circ.h(q[n-i-1])
    circ.mcRy(-thetap(1,n,sign),0,n)
    circ.h(q[n-1])
QuantumCircuit.Rket0_dagger = Rket0_dagger

# THE CIRCUIT OF FIG. 9
def hypercube(circ,n,t):
    circ.Rket0_dagger(n,-1)
    circ.barrier()
    circ.mcRz(-(-1-1/math.sqrt(N))*t,n)
    circ.barrier()
    circ.Rket0(n,-1)
    circ.barrier()
    circ.Rket0_dagger(n,+1)
    circ.barrier()
    circ.mcRz(-(-1+1/math.sqrt(N))*t,n)
    circ.barrier()
    circ.Rket0(n,+1)
    circ.barrier()
QuantumCircuit.hypercube = hypercube 


# In[6]:


succ_prob = {}

# Use Aer's modern simulator
backend = Aer.get_backend('aer_simulator') 

final_t = int(math.pi * math.sqrt(N))

for t in range(final_t):
    # Create quantum and classical registers
    q = QuantumRegister(n, 'qubit')
    c = ClassicalRegister(n, 'bit')
    circ = QuantumCircuit(q, c)
    
    # Apply H gates
    for i in range(n):
        circ.h(q[i])

    # Apply hypercube evolution t times
    #for j in range(t):
    #    circ.hypercube(n, 1)
    circ.hypercube(n, t) # this option is quicker
    
    # Measurement
    circ.measure(q, c)
    
    # Transpile the circuit for the backend (good practice)
    circ = transpile(circ, backend)
    
    # Run the circuit
    job = backend.run(circ, shots=8000)
    resultado = job.result()
    contagem = resultado.get_counts()
    
    # Calculate success probability
    succ_prob[t] = contagem.get('0' * n, 0) / 8000

approx_n = sorted(succ_prob.items()) # sorted by key, return a list of tuples


# ### Plot of the success probability $\left|\langle{0}|{\psi(t)}\rangle\right|^2$ as a function of the number of steps $t$

# In[7]:


plt.plot(*zip(*approx_n),"-o", label="approx (Qiskit)")
plt.plot(*zip(*exact_n), "-o", label="exact (Hiperwalk)")
plt.text(int(0.8*math.pi*math.sqrt(2**n)), 0.9, '$n=$'+str(n), fontsize = 22)
plt.xlabel('t', fontsize = 15)
plt.ylabel("success probability",fontsize = 15)
plt.legend(loc="upper left")
plt.show()


# ### Calculating the distance between the curves

# In[8]:


def distance(approx_n,exact_n):
    return (n,np.sqrt(sum([abs(approx_n[i][1]-exact_n[i][1])**2 for i in range(len(approx_n))]))/len(approx_n))


# In[9]:


distance(approx_n,exact_n)


# In[ ]:




