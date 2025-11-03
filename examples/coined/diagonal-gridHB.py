import hiperwalk as hpw
import numpy as np

import sys
#sys.stdout.reconfigure(line_buffering=False, write_through=False)

hpw.set_hpc(None) 
hpw.set_hpc("cpu") 
print("A, get_hpc()=", hpw.get_hpc() )

dim = 221*2
dim = 3*1

grid = hpw.Grid(dim, diagonal=True, periodic=False)
dtqw = hpw.Coined(grid, shift='persistent', coin='grover')
center = np.array([dim // 2, dim // 2])
psi0 = dtqw.state([[0.5, (center, center + (1, 1))],
                   [-0.5, (center, center + (1, -1))],
                   [-0.5, (center, center + (-1, 1))],
                   [0.5, (center, center + (-1, -1))]])
initialST=1; finalST=4
psi_final = dtqw.simulate(range=(initialST, finalST), state=psi0)
#psi_final = dtqw.simulate(range=(1, 29 + 1), state=psi0)
prob = dtqw.probability_distribution(psi_final)
#hpw.plot_probability_distribution(prob, graph=grid)



