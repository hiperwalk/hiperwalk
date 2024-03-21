import hiperwalk as hpw
from hiperwalk._constants import HPC
import numpy as np

dim = 121
grid = hpw.Grid(dim, diagonal=True)
dtqw = hpw.Coined(grid, shift='persistent', coin='grover')
center = np.array([dim // 2, dim // 2])
psi0 = dtqw.state([[0.5, (center, center + (1, 1))],
                   [-0.5, (center, center + (1, -1))],
                   [-0.5, (center, center + (-1, 1))],
                   [0.5, (center, center + (-1, -1))]])
psi_final = dtqw.simulate(time=dim // 2, state=psi0, hpc=HPC.NONE)
prob = dtqw.probability_distribution(psi_final)
hpw.plot_probability_distribution(prob, graph=grid)

