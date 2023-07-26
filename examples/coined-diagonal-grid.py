import hiperwalk as hpw
import numpy as np

dim = 121
grid = hpw.Grid(dim, diagonal=True)
center = np.array([dim // 2, dim // 2])
dtqw = hpw.Coined(grid, shift='persistent', coin='grover')
psi0 = dtqw.state([0.5, (center, center + (1, 1))],
                  [-0.5, (center, center + (1, -1))],
                  [-0.5, (center, center + (-1, 1))],
                  [0.5, (center, center + (-1, -1))])
psi_final = dtqw.simulate(time=dim // 2, initial_state=psi0, hpc=False)
prob = dtqw.probability_distribution(psi_final)
hpw.plot_probability_distribution(prob, graph=grid)

