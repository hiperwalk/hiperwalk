import hiperwalk as hpw
import numpy as np

dim = 121
grid = hpw.Grid(dim, diagonal=True, periodic=False)
dtqw = hpw.Coined(grid, shift='persistent', coin='grover')
center = np.array([dim // 2, dim // 2])
psi0 = dtqw.state([[0.5, (center, center + (1, 1))],
                   [-0.5, (center, center + (1, -1))],
                   [-0.5, (center, center + (-1, 1))],
                   [0.5, (center, center + (-1, -1))]])
psi_final = dtqw.simulate(range=(dim // 2, dim // 2 + 1),
                          state=psi0)
prob = dtqw.probability_distribution(psi_final)
hpw.plot_probability_distribution(prob, graph=grid)

