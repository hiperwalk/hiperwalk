import hiperwalk as hpw

dim = 121
lat = hpw.Lattice(dim, diagonal=True)
center = lat.get_central_vertex()
dtqw = hpw.CoinedWalk(lat, shift='persistent', coin='grover')
psi0 = dtqw.state([0.5, (center, center + (1, 1))],
                  [-0.5, (center, center + (1, -1))],
                  [-0.5, (center, center + (-1, 1))],
                  [0.5, (center, center + (-1, -1))])
psi_final = dtqw.simulate(time=dim // 2, initial_condition=psi0, hpc=False)
prob = dtqw.probability_distribution(psi_final)
hpw.plot_probability_distribution(prob, graph=lat)
