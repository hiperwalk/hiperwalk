import hiperwalk as hpw

N = 101
cycle = hpw.Cycle(N)
ctqw = hpw.ContinuousWalk(cycle, gamma=0.35)
psi0 = ctqw.ket(N // 2)
psi_final = ctqw.simulate(time=50, initial_condition=psi0)
prob = ctqw.probability_distribution(psi_final)
hpw.plot_probability_distribution(prob)
