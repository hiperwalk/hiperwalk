import hiperwalk as hpw

dim = 6
g = hpw.Hypercube(dim)
qw = hpw.Coined(g)
state = qw.state(*[[1, i] for i in range(dim)])
states = qw.simulate(time=(dim), initial_state=state)
probs = qw.probability_distribution(states)

hpw.plot_probability_distribution(probs, graph=g)
