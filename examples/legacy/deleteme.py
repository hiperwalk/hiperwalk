import networkx as nx
import matplotlib.pyplot as plt


g = nx.grid_graph(dim=(5,5))

print(g.nodes())
nx.draw(g, with_labels=True, labels={(0, 0): 'a'})
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = plt.gca()
ax.set_xticks([0, 2], [(0,0), 'a'])

plt.bar([0, 1, 2], [7, 8, 1])
plt.show()

plt.plot([0, 1, 2], [2, 1, 0])
plt.show()
