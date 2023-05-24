import sys
sys.path.append('..')
import hiperwalk as hpw
import networkx as nx
import numpy as np

dim = 4
l = hpw.Lattice(dim, periodic=False, diagonal=False)

arcs = [l.arc(label) for label in range(l.number_of_arcs())]
arcs_labels = [l.arc_label(arc[0], arc[1]) for arc in arcs]
print(arcs)
print(arcs_labels)
print(arcs_labels == list(range(l.number_of_arcs())))
