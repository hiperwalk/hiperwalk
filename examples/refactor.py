import sys
sys.path.append('..')
import hiperwalk as hpw
import networkx as nx
import numpy as np

num_vert = 4
l = hpw.Line(num_vert)
for i in range(l.number_of_arcs()):
    a = l.arc(i)
    #print(str(a) + ' -> ' + str(l.next_arc(a)))
    #print(str(l.arc(i)) + ' -> ' + str(l.arc(l.next_arc(i))))
    if (a != l.next_arc(l.previous_arc(a))
        or a != l.previous_arc(l.next_arc(a))
        or i != l.next_arc(l.previous_arc(i))
        or i != l.previous_arc(l.next_arc(i))):

        raise ValueError

print("OK")

qw = hpw.CoinedWalk(graph = l)
print(qw._shift.todense())
