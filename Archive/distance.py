# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 16:04:28 2017

@author: tharso
"""
import config as cfg
import networkx as nx
import os


def read_adj_matrix(file_name):
    G = nx.Graph()

    with open(file_name) as f_r:
        content = f_r.readlines()

    for k in range(0, len(content)):
        inf = content[k].split()
        if int(inf[2]) != 0:
            #print [int(inf[0]),int(inf[1])]
            G.add_edges_from([(int(inf[0]), int(inf[1]))])

    nx.draw(G)
    return G


def dijkstra(ProbPsi, PsiSize, G, file_name):
    print("[Hiperwalk] Computing distances...")
    alldists = nx.all_pairs_shortest_path_length(G)
    f = open(file_name, 'w')
    d = [0.] * PsiSize
    for i in ProbPsi:
        index = i[0]
        prob = i[1]
        for j in range(PsiSize):
            d[j] += prob * alldists[index][j + 1]
    for k in range(len(d)):
        f.write(str(d[k]) + '\n')
    f.close()


def nonblank_lines(f):
    for l in f:
        line = l.rstrip()
        if line:
            yield line


def generateDistance():
    fn = cfg.ADJMATRIX_PATH
    G = read_adj_matrix(fn)
    curpath = os.path.abspath(os.curdir)
    psiFileName = cfg.CUSTOM_INITIALSTATE_NAME
    linecount = 1
    PsiProb = []
    with open(psiFileName) as f_in:
        for line in nonblank_lines(f_in):
            num = line.split(' ')
            if len(num) == 1:
                re = float(num[0])
                im = 0.0
            elif len(num) >= 2:
                re = float(num[0])
                im = float(num[1])

            if (re != 0. or im != 0.):
                PsiProb.append([linecount, re * re + im * im])
            linecount += 1
    dijkstra(PsiProb, linecount - 1, G, curpath + "/distance_custom.dat")
