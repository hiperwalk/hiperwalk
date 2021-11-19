# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 02:10:29 2014

@author: aaron
"""

import numpy as np
import config as cfg
import ioFunctions as io

#import scipy.sparse as sps


###
### COINS
###
def hadamard(N):
    """ Retorna o operador de Hadamard no formato (1+0j)
    """

    H = np.ones((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):

            a = np.array(map(int, list('{0:010b}'.format(i))))
            b = np.array(map(int, list('{0:010b}'.format(j))))
            #print(a,b,np.dot(a,b))
            H[i][j] = (-1)**np.dot(a, b)

    H = H / np.sqrt(N)
    return H


def fourier(N):
    """ Retorna o operador de fourier para a dimensao N.
    """

    H = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            H[i][j] = np.cos((2 * np.pi * i * j) / N) + 1J * np.sin(
                (2 * np.pi * i * j) / N)
    H = H / np.sqrt(N)

    return H


def grover(N):

    G = np.dot((2.0 / N), np.ones((N, N), dtype=complex))

    for i in range(N):
        for j in range(N):
            if i == j:
                G[i][j] = G[i][j] - 1

    return G


def identity(N):
    """ Returns the identity of size N
    """
    return np.eye(N, dtype=complex)


def check_Unitarity(operator, size):
    norm = np.linalg.norm(operator, 2)
    if abs(norm - 1) > 0.00000001:
        print(
            "[HIPERWALK] WARNING: Operator is not unitary, with norm: %f1.16" %
            norm)


##
##  DTQW1D
##
def SHIFT_OPERATOR_1D(COINVECTORSIZE, GRAPHSIZE):
    """ |0><0|TENSOR SOMA(|i+1><i|) + |1><1|TENSOR SOMA(|i-1><i|)
    """
    trunc = int(GRAPHSIZE)

    try:
        f = open("HIPERWALK_TEMP_SHIFT_OPERATOR_1D.dat", 'w')
    except IOError:
        print("[HIPERWALK] Could not open file in directory.")

    for i in range(int(GRAPHSIZE) - 1):
        f.write("%d %d %d %d\n" % (i + 1 + 1, i + 1, 1, 0))
        f.write("%d %d %d %d\n" % (trunc + i + 1, trunc + i + 1 + 1, 1, 0))

    f.write("%d %d %d %d\n" % (1, trunc, 1, 0))
    f.write("%d %d %d %d\n" % (2 * trunc, trunc + 1, 1, 0))


def COIN_TENSOR_IDENTITY_1D(C, sizeI):
    try:
        A = open("HIPERWALK_TEMP_COIN_TENSOR_IDENTITY_1D.dat", 'w')
    except IOError:
        print("[HIPERWALK] Could not open file in directory.")

    for i in range(int(sizeI)):
        A.write("%d %d %1.16f %1.16f\n" %
                (i + 1, i + 1, C[0][0].real, C[0][0].imag))
        A.write("%d %d %1.16f %1.16f\n" %
                (i + 1, i + 1 + sizeI, C[0][1].real, C[0][1].imag))
        A.write("%d %d %1.16f %1.16f\n" %
                (i + 1 + sizeI, i + 1, C[1][0].real, C[1][0].imag))
        A.write("%d %d %1.16f %1.16f\n" %
                (i + 1 + sizeI, i + 1 + sizeI, C[1][1].real, C[1][1].imag))
    A.close()


##
##  DTQW2D
##


def DIAGONAL_SHIFT_OPERATOR_2D():
    try:
        A = open("HIPERWALK_TEMP_SHIFT_OPERATOR_2D.dat", 'w')
    except IOError:
        print("[HIPERWALK] Could not open file in directory.")

    for y in range(cfg.RANGEY[0], cfg.RANGEY[1] + 1, 1):
        for x in range(cfg.RANGEX[0], cfg.RANGEX[1] + 1, 1):

            ###
            ###       S0=|x+1,y+1><x,y|
            ###
            newX = x + 1
            newY = y + 1
            if newX > cfg.RANGEX[1]:
                newX = cfg.RANGEX[0]
            if newY > cfg.RANGEY[1]:
                newY = cfg.RANGEY[0]

            if cfg.GRAPHTYPE == "TORUS":
                indexOld = (x) * (cfg.RANGEY[1] + 1) + (y)
                indexNew = (newX) * (cfg.RANGEY[1] + 1) + (newY)

            else:
                indexOld = (x - cfg.RANGEX[0]) * (
                    cfg.RANGEY[1] - cfg.RANGEY[0] + 1) + (y - cfg.RANGEY[0])
                indexNew = (newX - cfg.RANGEX[0]) * (
                    cfg.RANGEY[1] - cfg.RANGEY[0] + 1) + (newY - cfg.RANGEY[0])
            A.write("%d %d 1 0\n" % (indexNew + 1, indexOld + 1))
            ###
            ###       S1=|x+1,y-1><x,y|
            ###
            newX = x + 1
            newY = y - 1
            if newX > cfg.RANGEX[1]:
                newX = cfg.RANGEX[0]
            if newY < cfg.RANGEY[0]:
                newY = cfg.RANGEY[1]
            if cfg.GRAPHTYPE == "TORUS":
                indexNew = (newX) * (cfg.RANGEY[1] + 1) + (newY)
            else:
                indexNew = (newX - cfg.RANGEX[0]) * (
                    cfg.RANGEY[1] - cfg.RANGEY[0] + 1) + (newY - cfg.RANGEY[0])
            A.write(
                "%d %d 1 0\n" %
                (cfg.GRAPHSIZE + indexNew + 1, cfg.GRAPHSIZE + indexOld + 1))
            ###
            ###       S2=|x-1,y+1><x,y|
            ###
            newX = x - 1
            newY = y + 1
            if newX < cfg.RANGEX[0]:
                newX = cfg.RANGEX[1]
            if newY > cfg.RANGEY[1]:
                newY = cfg.RANGEY[0]
            if cfg.GRAPHTYPE == "TORUS":
                indexNew = (newX) * (cfg.RANGEY[1] + 1) + (newY)
            else:
                indexNew = (newX - cfg.RANGEX[0]) * (
                    cfg.RANGEY[1] - cfg.RANGEY[0] + 1) + (newY - cfg.RANGEY[0])
            A.write("%d %d 1 0\n" % (2 * cfg.GRAPHSIZE + indexNew + 1,
                                     2 * cfg.GRAPHSIZE + indexOld + 1))
            ###
            ###       S3=|x-1,y-1><x,y|
            ###
            newX = x - 1
            newY = y - 1
            if newX < cfg.RANGEX[0]:
                newX = cfg.RANGEX[1]
            if newY < cfg.RANGEY[0]:
                newY = cfg.RANGEY[1]
            if cfg.GRAPHTYPE == "TORUS":
                indexNew = (newX) * (cfg.RANGEY[1] + 1) + (newY)
            else:
                indexNew = (newX - cfg.RANGEX[0]) * (
                    cfg.RANGEY[1] - cfg.RANGEY[0] + 1) + (newY - cfg.RANGEY[0])
            A.write("%d %d 1 0\n" % (3 * cfg.GRAPHSIZE + indexNew + 1,
                                     3 * cfg.GRAPHSIZE + indexOld + 1))
    A.close()


def NATURAL_SHIFT_OPERATOR_2D():
    try:
        A = open("HIPERWALK_TEMP_SHIFT_OPERATOR_2D.dat", 'w')
    except IOError:
        print("[HIPERWALK] Could not open file in directory.")

    for y in range(cfg.RANGEY[0], cfg.RANGEY[1] + 1, 1):
        for x in range(cfg.RANGEX[0], cfg.RANGEX[1] + 1, 1):

            ###
            ###       S0=|x+1,y+1><x,y|
            ###
            newX = x + 0
            newY = y + 1
            if newX > cfg.RANGEX[1]:
                newX = cfg.RANGEX[0]
            if newY > cfg.RANGEY[1]:
                newY = cfg.RANGEY[0]
            if cfg.GRAPHTYPE == "TORUS":
                indexOld = (x) * (cfg.RANGEY[1] + 1) + (y)
                indexNew = (newX) * (cfg.RANGEY[1] + 1) + (newY)

            else:
                indexOld = (x - cfg.RANGEX[0]) * (
                    cfg.RANGEY[1] - cfg.RANGEY[0] + 1) + (y - cfg.RANGEY[0])
                indexNew = (newX - cfg.RANGEX[0]) * (
                    cfg.RANGEY[1] - cfg.RANGEY[0] + 1) + (newY - cfg.RANGEY[0])
            A.write("%d %d 1 0\n" % (0 * cfg.GRAPHSIZE + indexNew + 1,
                                     0 * cfg.GRAPHSIZE + indexOld + 1))
            ###
            ###       S1=|x+1,y-1><x,y|
            ###
            newX = x + 1
            newY = y + 0
            if newX > cfg.RANGEX[1]:
                newX = cfg.RANGEX[0]
            if newY < cfg.RANGEY[0]:
                newY = cfg.RANGEY[1]
            if cfg.GRAPHTYPE == "TORUS":
                indexNew = (newX) * (cfg.RANGEY[1] + 1) + (newY)
            else:
                indexNew = (newX - cfg.RANGEX[0]) * (
                    cfg.RANGEY[1] - cfg.RANGEY[0] + 1) + (newY - cfg.RANGEY[0])
            A.write("%d %d 1 0\n" % (1 * cfg.GRAPHSIZE + indexNew + 1,
                                     1 * cfg.GRAPHSIZE + indexOld + 1))
            ###
            ###       S2=|x-1,y+1><x,y|
            ###
            newX = x - 1
            newY = y + 0
            if newX < cfg.RANGEX[0]:
                newX = cfg.RANGEX[1]
            if newY > cfg.RANGEY[1]:
                newY = cfg.RANGEY[0]
            if cfg.GRAPHTYPE == "TORUS":
                indexNew = (newX) * (cfg.RANGEY[1] + 1) + (newY)
            else:
                indexNew = (newX - cfg.RANGEX[0]) * (
                    cfg.RANGEY[1] - cfg.RANGEY[0] + 1) + (newY - cfg.RANGEY[0])
            A.write("%d %d 1 0\n" % (2 * cfg.GRAPHSIZE + indexNew + 1,
                                     2 * cfg.GRAPHSIZE + indexOld + 1))
            ###
            ###       S3=|x-1,y-1><x,y|
            ###
            newX = x + 0
            newY = y - 1
            if newX < cfg.RANGEX[0]:
                newX = cfg.RANGEX[1]
            if newY < cfg.RANGEY[0]:
                newY = cfg.RANGEY[1]
            if cfg.GRAPHTYPE == "TORUS":
                indexNew = (newX) * (cfg.RANGEY[1] + 1) + (newY)
            else:
                indexNew = (newX - cfg.RANGEX[0]) * (
                    cfg.RANGEY[1] - cfg.RANGEY[0] + 1) + (newY - cfg.RANGEY[0])
            A.write("%d %d 1 0\n" % (3 * cfg.GRAPHSIZE + indexNew + 1,
                                     3 * cfg.GRAPHSIZE + indexOld + 1))

    A.close()


def NOVO_SHIFT_OPERATOR_2D():
    try:
        A = open("HIPERWALK_TEMP_SHIFT_OPERATOR_2D.dat", 'w')
    except IOError:
        print("[HIPERWALK] Could not open file in directory.")

    for y in range(cfg.RANGEY[0], cfg.RANGEY[1] + 1, 1):
        for x in range(cfg.RANGEX[0], cfg.RANGEX[1] + 1, 1):

            ###
            ###       S0=|x+1,y+1><x,y|
            ###
            newX = x + 1
            newY = y + 0

            if newX > cfg.RANGEX[1]:
                newX = cfg.RANGEX[0]
            indexOld = (x - cfg.RANGEX[0]) * (cfg.RANGEY[1] - cfg.RANGEY[0] +
                                              1) + (y - cfg.RANGEY[0])
            indexNew = (newX - cfg.RANGEX[0]) * (
                cfg.RANGEY[1] - cfg.RANGEY[0] + 1) + (newY - cfg.RANGEY[0])
            A.write("%d %d 1 0\n" % (0 * cfg.GRAPHSIZE + indexNew + 1,
                                     0 * cfg.GRAPHSIZE + indexOld + 1))

            ###
            ###       S1=|x+1,y-1><x,y|
            ###
            newX = x - 1
            newY = y + 0
            if newX < cfg.RANGEX[0]:
                newX = cfg.RANGEX[1]
            indexNew = (newX - cfg.RANGEX[0]) * (
                cfg.RANGEY[1] - cfg.RANGEY[0] + 1) + (newY - cfg.RANGEY[0])
            A.write("%d %d 1 0\n" % (1 * cfg.GRAPHSIZE + indexNew + 1,
                                     1 * cfg.GRAPHSIZE + indexOld + 1))

            ###
            ###       S2=|x-1,y+1><x,y|
            ###
            newX = x + 0
            newY = y + 1
            if newY > cfg.RANGEY[1]:
                newY = cfg.RANGEY[0]
            indexNew = (newX - cfg.RANGEX[0]) * (
                cfg.RANGEY[1] - cfg.RANGEY[0] + 1) + (newY - cfg.RANGEY[0])
            A.write("%d %d 1 0\n" % (2 * cfg.GRAPHSIZE + indexNew + 1,
                                     2 * cfg.GRAPHSIZE + indexOld + 1))

            ###
            ###       S3=|x-1,y-1><x,y|
            ###
            newX = x + 0
            newY = y - 1
            if newY < cfg.RANGEY[0]:
                newY = cfg.RANGEY[1]
            indexNew = (newX - cfg.RANGEX[0]) * (
                cfg.RANGEY[1] - cfg.RANGEY[0] + 1) + (newY - cfg.RANGEY[0])
            A.write("%d %d 1 0\n" % (3 * cfg.GRAPHSIZE + indexNew + 1,
                                     3 * cfg.GRAPHSIZE + indexOld + 1))

    A.close()


def COIN_TENSOR_IDENTITY_OPERATOR_2D():
    try:
        f = open("HIPERWALK_TEMP_COIN_TENSOR_IDENTITY_OPERATOR_2D.dat", "w")
    except IOError:
        print("[HIPERWALK] Could not open file in directory.")

    for i in range(4):
        for j in range(4):
            for w in range(cfg.GRAPHSIZE):
                f.write(
                    "%d %d %1.16f %1.16f\n" %
                    (cfg.GRAPHSIZE * i + w + 1, cfg.GRAPHSIZE * j + w + 1,
                     cfg.COINOPERATOR[i][j].real, cfg.COINOPERATOR[i][j].imag))
    f.close()


##
##  STAGGERED
##
def STAGGERED1D():
    #    STAGGERED_EVEN_OPERATOR_1D()
    #    STAGGERED_ODD_OPERATOR_1D()
    teste_STAGGERED_EVEN_OPERATOR_1D()
    teste_STAGGERED_ODD_OPERATOR_1D()


def STAGGERED_EVEN_OPERATOR_1D():
    try:
        f = open("HIPERWALK_TEMP_STAGGERED_EVEN_OPERATOR_1D.dat", 'w')
    except IOError:
        print("[HIPERWALK] Could not open file in directory.")

    alpha = float(cfg.STAGGERED_COEFICIENTS[0][0])
    phi = float(cfg.STAGGERED_COEFICIENTS[0][1])
    aux = np.zeros([2, 2], dtype=complex)
    #    aux[0][0]=np.cos(alpha)
    #    aux[0][1]=np.sin(alpha)*(np.cos(phi)+1J*np.sin(phi))
    #    aux[1][0]=aux[0][1]
    #    aux[1][1]= 2 * np.sin(alpha/2) * np.sin(alpha/2) * ( np.cos(2*phi)+1J*np.sin(2*phi))-1

    aux[0][0] = 2 * np.cos(alpha / 2) * np.cos(alpha / 2) - 1
    aux[0][1] = 2 * np.cos(alpha / 2) * np.sin(
        alpha / 2) * (np.cos(phi) - 1j * np.sin(phi))
    aux[1][0] = 2 * np.cos(alpha / 2) * np.sin(
        alpha / 2) * (np.cos(phi) + 1j * np.sin(phi))
    aux[1][1] = 2 * np.sin(alpha / 2) * np.sin(alpha / 2) - 1

    for i in np.arange(0, cfg.STATE.shape[0] - 1, 2):
        f.write("%d %d %1.16f %1.16f\n" %
                (i + 1, i + 1, aux[0][0].real, aux[0][0].imag))
        f.write("%d %d %1.16f %1.16f\n" %
                (i + 2, i + 1, aux[1][0].real, aux[1][0].imag))
        f.write("%d %d %1.16f %1.16f\n" %
                (i + 1, i + 2, aux[0][1].real, aux[0][1].imag))
        f.write("%d %d %1.16f %1.16f\n" %
                (i + 2, i + 2, aux[1][1].real, aux[1][1].imag))
    f.close()


def STAGGERED_ODD_OPERATOR_1D():
    try:
        f = open("HIPERWALK_TEMP_STAGGERED_ODD_OPERATOR_1D.dat", 'w')
    except IOError:
        print("[HIPERWALK] Could not open file in directory.")

    beta = float(cfg.STAGGERED_COEFICIENTS[1][0])
    phi = float(cfg.STAGGERED_COEFICIENTS[1][1])
    aux = np.zeros([2, 2], dtype=complex)
    #    aux[0][0]=np.cos(beta)
    #    aux[0][1]=np.sin(beta)*(np.cos(phi)+1J*np.sin(phi))
    #    aux[1][0]=aux[0][1]
    #    aux[1][1]=2*np.sin(beta/2)*np.sin(beta/2)*(np.cos(2*phi)+1J*np.sin(2*phi))-1

    aux[0][0] = 2 * np.sin(beta / 2) * np.sin(beta / 2) - 1
    aux[0][1] = 2 * np.sin(beta / 2) * np.cos(
        beta / 2) * (np.cos(phi) - 1J * np.sin(phi))
    aux[1][0] = 2 * np.sin(beta / 2) * np.cos(
        beta / 2) * (np.cos(phi) + 1J * np.sin(phi))
    aux[1][1] = 2 * np.cos(beta / 2) * np.cos(beta / 2) - 1

    f.write("%d %d %1.16f %1.16f\n" % (1, 1, aux[1][1].real, aux[1][1].imag))
    f.write("%d %d %1.16f %1.16f\n" %
            (1, cfg.STATE.shape[0], aux[0][1].real, aux[0][1].imag))
    f.write("%d %d %1.16f %1.16f\n" %
            (cfg.STATE.shape[0], 1, aux[1][0].real, aux[1][0].imag))
    f.write("%d %d %1.16f %1.16f\n" % (cfg.STATE.shape[0], cfg.STATE.shape[0],
                                       aux[0][0].real, aux[0][0].imag))

    for i in np.arange(1, cfg.STATE.shape[0] - 2, 2):
        f.write("%d %d %1.16f %1.16f\n" %
                (i + 1, i + 1, aux[0][0].real, aux[0][0].imag))
        f.write("%d %d %1.16f %1.16f\n" %
                (i + 2, i + 1, aux[1][0].real, aux[1][0].imag))
        f.write("%d %d %1.16f %1.16f\n" %
                (i + 1, i + 2, aux[0][1].real, aux[0][1].imag))
        f.write("%d %d %1.16f %1.16f\n" %
                (i + 2, i + 2, aux[1][1].real, aux[1][1].imag))
    f.close()


def teste_STAGGERED_EVEN_OPERATOR_1D():
    try:
        f = open("HIPERWALK_TEMP_STAGGERED_EVEN_OPERATOR_1D.dat", 'w')
    except IOError:
        print("[HIPERWALK] Could not open file in directory.")

    totalVerticesPerPatch = cfg.TESSELLATIONPOLYGONS[0]
    vertices_in_X = cfg.TESSELLATIONPOLYGONS[0]
    array_Index = np.zeros((totalVerticesPerPatch, 1), dtype=int)
    array_Values = np.zeros((totalVerticesPerPatch, 1), dtype=complex)

    #    cfg.Ueven=np.zeros((cfg.STATESIZE,cfg.STATESIZE),dtype=float)

    counter = 0
    for i in range(int(
            cfg.TOTAL_PATCHES_IN_X)):  ### Run in all patches in axis X

        counter = 0

        for x in range(
                vertices_in_X):  ### Run in all vertices per patches in axis X
            #                auxX=cfg.OVERLAPX*i+x+cfg.RANGEX[0]
            auxX = cfg.OVERLAPX * i + x + cfg.RANGEX[0]
            if cfg.GRAPHTYPE == "CYCLE":
                index = (auxX)
            elif cfg.GRAPHTYPE == "LINE":
                index = (auxX - cfg.RANGEX[0])

            array_Index[counter] = index
            counter = counter + 1

        counter = 0
        for k in range(0, cfg.STAGGERED_COEFICIENTS.shape[1], 2):
            array_Values[counter] = cfg.STAGGERED_COEFICIENTS[0][
                k] + 1J * cfg.STAGGERED_COEFICIENTS[0][k + 1]
            counter = counter + 1

        for i1 in range(totalVerticesPerPatch):
            for j1 in range(totalVerticesPerPatch):
                value = 2 * array_Values[i1] * np.conj(array_Values[j1])
                if (array_Index[i1] == array_Index[j1]):
                    value = value - 1
                f.write("%d %d %1.16f %1.16f\n" %
                        (array_Index[i1] + 1, array_Index[j1] + 1, value.real,
                         value.imag))


#
#                i2=array_Index[i1]
#                j2=array_Index[j1]
#                cfg.Ueven[i2,j2]=value
    f.close()


def teste_STAGGERED_ODD_OPERATOR_1D():
    try:
        f = open("HIPERWALK_TEMP_STAGGERED_ODD_OPERATOR_1D.dat", 'w')
    except IOError:
        print("[HIPERWALK] Could not open file in directory.")

    totalVerticesPerPatch = cfg.TESSELLATIONPOLYGONS[0]

    array_Index = np.zeros((totalVerticesPerPatch, 1), dtype=int)
    array_Values = np.zeros((totalVerticesPerPatch, 1), dtype=complex)

    #    cfg.Uodd=np.zeros((cfg.STATESIZE,cfg.STATESIZE),dtype=float)

    counter = 0

    #    print cfg.TESSELLATIONDISPLACEMENT

    for i in range(int(
            cfg.TOTAL_PATCHES_IN_X)):  ### Run in all patches in axis X
        counter = 0
        for x in range(int(cfg.TESSELLATIONPOLYGONS[0])
                       ):  ### Run in all vertices per patches in axis X
            auxX = cfg.OVERLAPX * i + x + cfg.RANGEX[
                0] + cfg.TESSELLATIONDISPLACEMENT[0]

            if cfg.GRAPHTYPE == "CYCLE":
                if auxX > cfg.RANGEX[1]:
                    auxX = cfg.RANGEX[0]
                index = (auxX)
            elif cfg.GRAPHTYPE == "LINE":
                if auxX > cfg.RANGEX[1]:
                    auxX = cfg.RANGEX[0]
                index = (auxX - cfg.RANGEX[0])

            array_Index[counter] = index
            counter = counter + 1
        counter = 0
        for k in range(0, cfg.STAGGERED_COEFICIENTS.shape[1], 2):
            array_Values[counter] = cfg.STAGGERED_COEFICIENTS[1][
                k] + 1J * cfg.STAGGERED_COEFICIENTS[1][k + 1]
            counter = counter + 1

        for i1 in range(totalVerticesPerPatch):
            for j1 in range(totalVerticesPerPatch):
                value = 2 * array_Values[i1] * np.conj(array_Values[j1])
                if (array_Index[i1] == array_Index[j1]):
                    value = value - 1
                f.write("%d %d %1.16f %1.16f\n" %
                        (array_Index[i1] + 1, array_Index[j1] + 1, value.real,
                         value.imag))

#                    i2=array_Index[i1]
#                    j2=array_Index[j1]
#                    cfg.Uodd[i2,j2]=value
    f.close()


###
### 2D
###
def STAGGERED2D():

    STAGGERED_EVEN_OPERATOR_2D()
    STAGGERED_ODD_OPERATOR_2D()


def STAGGERED_EVEN_OPERATOR_2D():
    f = open("HIPERWALK_TEMP_STAGGERED_EVEN_OPERATOR_2D.dat", 'w')

    totalVerticesPerPatch = cfg.TESSELLATIONPOLYGONS[
        0] * cfg.TESSELLATIONPOLYGONS[1]
    array_Index = np.zeros((totalVerticesPerPatch, 1), dtype=int)
    array_Values = np.zeros((totalVerticesPerPatch, 1), dtype=complex)

    index = 0

    #    cfg.Ueven=np.zeros((cfg.STATESIZE,cfg.STATESIZE),dtype=float)

    counter = 0
    for i in range(int(
            cfg.TOTAL_PATCHES_IN_X)):  ### Run in all patches in axis X
        for j in range(int(
                cfg.TOTAL_PATCHES_IN_Y)):  ### Run in all patches in axis Y
            counter = 0

            for x in range(int(cfg.TESSELLATIONPOLYGONS[0])
                           ):  ### Run in all vertices per patches in axis X
                for y in range(
                        int(cfg.TESSELLATIONPOLYGONS[1]
                            )):  ### Run in all vertices per patches in axis Y

                    auxX = cfg.OVERLAPX * i + x + cfg.RANGEX[0]
                    auxY = cfg.OVERLAPY * j + y + cfg.RANGEY[0]

                    if cfg.GRAPHTYPE == "TORUS":
                        index = (auxX) * (cfg.RANGEY[1] - cfg.RANGEY[0]) + (
                            auxY)
                    elif cfg.GRAPHTYPE == "LATTICE":
                        index = (auxX - cfg.RANGEX[0]) * (
                            cfg.RANGEY[1] - cfg.RANGEY[0] + 1) + (
                                auxY - cfg.RANGEY[0])

#                    print i,j,auxX,auxY,index

                    array_Index[counter] = index
                    counter = counter + 1

            counter = 0
            for k in range(0, cfg.STAGGERED_COEFICIENTS.shape[1], 2):
                array_Values[counter] = cfg.STAGGERED_COEFICIENTS[0][
                    k] + 1J * cfg.STAGGERED_COEFICIENTS[0][k + 1]
                counter = counter + 1

            for i1 in range(totalVerticesPerPatch):
                for j1 in range(totalVerticesPerPatch):
                    value = 2 * array_Values[i1] * np.conj(array_Values[j1])
                    if (array_Index[i1] == array_Index[j1]):
                        value = value - 1
                    f.write("%d %d %1.16f %1.16f\n" %
                            (array_Index[i1] + 1, array_Index[j1] + 1,
                             value.real, value.imag))


#
#                    i2=array_Index[i1]
#                    j2=array_Index[j1]
#                    cfg.Ueven[i2,j2]=value
    f.close()


def STAGGERED_ODD_OPERATOR_2D():
    f = open("HIPERWALK_TEMP_STAGGERED_ODD_OPERATOR_2D.dat", 'w')

    totalVerticesPerPatch = cfg.TESSELLATIONPOLYGONS[
        0] * cfg.TESSELLATIONPOLYGONS[1]
    array_Index = np.zeros((totalVerticesPerPatch, 1), dtype=int)
    array_Values = np.zeros((totalVerticesPerPatch, 1), dtype=complex)

    #    cfg.Uodd=np.zeros((cfg.STATESIZE,cfg.STATESIZE),dtype=float)

    index = 0
    counter = 0
    for i in range(int(
            cfg.TOTAL_PATCHES_IN_X)):  ### Run in all patches in axis X
        for j in range(int(
                cfg.TOTAL_PATCHES_IN_Y)):  ### Run in all patches in axis Y
            counter = 0
            for x in range(int(cfg.TESSELLATIONPOLYGONS[0])
                           ):  ### Run in all vertices per patches in axis X
                for y in range(
                        int(cfg.TESSELLATIONPOLYGONS[1]
                            )):  ### Run in all vertices per patches in axis Y
                    auxX = cfg.OVERLAPX * i + x + cfg.RANGEX[
                        0] + cfg.TESSELLATIONDISPLACEMENT[0]
                    auxY = cfg.OVERLAPY * j + y + cfg.RANGEY[
                        0] + cfg.TESSELLATIONDISPLACEMENT[1]

                    if cfg.GRAPHTYPE == "TORUS":
                        if auxX == cfg.RANGEX[1]:
                            auxX = cfg.RANGEX[0]
                        if auxY == cfg.RANGEY[1]:
                            auxY = cfg.RANGEY[0]
                        index = (auxX) * (cfg.RANGEY[1] - cfg.RANGEY[0]) + (
                            auxY)
                    elif cfg.GRAPHTYPE == "LATTICE":
                        if auxX > cfg.RANGEX[1]:
                            auxX = cfg.RANGEX[0]
                        if auxY > cfg.RANGEY[1]:
                            auxY = cfg.RANGEY[0]
                        index = (auxX - cfg.RANGEX[0]) * (
                            cfg.RANGEY[1] - cfg.RANGEY[0] + 1) + (
                                auxY - cfg.RANGEY[0])

                    array_Index[counter] = index
                    counter = counter + 1
            counter = 0
            for k in range(0, cfg.STAGGERED_COEFICIENTS.shape[1], 2):
                array_Values[counter] = cfg.STAGGERED_COEFICIENTS[1][
                    k] + 1J * cfg.STAGGERED_COEFICIENTS[1][k + 1]
                counter = counter + 1

            for i1 in range(totalVerticesPerPatch):
                for j1 in range(totalVerticesPerPatch):
                    value = 2 * array_Values[i1] * np.conj(array_Values[j1])
                    if (array_Index[i1] == array_Index[j1]):
                        value = value - 1
                    f.write("%d %d %1.16f %1.16f\n" %
                            (array_Index[i1] + 1, array_Index[j1] + 1,
                             value.real, value.imag))


#                    i2=array_Index[i1]
#                    j2=array_Index[j1]
#                    cfg.Uodd[i2,j2]=value
    f.close()


def generateOPERATORS_CUSTOM():
    qtdOperators = len(cfg.CUSTON_OPERATORS_NAME)
    cfg.CUSTOM_UNITARY = np.zeros(
        (cfg.STATESIZE, cfg.STATESIZE * qtdOperators), dtype=complex)

    for i in range(0, qtdOperators):
        currentLine = 0
        currenColumn = 0
        for line in open(cfg.CUSTON_OPERATORS_NAME[i], 'r'):

            if line == "\n":
                continue
            if line.startswith("#"):
                continue
            line = line.split()

            for j in range(0, 2 * cfg.STATESIZE, 2):
                cfg.CUSTOM_UNITARY[currentLine][
                    i * cfg.STATESIZE +
                    currenColumn] = float(line[j]) + 1J * float(line[j + 1])
                currenColumn = currenColumn + 1
            currenColumn = 0
            currentLine = currentLine + 1


#    print cfg.CUSTOM_UNITARY
#            op.check_Unitarity(cfg.COINOPERATOR,N)


def local_printS_2D(N):
    s = np.zeros([cfg.STATESIZE, cfg.STATESIZE], dtype=int)
    f = open("S.dat", "w")
    for k in range(4):
        for x in range(N):
            for y in range(N):
                if k == 0:
                    f.write("%d %d 1\n" % (T(
                        (x + 1) % N, y, iC(k), N) + 1, T(x, y, k, N) + 1))
                    s[T((x + 1) % N, y, iC(k), N)][T(x, y, k, N)] = 1
                elif k == 1:
                    f.write("%d %d 1\n" %
                            (T(x,
                               (y + 1) % N, iC(k), N) + 1, T(x, y, k, N) + 1))
                    s[T(x, (y + 1) % N, iC(k), N)][T(x, y, k, N)] = 1
                elif k == 2:
                    f.write("%d %d 1\n" % (T(
                        (x - 1) % N, y, iC(k), N) + 1, T(x, y, k, N) + 1))
                    s[T((x - 1) % N, y, iC(k), N)][T(x, y, k, N) + 1] = 1
                elif k == 3:
                    f.write("%d %d 1\n" %
                            (T(x,
                               (y - 1) % N, iC(k), N) + 1, T(x, y, k, N) + 1))
                    s[T(x, (y - 1) % N, iC(k), N)][T(x, y, k, N)] = 1

    return s


def printCtI_2D(N, Coin):
    f = open("CtI.dat", "w")
    for i in range(4):
        for j in range(4):
            for w in range(N * N):
                f.write("%d %d %f %f\n" %
                        (N * N * i + w + 1, N * N * j + w + 1, Coin[i][j].real,
                         Coin[i][j].imag))


D = 0
G = [[-0.5, 0.5, 0.5, 0.5], [0.5, -0.5, 0.5, 0.5], [0.5, 0.5, -0.5, 0.5],
     [0.5, 0.5, 0.5, -0.5]]
mI = [[-1, 0.0, 0.0, 0.0], [0.0, -1, 0.0, 0.0], [0.0, 0.0, -1, 0.0],
      [0.0, 0.0, 0.0, -1]]


def iC(k):
    return (k + 2) % 4


def T(x, y, k, N):
    if D == 1:
        return "(" + str(x) + "," + str(y) + ")"
    else:
        return k * N * N + x * N + y


def printD_2D(N):

    f = open("D.dat", "w")
    for x in range(N):
        for y in range(N):
            f.write("%f\n" % (np.sqrt(x * x + y * y)))


#
#def local_printD_2D( N ):
#    d=np.zeros([cfg.GRAPHSIZE,1],dtype=float)
#    f = open("D.dat", "w" )
#    i=0
#    for x in range( N ):
#        for y in range( N ):
#            f.write("%f\n"%(np.sqrt( x*x + y*y )))
#            d[i]=np.sqrt( x*x + y*y )
#            i=i+1
#    return d

#def printS_2D( N ):
#    f = open("S.dat", "w" )
#    for k in range( 4 ):
#        for x in range( N ):
#            for y in range( N ):
#                if k == 0:
#                    f.write("%d %d 1\n"%(T( (x + 1) % N, y, iC(k), N )+1, T( x, y, k, N )+1))
#                elif k == 1:
#                    f.write("%d %d 1\n"%(T( x, (y + 1) % N,iC(k), N )+1, T( x, y,k, N )+1))
#                elif k == 2:
#                    f.write("%d %d 1\n"%(T( (x-1) % N, y, iC(k),N )+1, T( x, y,k, N )+1))
#                elif k == 3:
#                    f.write("%d %d 1\n"%(T( x, (y - 1) % N,iC(k), N )+1, T( x, y,k, N )+1))
#########################################################################
#########################################################################
#########################################################################
