# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 22:38:21 2014

@author: aaron

File of functions for calculating STANDARD DEVIATIONS
"""
import numpy as np

import config as cfg

 

def distances_vector_1D(xi,xf):
     cfg.DISTANCES_VECTOR=np.array([np.arange(xi,xf+1,1)])
     cfg.DISTANCES_VECTOR=cfg.DISTANCES_VECTOR.transpose()

 
def generateDistance_CUSTOM():
    fileIn=str(cfg.CUSTOM_LABELS_NAME[0])
    
    qtdeValues = 0  
    for line in open(fileIn,'r'):  
        if line == "\n":  
            continue  
        if line.startswith("#"):  
            continue  
        qtdeValues = qtdeValues + 1

    cfg.GRAPHSIZE=qtdeValues



#            
#def save_XY_Vectors():
#    X = open("X.dat", "w" ) 
#    Y = open("Y.dat", "w" ) 
#    
#    for x in range( cfg.RANGEX[0],cfg.RANGEX[1]+1 ):
#        for y in range( cfg.RANGEY[0],cfg.RANGEY[1]+1 ):
#            X.write("%f\n"%(x))
#            Y.write("%f\n"%(y))
#    X.close()
#    Y.close()        
    
def calculateProbabilities():
    probabilities=np.zeros((cfg.GRAPHSIZE,1),dtype=float)
#    print cfg.COINVECTORDIMENSION
    for i in range(int(cfg.GRAPHSIZE)):
        a=0
        if cfg.WALK=="STAGGERED1D":
            a=cfg.STATE[i]*np.conjugate(cfg.STATE[i])

        else:
#            if cfg.WALK=="DTQW1D":
            for j in range(int(cfg.COINVECTORDIMENSION)):
                a=a+cfg.STATE[(j*cfg.GRAPHSIZE)+i]*np.conjugate(cfg.STATE[(j*cfg.GRAPHSIZE)+i])

        probabilities[i,0]=a.real
    return probabilities
    
    
    
def calculateProbabilities_STAGGERED():
    
    PROBABILITIES=np.zeros((cfg.GRAPHSIZE,1),dtype=float)
    for i in range(int(cfg.GRAPHSIZE)):
        PROBABILITIES[i]= (cfg.STATE[i]).real**2+(cfg.STATE[i]).imag**2
    return PROBABILITIES

def calculateProbabilities_COINED():
    
    PROBABILITIES=np.zeros((cfg.GRAPHSIZE,1),dtype=float)
    for c in range(cfg.COINVECTORDIMENSION):
        for i in range(int(cfg.GRAPHSIZE)):
            PROBABILITIES[i]= PROBABILITIES[i]+(cfg.STATE[c*cfg.GRAPHSIZE+i]).real*(cfg.STATE[c*cfg.GRAPHSIZE+i]).real+(cfg.STATE[c*cfg.GRAPHSIZE+i]).imag*(cfg.STATE[c*cfg.GRAPHSIZE+i]).imag
    return PROBABILITIES



    
def vec_prod(array1,array2):

    returnVector=np.zeros(array1.shape,dtype=float)
    aux=returnVector.shape[0]
    for i in range(aux):
        returnVector[i]=array1[i]*array2[i]
    return returnVector

def vec_sum(array):
    return np.sum(array)
    
def statistics(probabilities,i):
    tmp = vec_prod( cfg.DISTANCES_SQUARE, probabilities )
    
    s1  = vec_sum( tmp )    
    
    tmp = vec_prod( cfg.DISTANCES_VECTOR, probabilities )
    
    mean = vec_sum( tmp )
    
    stddev = np.sqrt(abs(s1-mean*mean))
    return mean,s1,stddev

def calculateProbabilities_COINED2D():
    PROBABILITIES=np.zeros((cfg.GRAPHSIZE,1),dtype=float)
    for c in range(cfg.COINVECTORDIMENSION):
        for i in range(cfg.GRAPHSIZE):
            PROBABILITIES[i]=PROBABILITIES[i]+cfg.STATE[c*cfg.GRAPHSIZE+i].real**2+cfg.STATE[c*cfg.GRAPHSIZE+i].imag**2
    return PROBABILITIES
    
def distances_vector_2D():
     cfg.DISTANCES_VECTOR_X=np.zeros((cfg.GRAPHSIZE,1),float)
     cfg.DISTANCES_VECTOR_Y=np.zeros((cfg.GRAPHSIZE,1),float)
     cfg.DISTANCES_VECTOR_SQUARE_X=np.zeros((cfg.GRAPHSIZE,1),float)
     cfg.DISTANCES_VECTOR_SQUARE_Y=np.zeros((cfg.GRAPHSIZE,1),float)
     
     i=0
     for x in range( cfg.RANGEX[0],cfg.RANGEX[1]+1 ):
         for y in range( cfg.RANGEY[0],cfg.RANGEY[1]+1 ):
            cfg.DISTANCES_VECTOR_X[i]=x
            cfg.DISTANCES_VECTOR_SQUARE_X[i]=x*x
            cfg.DISTANCES_VECTOR_Y[i]=y
            cfg.DISTANCES_VECTOR_SQUARE_Y[i]=y*y
            i=i+1
            
def satistics2D(probabilities):
    aux1=aux2=aux3=aux4=0

    for i in range(cfg.GRAPHSIZE):
        aux1=aux1+cfg.DISTANCES_VECTOR_X[i]*probabilities[i]
        aux2=aux2+cfg.DISTANCES_VECTOR_Y[i]*probabilities[i]
        aux3=aux3+cfg.DISTANCES_VECTOR_SQUARE_X[i]*probabilities[i]
        aux4=aux4+cfg.DISTANCES_VECTOR_SQUARE_Y[i]*probabilities[i]


    
    meanX=aux1
    meanY=aux2
    meanX2=aux3
    meanY2=aux4
    
    varX=meanX2-meanX*meanX
    varY=meanY2-meanY*meanY
    stdv=np.sqrt( varX + varY )


        
    return meanX,meanY,varX,varY,stdv
    
