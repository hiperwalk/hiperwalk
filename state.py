# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 23:25:47 2014

@author: aaron
"""
import numpy as np
import config as cfg


def generateState():

    if cfg.WALK=="DTQW1D":
        return generateState_DTQW1D()

    elif cfg.WALK=="DTQW2D":
        return generateState_DTQW2D()

    elif cfg.WALK=="COINLESS1D":
        return generateState_COINLESS1D()

    elif cfg.WALK=="COINLESS2D":
        return generateState_COINLESS2D()
        
    elif cfg.WALK=="CUSTOM":
        return generateState_CUSTOM()
        
def generateState_DTQW1D():
    
    if cfg.GRAPHTYPE=="CYCLE":
        rangeX=[int(0),int(cfg.GRAPHSIZE)-1]
        cfg.STATESIZE=cfg.COINVECTORDIMENSION * cfg.GRAPHSIZE

        cfg.STATE=np.zeros((cfg.STATESIZE,1),dtype=complex)
        for i in range(cfg.STATE_COMPONENTS.shape[0]):
            # Find the index of the state.                        
            index=cfg.STATE_COMPONENTS[i][2]*(cfg.GRAPHSIZE)+cfg.STATE_COMPONENTS[i][3]
            # Set the state[index] with the component state[index]=a+bi.
            cfg.STATE[index][0]=cfg.STATE[index][0]+cfg.STATE_COMPONENTS[i][0]+1J*cfg.STATE_COMPONENTS[i][1]
    
    elif cfg.GRAPHTYPE=="LINE":
        upperBound=return_MAX_Position(cfg.STATE_COMPONENTS,3)
        lowerBound=return_MIN_Position(cfg.STATE_COMPONENTS,3)
        rangeX=[int(lowerBound-cfg.STEPS) ,int(upperBound+cfg.STEPS)]
        cfg.GRAPHSIZE=( ( (upperBound-lowerBound) + 1) + 2*cfg.STEPS)
#        print cfg.COINVECTORDIMENSION 
        cfg.STATESIZE=int(cfg.COINVECTORDIMENSION * cfg.GRAPHSIZE)
        cfg.STATE=np.zeros((cfg.STATESIZE,1),dtype=complex)

    
        for i in range(cfg.STATE_COMPONENTS.shape[0]):
            # Find the index of the state.
            index=cfg.STATE_COMPONENTS[i][2]*cfg.GRAPHSIZE + cfg.STEPS + (cfg.STATE_COMPONENTS[i][3]-lowerBound)
            # Set the state[index] with the component state[index]=a+bi.
            cfg.STATE[index][0]=cfg.STATE[index][0]+cfg.STATE_COMPONENTS[i][0]+1J*cfg.STATE_COMPONENTS[i][1]
    cfg.RANGEX=rangeX   

    

def generateState_DTQW2D():



    if cfg.GRAPHTYPE=="TORUS":
        #checkBounderies of TORUS with initialState, if it exceed the size
        checkTorusBoundaries()
        if len(cfg.TORUSSIZE)==1:
            upperBoundX=int(cfg.TORUSSIZE[0])-1
            lowerBoundX=0
            cfg.RANGEX=[int(lowerBoundX) ,int(upperBoundX)]
            cfg.SIZEX=int(upperBoundX)
            upperBoundY=int(cfg.TORUSSIZE[0])-1
            lowerBoundY=0
            cfg.RANGEY=[int(lowerBoundY) ,int(upperBoundY)]
            cfg.SIZEY=int(upperBoundX)
            cfg.GRAPHSIZE=int(cfg.TORUSSIZE[0])*int(cfg.TORUSSIZE[0])
            
        elif len(cfg.TORUSSIZE)==2:
            upperBoundX=int(cfg.TORUSSIZE[0])-1
            lowerBoundX=0
            cfg.RANGEX=[int(lowerBoundX) ,int(upperBoundX)]
            cfg.SIZEX=int(upperBoundX)
            upperBoundY=int(cfg.TORUSSIZE[1])-1
            lowerBoundY=0
            cfg.RANGEY=[int(lowerBoundY) ,int(upperBoundY)]
            cfg.SIZEY=int(upperBoundY)
            cfg.GRAPHSIZE=int(cfg.TORUSSIZE[0])*int(cfg.TORUSSIZE[1])



    
    else:

        upperBoundX=return_MAX_Position(cfg.STATE_COMPONENTS,3)
        lowerBoundX=return_MIN_Position(cfg.STATE_COMPONENTS,3)
        cfg.RANGEX=[int(lowerBoundX-cfg.STEPS) ,int(upperBoundX+cfg.STEPS)]    
        cfg.sizeX=cfg.RANGEX[1]-cfg.RANGEX[0]+1
        
        upperBoundY=return_MAX_Position(cfg.STATE_COMPONENTS,4)
        lowerBoundY=return_MIN_Position(cfg.STATE_COMPONENTS,4)
        cfg.RANGEY=[int(lowerBoundY-cfg.STEPS) ,int(upperBoundY+cfg.STEPS)]
        cfg.sizeY=cfg.RANGEY[1]-cfg.RANGEY[0]+1

        cfg.GRAPHSIZE=cfg.sizeX*cfg.sizeY


#    print cfg.RANGEX
#    print cfg.RANGEY
#    
    cfg.STATESIZE=cfg.GRAPHSIZE*cfg.COINVECTORDIMENSION

    cfg.STATE=np.zeros((cfg.STATESIZE,1),dtype=complex)

    for i in range(cfg.STATE_COMPONENTS.shape[0]):
        # Find the index of the state.
        index=cfg.STATE_COMPONENTS[i][2]*cfg.GRAPHSIZE + (cfg.STATE_COMPONENTS[i][3]-cfg.RANGEX[0])*(cfg.RANGEY[1]-cfg.RANGEY[0]+1)+(cfg.STATE_COMPONENTS[i][4]-cfg.RANGEY[0])
        # Set the state[index] with the component state[index]=a+bi.
        cfg.STATE[index][0]=cfg.STATE[index][0]+cfg.STATE_COMPONENTS[i][0]+1J*cfg.STATE_COMPONENTS[i][1]

#def index():
#    (cfg.RANGEX[1]-cfg.RANGEX[0]+1)*(cfg.STATE_COMPONENTS[i][3]-cfg.RANGEX[0])+(cfg.STATE_COMPONENTS[i][4]-cfg.RANGEY[0])
def generateState_COINLESS1D():
    
    if cfg.GRAPHTYPE=="CYCLE":
#        if cfg.GRAPHSIZE%2==1:
#            print("COINLESS Quantum Walk allows only cycles with even sites.")
#            exit(-1)

        aux=cfg.GRAPHSIZE
        aux=np.ceil(aux/cfg.TESSELLATIONPOLYGONS[0])
        aux=aux*cfg.TESSELLATIONPOLYGONS[0]
        cfg.RANGEX=[0,aux-1]
        cfg.TOTAL_PATCHES_IN_X=np.ceil(aux/cfg.TESSELLATIONPOLYGONS[0])
        cfg.STATESIZE=cfg.RANGEX[1]+1
        cfg.STATE=np.zeros((cfg.STATESIZE,1),dtype=complex)
        for i in range(cfg.STATE_COMPONENTS.shape[0]):
            # Find the index of the state.                        
            index=cfg.STATE_COMPONENTS[i][2]
            # Set the state[index] with the component state[index]=a+bi.
            cfg.STATE[index][0]=cfg.STATE[index][0]+cfg.STATE_COMPONENTS[i][0]+1J*cfg.STATE_COMPONENTS[i][1]
    
    elif cfg.GRAPHTYPE=="LINE":



        upperBoundX=return_MAX_Position(cfg.STATE_COMPONENTS,2)
        lowerBoundX=return_MIN_Position(cfg.STATE_COMPONENTS,2)

        auxX0=lowerBoundX-cfg.TESSELLATIONPOLYGONS[0]*cfg.STEPS
        auxX1=upperBoundX+cfg.TESSELLATIONPOLYGONS[0]*cfg.STEPS
     
        auxX3=(auxX1-auxX0)+1
        auxX3=np.ceil(auxX3/cfg.TESSELLATIONPOLYGONS[0])### Roof of the division.
        cfg.TOTAL_PATCHES_IN_X=auxX3
        auxX3=auxX3*cfg.TESSELLATIONPOLYGONS[0]
        
        cfg.RANGEX=[int(auxX0) ,int(auxX0+auxX3-1)]    
        cfg.SIZEX=cfg.RANGEX[1]-cfg.RANGEX[0]+1







#        lowerBoundX=return_MIN_Position(cfg.STATE_COMPONENTS,2)
#        auxMin=lowerBoundX-cfg.TESSELLATIONPATCHES[0]*cfg.STEPS
#        auxMin=np.ceil(auxMin/cfg.TESSELLATIONPATCHES[0])*cfg.TESSELLATIONPATCHES[0]### Roof of the division.
#
#
#        upperBoundX=return_MAX_Position(cfg.STATE_COMPONENTS,2)
#        auxMax=upperBoundX+cfg.TESSELLATIONPATCHES[0]*cfg.STEPS
#        auxMax=np.ceil(auxMax/cfg.TESSELLATIONPATCHES[0])*cfg.TESSELLATIONPATCHES[0]-1### Roof of the division.
#        
#        cfg.RANGEX=[auxMin,auxMax]    
#        cfg.SIZEX=int(cfg.RANGEX[1]-cfg.RANGEX[0]+1)
#        cfg.TOTAL_PATCHES_IN_X=int(cfg.SIZEX/cfg.TESSELLATIONPATCHES[0])





#        if (lowerBound%2==1 and upperBound%2==0) or (lowerBound%2==0 and upperBound%2==1):
#            cfg.GRAPHSIZE=( ( (upperBound-lowerBound) + 51) + 2*cfg.STEPS)
##            print("IMPAR lb=%d ub=%d GRAPHSIZE=%d"%(lowerBound,upperBound,cfg.GRAPHSIZE))
#        else:
#            cfg.GRAPHSIZE=( ( (upperBound-lowerBound) + 52) + 2*cfg.STEPS)
##            print("PAR lb=%d ub=%d GRAPHSIZE=%d"%(lowerBound,upperBound,cfg.GRAPHSIZE))                

            
        cfg.STATESIZE=cfg.SIZEX
        cfg.STATE=np.zeros((cfg.STATESIZE,1),dtype=complex)

        for i in range(cfg.STATE_COMPONENTS.shape[0]):
            # Find the index of the state.
            index=(cfg.STATE_COMPONENTS[i][2]-cfg.RANGEX[0])
            # Set the state[index] with the component state[index]=a+bi.
            cfg.STATE[index][0]=cfg.STATE[index][0]+cfg.STATE_COMPONENTS[i][0]+1J*cfg.STATE_COMPONENTS[i][1]
        


def generateState_COINLESS2D():

    if cfg.GRAPHTYPE=="TORUS":
        
        #checkBounderies of TORUS with initialState, if it exceed the size
        checkTorusBoundaries()
        if len(cfg.TORUSSIZE)==1:

            upperBoundX=int(cfg.TORUSSIZE[0])-1
            lowerBoundX=0
            auxX3=(upperBoundX-lowerBoundX)+1
            auxX3=np.ceil(auxX3/cfg.TESSELLATIONPOLYGONS[0])
            cfg.TOTAL_PATCHES_IN_X=auxX3
            cfg.TOTAL_PATCHES_IN_Y=auxX3
            upperBoundX=cfg.TOTAL_PATCHES_IN_X*cfg.TESSELLATIONPOLYGONS[0]
            cfg.RANGEX=[int(lowerBoundX) ,int(upperBoundX)]
            cfg.SIZEX=int(upperBoundX)
            cfg.RANGEY=cfg.RANGEX
            cfg.SIZEY=cfg.SIZEX
            cfg.GRAPHSIZE=int(cfg.cfg.SIZEX*cfg.SIZEY)

            
        elif len(cfg.TORUSSIZE)==2:
           
            upperBoundX=int(cfg.TORUSSIZE[0])-1
            lowerBoundX=0
            cfg.RANGEX=[int(lowerBoundX) ,int(upperBoundX)]
            cfg.SIZEX=int(upperBoundX)
            upperBoundY=int(cfg.TORUSSIZE[1])-1
            lowerBoundY=0
            cfg.RANGEY=[int(lowerBoundY) ,int(upperBoundY)]
            cfg.SIZEY=int(upperBoundY)
            cfg.GRAPHSIZE=int(cfg.TORUSSIZE[0])*int(cfg.TORUSSIZE[1])


    
    elif cfg.GRAPHTYPE=="LATTICE":



        upperBoundX=return_MAX_Position(cfg.STATE_COMPONENTS,2)
        lowerBoundX=return_MIN_Position(cfg.STATE_COMPONENTS,2)
        auxX0=lowerBoundX-cfg.TESSELLATIONPOLYGONS[0]*cfg.STEPS
        auxX1=upperBoundX+cfg.TESSELLATIONPOLYGONS[0]*cfg.STEPS
        auxX3=(auxX1-auxX0)+1
        auxX3=np.ceil(auxX3/cfg.TESSELLATIONPOLYGONS[0])### Roof of the division.
        cfg.TOTAL_PATCHES_IN_X=auxX3
        auxX3=auxX3*cfg.TESSELLATIONPOLYGONS[0]
        
        cfg.RANGEX=[int(auxX0) ,int(auxX0+auxX3-1)]    

        cfg.SIZEX=cfg.RANGEX[1]-cfg.RANGEX[0]+1




        upperBoundY=return_MAX_Position(cfg.STATE_COMPONENTS,3)
        lowerBoundY=return_MIN_Position(cfg.STATE_COMPONENTS,3)        
        auxY0=lowerBoundY-cfg.TESSELLATIONPOLYGONS[1]*cfg.STEPS
        auxY1=upperBoundY+cfg.TESSELLATIONPOLYGONS[1]*cfg.STEPS
        auxY3=(auxY1-auxY0)+1
        auxY3=np.ceil(auxY3/cfg.TESSELLATIONPOLYGONS[1])
        cfg.TOTAL_PATCHES_IN_Y=auxY3
        auxY3=auxY3*cfg.TESSELLATIONPOLYGONS[1]
        cfg.RANGEY=[int(auxY0) ,int(auxY0+auxY3-1)]    
        cfg.SIZEY=cfg.RANGEY[1]-cfg.RANGEY[0]+1

        cfg.GRAPHSIZE=cfg.SIZEX*cfg.SIZEY

    cfg.STATESIZE=cfg.GRAPHSIZE
    cfg.STATE=np.zeros((cfg.STATESIZE,1),dtype=complex)

    for i in range(cfg.STATE_COMPONENTS.shape[0]):
        # Find the index of the state.
        index=(cfg.STATE_COMPONENTS[i][2]-cfg.RANGEX[0])*(cfg.RANGEY[1]-cfg.RANGEY[0]+1)+(cfg.STATE_COMPONENTS[i][3]-cfg.RANGEY[0])
        index=int(index)
        # Set the state[index] with the component state[index]=a+bi.
        cfg.STATE[index][0]=cfg.STATE[index][0]+cfg.STATE_COMPONENTS[i][0]+1J*cfg.STATE_COMPONENTS[i][1]




def generateState_CUSTOM():

    fileIn=str(cfg.CUSTOM_INITIALSTATE_NAME)
    qtdeValues = 0  
    for line in open(fileIn,'r'):  
        if line == "\n":  
            continue  
        if line.startswith("#"):  
            continue  
        qtdeValues = qtdeValues + 1

    cfg.STATESIZE=qtdeValues





#    cfg.STATE=np.zeros((qtdeValues,1),dtype=complex)
#    actual = 0  
#    
#    for line in open(fileIn):  
#        if line == "\n":  
#            continue  
#        if line.startswith("#"):  
#            continue          
#        line = line.split()  
#        a=float(line[0])
#        b=1J*float(line[1])
#
#        cfg.STATE[actual] = a+b
#        actual = actual + 1
#    
#    
#    print cfg.STATE
#


def return_MAX_Position(array,position):
    return max(array[:,position])
    
def return_MIN_Position(array,position):
    return min(array[:,position])
    
    
    
    
def checkTorusBoundaries():

    if cfg.WALK=="DTQW2D":
        auxX=cfg.STATE_COMPONENTS[:,3]
        auxY=cfg.STATE_COMPONENTS[:,4]
    elif cfg.WALK=="COINLESS2D":

        

        auxX=cfg.STATE_COMPONENTS[:,2]
        auxY=cfg.STATE_COMPONENTS[:,3]

    if  auxX.any()< 0 or auxY.any<0:
        print("[HIPERWALK] GRAPH TORUS only accept positives sites.")
        exit(-1)


    if len(cfg.TORUSSIZE)==1:

        if cfg.WALK=="COINLESS2D":
            if cfg.TORUSSIZE[0] % cfg.TESSELLATIONPOLYGONS[0] !=0 or cfg.TORUSSIZE[0] % cfg.TESSELLATIONPOLYGONS[1] !=0:
                print("[HIPERWALK] TORUS size must be multiple of number of patches")
                exit(-1)

        for i in auxX:
            if i >= int(cfg.TORUSSIZE[0]):
                print("[HIPERWALK] BEGINSTATE exceed TORUS boundaries.")
                exit(-1)
    
        for i in auxY:
            if i >= int(cfg.TORUSSIZE[0]):
                print("[HIPERWALK] BEGINSTATE exceed TORUS boundaries.")
                exit(-1)

    if len(cfg.TORUSSIZE)==2:

        if cfg.WALK=="COINLESS2D":
            if cfg.TORUSSIZE[0] % cfg.TESSELLATIONPOLYGONS[0] !=0 or cfg.TORUSSIZE[1] % cfg.TESSELLATIONPOLYGONS[1] !=0:
                print("[HIPERWALK] TORUS size must be multiple of number of patches")
                exit(-1)        
        
        for i in auxX:
            if i >= int(cfg.TORUSSIZE[0]):
                print("[HIPERWALK] BEGINSTATE exceed TORUS boundaries.")
                exit(-1)
    
        for i in auxY:
            if i >= int(cfg.TORUSSIZE[1]):
                print("[HIPERWALK] BEGINSTATE exceed TORUS boundaries.")
                exit(-1)
                
                
                