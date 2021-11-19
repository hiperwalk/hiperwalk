# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 00:44:17 2014

@author: aaron
"""
import config as cfg
import distance as dist
import numpy as np
import state as st
import operators as op
import standardDeviation as sta
import os
import os.path

def preParsing(inputFile):
    f=open(inputFile,'r')
    while True:
        line=f.readline()
        line=str.upper(line)
        if not line: break     ### EOF ###
        line=line.split()
        if line!='\n' and line:



            if line[0]=="WALK":
                if line.__len__()==1:
                    print("[HIPERWALK] Syntax error at  WALK BLOCK, missing type.")
                    exit(-1)
    
                elif (line[1] != "DTQW") and (line[1] != "STAGGERED") and (line[1] != "SZEGEDY") and (line[1] != "CUSTOM"):
                    print("[HIPERWALK] Syntax error at  WALK BLOCK, unknown walk type, %s."%line[1])
                    exit(-1)
                line[1]=str.upper(line[1])
                cfg.WALK=line[1]
    
        
            elif line[0]=="STEPS":
                if line.__len__()==1:
                    print("[HIPERWALK] Syntax error at STEPS BLOCK.")
                    exit(-1)
                cfg.STEPS=int(line[1])



            elif line[0]=="GRAPH":
                line[1]=str.upper(line[1])
                if line.__len__()==1:
                    print("[HIPERWALK] Syntax error at  GRAPH BLOCK, missing type.")
                    exit(-1)
                elif line[1]=="CYCLE":
                    if line.__len__()==2:
                        print("[HIPERWALK] Syntax error at  GRAPH BLOCK, missing number of sites.")
                        exit(-1)                
                    cfg.GRAPHTYPE="CYCLE"
                    cfg.GRAPHSIZE=int(line[2])
                    if cfg.WALK=="DTQW":                
                        cfg.WALK="DTQW1D"
                    elif cfg.WALK=="STAGGERED":
                        cfg.WALK="STAGGERED1D"
    
                elif line[1]=="LINE":
                    cfg.GRAPHSIZE=int(2*cfg.STEPS+1)
                    cfg.GRAPHTYPE="LINE"                
                    if cfg.WALK=="DTQW":                
                        cfg.WALK="DTQW1D"
                    elif cfg.WALK=="STAGGERED":
                        cfg.WALK="STAGGERED1D"
                
                elif line[1]=="TORUS":
                    if line.__len__()==2:
                        print("[HIPERWALK] Syntax error at  GRAPH BLOCK, missing size of TORUS.")
                        exit(-1)  
                    cfg.TORUSSIZE = [int(row)  for row in line[2:]]
                    cfg.GRAPHTYPE="TORUS"
                    if cfg.WALK=="DTQW":                
                        cfg.WALK="DTQW2D"
                    elif cfg.WALK=="STAGGERED":
                        cfg.WALK="STAGGERED2D"
    
                elif line[1]=="LATTICE":
                    cfg.TORUSSIZE = [int(row)  for row in line[2:]]
                    cfg.GRAPHTYPE="LATTICE"
                    if cfg.WALK=="DTQW":                
                        cfg.WALK="DTQW2D"
                    elif cfg.WALK=="STAGGERED":
                        cfg.WALK="STAGGERED2D"
    
                else:
                    print("[HIPERWALK] Syntax ERROR on GRAPH BLOCK, unknown type.")
                    exit(-1)



            elif line[0]=="GRAPHDIMENSION":
                if line.__len__()==1:
                    print("[HIPERWALK] Syntax error at  GRAPHDIMENSION BLOCK, missing value.")            
                if int(line[1])<= 0:
                    print("[HIPERWALK] Syntax error at  GRAPHDIMENSION BLOCK, value must be positive integer.")
                    exit(-1)
                else:
                    cfg.GRAPHDIMENSION=int(line[1])
                
                
            elif line[0]=="POLYGONS":
                if line.__len__()==1:
                    print("[HIPERWALK] Syntax error at POLYGONS BLOCK, missing values.")
                    exit(-1)
                elif cfg.WALK=="STAGGERED1D" and len(line)==2:
                    cfg.TESSELLATIONPOLYGONS=[int(line[1])]
                    cfg.NUMBER_OF_COEFICIENTS=int(line[1])
                
                elif cfg.WALK=="STAGGERED2D" and len(line)==3:
                    cfg.TESSELLATIONPOLYGONS=[int(line[1]),int(line[2])]
                    cfg.NUMBER_OF_COEFICIENTS=int(line[1])*int(line[2])
                else:
                    print("[HIPERWALK] Syntax error at TESSELLATIONPOLYGONS BLOCK, too many arguments.")
                    exit(-1)
            
def parsingFile(inputFile):

    f=open(inputFile,'r')
    while True:
        line=f.readline()

        if not line: break     ### EOF ###
        line=line.split()
        if line!='\n' and line:
            parsingLines(line,f)

def parsingLines(line,f):
        line[0]=str.upper(line[0])




        if line[0]=="LATTYPE":
            if line.__len__()==1:
                print("[HIPERWALK] Syntax error at  LATTYPE, missing lattice type.")
                exit(-1)       
            if (line[1] != "NATURAL") and (line[1] != "DIAGONAL") and (line[1] != "TEST"):
                print("[HIPERWALK] Syntax error at  LATTYPE, unknown lattice type, %s"%line[1])
                exit(-1)       
            cfg.LATTYPE=line[1]


     

        elif line[0]=="BEGINSTATE":
            cfg.STATE_COMPONENTS=[]
            totalComponents=0
            while line[0]!="ENDSTATE":

                line=f.readline()

                while line=="\n":     ### Parsing \n
                    line=f.readline()
                    
                line=str.upper(line)
                line=line.split()

                if line[0]!="ENDSTATE":
                    totalComponents=totalComponents+1
                    if cfg.WALK=="DTQW1D":                        
                        cfg.STATE_COMPONENTS=np.append(cfg.STATE_COMPONENTS,[float(line[0]),float(line[1]),int(line[2]),int(line[3])],0)
                    elif cfg.WALK=="DTQW2D":                        
                        cfg.STATE_COMPONENTS=np.append(cfg.STATE_COMPONENTS,[float(line[0]),float(line[1]),int(line[2]),int(line[3]),int(line[4])],0)                            
                    elif cfg.WALK=="STAGGERED1D": 
                        cfg.STATE_COMPONENTS=np.append(cfg.STATE_COMPONENTS,[float(line[0]),float(line[1]),int(line[2])],0)                            
                    elif cfg.WALK=="STAGGERED2D": 
                        cfg.STATE_COMPONENTS=np.append(cfg.STATE_COMPONENTS,[float(line[0]),float(line[1]),int(line[2]),int(line[3])],0)                        
        
            if cfg.WALK=="DTQW1D":
                cfg.STATE_COMPONENTS.shape=totalComponents,4
                aux=int((st.return_MAX_Position(cfg.STATE_COMPONENTS,2)+1))
                if aux==0:
                    aux=aux+2
                elif aux==1:
                    aux=aux+1
                cfg.COINVECTORDIMENSION=aux
                
            if cfg.WALK=="DTQW2D":
                cfg.STATE_COMPONENTS.shape=totalComponents,5
                cfg.COINVECTORDIMENSION=4

            if cfg.WALK=="STAGGERED1D":
                cfg.STATE_COMPONENTS.shape=totalComponents,3

            if cfg.WALK=="STAGGERED2D":
                cfg.STATE_COMPONENTS.shape=totalComponents,4
            st.generateState()



                
        elif line[0]=="BEGINCOIN" :
            line=f.readline()
            while line=="\n":     ### Parsing \n
                line=f.readline()                
            line=str.upper(line)
            line=line.split()
            
            if line[0]=="IDENTITY":
                cfg.COINOPERATORNAME="IDENTITY"
                cfg.COINOPERATOR=op.identity(int(line[1]))
            elif line[0]=="HADAMARD":
                cfg.COINOPERATORNAME="HADAMARD"
                cfg.COINOPERATOR=op.hadamard(int(line[1]))
            elif line[0]=="FOURIER":
                cfg.COINOPERATORNAME="FOURIER"
                cfg.COINOPERATOR=op.fourier(int(line[1]))
            elif line[0]=="GROVER":
                cfg.COINOPERATORNAME="GROVER"
                cfg.COINOPERATOR=op.grover(int(line[1]))
            else:
                N=len(line)/2
                cfg.COINOPERATOR=np.zeros((N,N),dtype=complex)
                for i in range(N):
                    aux=0
                    for j in range(0,2*N,2):
                        cfg.COINOPERATOR[i][aux]=float(line[j])+1J*float(line[j+1])
                        aux=aux+1
                    line=f.readline()
                    line=line.split()
                op.check_Unitarity(cfg.COINOPERATOR,N)
            



        elif line[0]=="BEGINTESSELLATION":
            totalComponents=0
            while line[0]!="ENDTESSELLATION":

                line=f.readline()
                while line=="\n":     ### Parsing \n
                    line=f.readline()
                line=str.upper(line)
                line=line.split()

                if line[0]!="ENDTESSELLATION":
                    if len(line)==2*cfg.NUMBER_OF_COEFICIENTS:
                        st.checkUnitarity(line)
                        cfg.STAGGERED_COEFICIENTS=np.append(cfg.STAGGERED_COEFICIENTS,[float(row)  for row in line],1)

                        line=f.readline()
                        line=str.upper(line)
                        line=line.split()
                        st.checkUnitarity(line)
                        if len(line)==2*cfg.NUMBER_OF_COEFICIENTS:
                            cfg.STAGGERED_COEFICIENTS=np.append(cfg.STAGGERED_COEFICIENTS,[float(row)  for row in line],1)
                        else:
                            print("[HIPERWALK] Syntax error at BEGINTESSELLATION BLOCK, invalid number of values.")
                            exit(-1)
                            
                        cfg.STAGGERED_COEFICIENTS.shape=2,cfg.NUMBER_OF_COEFICIENTS*2


                    else:
                        print("[HIPERWALK] Syntax error at BEGINTESSELLATION BLOCK, invalid number of values.")
                        exit(-1)




        elif line[0]=="DISPLACEMENT":
            if line.__len__()==1:
                print("[HIPERWALK] Syntax error at  DISPLACEMENT BLOCK, missing values.")
                exit(-1)
            elif cfg.WALK=="STAGGERED1D":
                cfg.TESSELLATIONDISPLACEMENT=[int(line[1])]
            elif cfg.WALK=="STAGGERED2D":
                cfg.TESSELLATIONDISPLACEMENT=[int(line[1]),int(line[2])]




        elif line[0]=="DIRECTORY":
            if line.__len__()==1:
                print("[HIPERWALK] Syntax error at  DIRECTORY BLOCK, missing name.")
                exit(-1)
            cfg.DIRECTORY=line[1]



        elif line[0]=="ANIMATION":
            if line.__len__()==1:
                print("[HIPERWALK] Syntax error at  ANIMATION BLOCK, missing name.")
                exit(-1)
            if line[1]=="TRUE":
                cfg.ANIMATION=1
            elif line[1]=="FALSE":
                cfg.ANIMATION=0
            else:
                print("[HIPERWALK] Syntax error at  ANIMATION BLOCK, unknown value, %s."%line[1])
                exit(-1)


        elif line[0]=="DELAY":
            if line.__len__()==1:
                print("[HIPERWALK] Syntax error at  DELAY BLOCK, missing value.")
                exit(-1)
            if int(line[1])<= 0:
                print("[HIPERWALK] Syntax error at  DELAY BLOCK, value must be positive integer.")
                exit(-1)
            cfg.DELAY=int(line[1])


        elif line[0]=="SIMULATION":
            line[1]=str.upper(line[1])
            if line.__len__()==1:
                print("[HIPERWALK] Syntax error at  SIMULATION BLOCK, missing simulation type.")
                exit(-1)
            if (line[1] != "LOCAL") and (line[1] != "PARALLEL"):
                print("[HIPERWALK] Syntax error at  SIMULATION BLOCK, unknown type. %s"%line[1])
                exit(-1)
            cfg.SIMULATIONTYPE=line[1]
            




        elif line[0]=="HARDWAREID":
                if line.__len__()==1:
                    print("[HIPERWALK] Syntax error at  HARDWAREID BLOCK, missing platform ID.")
                    exit(-1)
                cfg.HARDWAREID=int(line[1])

                
        elif line[0]=="PLOTTING":
            if line.__len__()==1:
                print("[HIPERWALK] Syntax error at  PLOTTING BLOCK, missing name.")
                exit(-1)
            line[1]=str.upper(line[1])                
            if line[1]=="ZEROS":
                cfg.PLOTTING_ZEROS=1
            
            
        elif line[0]=="PLOTS":
            if line.__len__()==1:
                print("[HIPERWALK] Syntax error at  PLOTS BLOCK, missing value.")            
                exit(-1)
            if line[1]=="TRUE":
                cfg.GNUPLOT=1
            elif line[1]=="FALSE":
                cfg.GNUPLOT=0
            else:
                print("[HIPERWALK] Syntax error at  PLOTS BLOCK, unknown value, %s."%line[1])
                exit(-1)


        elif line[0]=="VARIANCE":
            if line.__len__()==1:
                print("[HIPERWALK] Syntax error at  VARIANCE BLOCK, missing value.")            
                exit(-1)
            if line[1]=="TRUE":
                cfg.VARIANCE=1
            elif line[1]=="FALSE":
                cfg.VARIANCE=0
            else:
                print("[HIPERWALK] Syntax error at  VARIANCE BLOCK, unknown value, %s."%line[1])
                exit(-1)
       
       
        elif line[0]=="ALLSTATES":
            if line.__len__()==1:
                print("[HIPERWALK] Syntax error at  ALLSTATES BLOCK, missing value.")            
            if int(line[1])<= 0:
                print("[HIPERWALK] Syntax error at  ALLSTATES BLOCK, value must be positive integer.")
                exit(-1)
            else:
                cfg.ALLSTATES=1
                cfg.SAVE_STATES_MULTIPLE_OF_N=int(line[1])
        
        elif line[0]=="ALLPROBS":
            if line.__len__()==1:
                print("[HIPERWALK] Syntax error at  ALLPROBS BLOCK, missing value.")            
            if int(line[1])<= 0:
                print("[HIPERWALK] Syntax error at  ALLPROBS BLOCK, value must be positive integer.")
                exit(-1)
            else:
                cfg.ALLPROBS=1
                cfg.SAVE_PROBS_MULTIPLE_OF_N=int(line[1])
        elif line[0]=="INITIALSTATE":
            
            if line.__len__()==1:
                print("[HIPERWALK] Syntax error at  INITIALSTATE, missing filename.")
                exit(-1)

            if not os.path.isfile( line[1] ) :
                print("[HIPERWALK] Error at  INITIALSTATE, file '%s' not found."%(line[1]))
                exit(-1)
                
            a=str(os.path.abspath(str(line[1])))
            cfg.CUSTOM_INITIALSTATE_NAME=a
            st.generateState_CUSTOM()
            
        elif line[0]=="ADJMATRIX":
            if line.__len__()==1:
                print("[HIPERWALK] Syntax error at  ADJMATRIX, missing filename.")
                exit(-1)

            if not os.path.isfile( line[1] ) :
                print("[HIPERWALK] Error at  ADJMATRIX, file '%s' not found."%(line[1]))
                exit(-1)
                
            a=str(os.path.abspath(str(line[1])))
            cfg.ADJMATRIX_PATH=a
        

        elif line[0]=="OVERLAP":
            if line.__len__()==1:
                print("[HIPERWALK] Syntax error at  OVERLAP BLOCK, missing value.")            
                exit(-1)
            if line[1]=="TRUE":
                cfg.OVERLAP=1
            elif line[1]=="FALSE":
                cfg.OVERLAP=0
            else:
                print("[HIPERWALK] Syntax error at  OVERLAP BLOCK, unknown value, %s."%line[1])
                exit(-1)



        elif line[0]=="UNITARY":
             if line.__len__()==1:
                print("[HIPERWALK] Syntax error at  UNITARY, missing filename.")
                exit(-1)
             for i in range( 1, len( line) ):
                if( os.path.isfile( line[i] ) == False ):
                    print("[HIPERWALK] Error at  UNITARY, file '%s' not found."%(line[i]))
                    exit(-1)

                a=str(os.path.abspath(str(line[i])))
                cfg.CUSTON_OPERATORS_NAME.append( a )
                cfg.CUSTOM_UNITARY_COUNTER=len(line)-1

        elif line[0]=="LABELS":
             if line.__len__()==1:
                print("[HIPERWALK] Syntax error at  LABELS , missing filename.")
                exit(-1)
             for i in range( 1, len( line) ):
                if( os.path.isfile( line[i] ) == False ):
                    print("[HIPERWALK] Error at  LABELS, file '%s' not found."%(line[i]))
                    exit(-1)

                a=str(os.path.abspath(str(line[i])))
                cfg.CUSTOM_LABELS_NAME.append( a )
                cfg.CUSTOM_LABELS_NAME_COUNTER=len(line)-1
             sta.generateDistance_CUSTOM()


        elif line[0]=="DEBUG":
            cfg.DEBUG=1               
