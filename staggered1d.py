# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 17:03:06 2014

@author: aaron
"""

import neblina as nb                ### Module for neblina interpreter.
import operators as op              ### Module for operators functions.
import ioFunctions as io            ### Module for write on disk functions.
import gnuplot as gnuplot  ### Module for Gnuplot functions.
import config as cfg                ### Module for global variables for Quantum Walk.
import standardDeviation as sd      ### Module for Standard Deviation functions.
import numpy as np
import testmode

def run():
    probabilities=[]

    
    
    if not cfg.OVERLAP:
        cfg.OVERLAPX=int(cfg.TESSELLATIONPOLYGONS[0])

        
    io.savetxt("HIPERWALK_TEMP_PSI.dat",cfg.STATE,float,'%1.16f')
    op.STAGGERED1D()


    sd.distances_vector_1D(cfg.RANGEX[0],cfg.RANGEX[1])  
    cfg.DISTANCE_VECTOR_SIZE=cfg.GRAPHSIZE
        
#    nb.generating_STAGGERED1D_NBL()
    nb.runCore_STAGGERED1D()
    
    cfg.STATE=nb.neblina_state_to_vector("NEBLINA_TEMP_final_state.dat")
    probabilities=nb.neblina_distribution_to_vector("NEBLINA_TEMP_final_distribution.dat")


    output = open("final_distribution.dat",'w')
    output1 = open("final_state.dat",'w')
    output.write("#POSITION \t PROBABILITY\n")          
    output1.write("#POSITION \t Re(amplitude) \t \t \t Im(amplitude)\n")        
    
    for i in range(int(cfg.GRAPHSIZE)):

        output.write("%d \t \t \t%1.16f\n"%(cfg.RANGEX[0]+i,probabilities[i]))
        output1.write("%d \t \t \t%1.16f\t\t\t%1.16f\n"%(cfg.RANGEX[0]+i,cfg.STATE[i].real,cfg.STATE[i].imag))
        
    output.close()
    output1.close()    



    if cfg.GNUPLOT:
        io.savetxt("HIPERWALK_TEMP_PROBABILITIES.dat",probabilities,float,'%1.16f')

        gnuplot.template_STAGGERED1D("HIPERWALK_TEMP_PROBABILITIES.dat","final_distribution.eps","EPS")
        if cfg.STEPS>1:
            gnuplot.plotStatistics1D()
        if cfg.ANIMATION == 1:
            gnuplot.plotAnimation1D()



    if cfg.TEST_MODE:
        modelVector=testmode.create_STAGGERED1D_test_vector()
        returnNeblina=nb.neblina_distribution_to_vector("NEBLINA_TEMP_final_distribution.dat")
        if np.linalg.norm(modelVector-returnNeblina,np.inf) == float(0):
            return 1
        else:
            return 0
            

    return 1