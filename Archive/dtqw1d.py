# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 16:19:56 2014

@author: aaron
"""
import config as cfg
import ioFunctions as io
#import numpy as np
import gnuplot as gnuplot
import neblina as nb
import operators as op
import standardDeviation as sd
import testmode
import numpy as np

def run():
    probabilities=[]

    io.savetxt("HIPERWALK_TEMP_PSI.dat",cfg.STATE,float,'%1.16f')
    op.COIN_TENSOR_IDENTITY_1D(cfg.COINOPERATOR,cfg.GRAPHSIZE)
    op.SHIFT_OPERATOR_1D(cfg.COINVECTORDIMENSION,cfg.GRAPHSIZE)
#    sd.distances_vector_1D(cfg.RANGEX[0],cfg.RANGEX[1])  
    cfg.DISTANCE_VECTOR_SIZE=cfg.GRAPHSIZE
#    io.savetxt("HIPERWALK_TEMP_DISTANCE_VECTOR.dat",cfg.DISTANCES_VECTOR,int,'%1.16f')    

#    nb.generating_DTQW1D_NBL()
    nb.runCore_DTQW1D()

    cfg.STATE=nb.neblina_state_to_vector("NEBLINA_TEMP_final_state.dat")
    probabilities=nb.neblina_distribution_to_vector("NEBLINA_TEMP_final_distribution.dat")
    output = open("final_distribution.dat",'w')
    output1 = open("final_state.dat",'w')
    output.write("#POSITION \t PROBABILITY\n")          
    output1.write("#POSITION \t COIN \t Re(amplitude) \t Im(amplitude)\n")        
    
    for j in range(cfg.COINVECTORDIMENSION):
        for i in range(int(cfg.GRAPHSIZE)):
            if j==0:
                output.write("%d \t \t \t%1.16f\n"%(cfg.RANGEX[0]+i,probabilities[i]))
            output1.write("%d \t \t \t \t %d\t \t %1.16f \t %1.16f\n"%(cfg.RANGEX[0]+i,j,cfg.STATE[i+j*cfg.GRAPHSIZE].real,cfg.STATE[i+j*cfg.GRAPHSIZE].imag))         

    output.close()
    output1.close()    
            

        





    if cfg.GNUPLOT:
        io.savetxt("HIPERWALK_TEMP_PROBABILITIES.dat",probabilities,float,'%1.16f')
        gnuplot.template_DTQW1D("HIPERWALK_TEMP_PROBABILITIES.dat","final_distribution.eps","EPS")
        if cfg.STEPS>1:
            gnuplot.plotStatistics1D()
        if cfg.ANIMATION == 1:
            gnuplot.plotAnimation1D()
        



    if cfg.TEST_MODE:
        modelVector=testmode.create_DTQW1D_test_vector()
        returnNeblina=nb.neblina_distribution_to_vector("NEBLINA_TEMP_final_distribution.dat")
        if np.linalg.norm(modelVector-returnNeblina,np.inf) == float(0):
            return 1
        else:
            return 0
            
    return 1