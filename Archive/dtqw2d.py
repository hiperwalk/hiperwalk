import numpy as np
import config as cfg
import ioFunctions as io   
import neblina as nb
import operators as op
import gnuplot as gnuplot
import standardDeviation as sd
import testmode
import os
#from qwalk import test_qwalk as tq



def run():
    probabilities=[]

    if cfg.LATTYPE=="DIAGONAL":
        op.DIAGONAL_SHIFT_OPERATOR_2D()
    elif cfg.LATTYPE=="NATURAL":
        op.NATURAL_SHIFT_OPERATOR_2D()


    op.COIN_TENSOR_IDENTITY_OPERATOR_2D()

    cfg.DISTANCE_VECTOR_SIZE=cfg.GRAPHSIZE
    
    io.savetxt("HIPERWALK_TEMP_PSI.dat",cfg.STATE,float,'%1.16f')

#    nb.generating_DTQW2D_NBL()
    nb.runCore_DTQW2D()
    
    cfg.STATE=nb.neblina_state_to_vector("NEBLINA_TEMP_final_state.dat")
    probabilities=nb.neblina_distribution_to_vector("NEBLINA_TEMP_final_distribution.dat")

    output = open("final_distribution.dat",'w')
    output1 = open("final_state.dat",'w')
    output.write("POS X \t POS Y \t PROBABILITY\n")          
    output1.write("POS X \t POS Y \t COIN \t Re(amplitude)\t \t Im(amplitude)\n")   

#    probabilities=sd.calculateProbabilities_COINED2D()

    
    for x in range(cfg.RANGEX[0],cfg.RANGEX[1]+1,1):
        for y in range(cfg.RANGEY[0],cfg.RANGEY[1]+1,1):
            index=(x-cfg.RANGEX[0])*(cfg.RANGEX[1]-cfg.RANGEX[0]+1)+(y-cfg.RANGEY[0])

            output.write("%d\t %d\t %1.16f\n"%(x,y,probabilities[index]))
            for c in range(cfg.COINVECTORDIMENSION):
                output1.write("%d\t %d\t %d\t%1.16f \t %1.16f\n"%(x,y,c,cfg.STATE[cfg.GRAPHSIZE*c+index].real,cfg.STATE[cfg.GRAPHSIZE*c+index].imag))         
    output.close()
    output1.close()

    if cfg.GNUPLOT:
        io.savetxt("HIPERWALK_TEMP_PROBABILITIES",probabilities,float,'%1.16f')
        gnuplot.template_DTQW2D("HIPERWALK_TEMP_PROBABILITIES","final_distribution.eps","EPS")

        if cfg.STEPS>1:
            gnuplot.plotStatistics2d()
        if cfg.ANIMATION == 1:
            gnuplot.plotAnimation2D()



    if cfg.TEST_MODE:
        modelVector=testmode.create_DTQW2D_test_vector()
        returnNeblina=nb.neblina_distribution_to_vector("NEBLINA_TEMP_final_distribution.dat")
        if np.linalg.norm(modelVector-returnNeblina,np.inf) == float(0):
            return 1
        else:
            return 0
        
    return 1