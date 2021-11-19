# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 16:45:09 2015

@author: Aaron Leao
         aaron@lncc.br
"""
import operators as op
import config as cfg
import ioFunctions as io 
import neblina as nb
import gnuplot as gnuplot
import testmode
import numpy as np
def run():
    
    if not cfg.OVERLAP:
        cfg.OVERLAPX=int(cfg.TESSELLATIONPOLYGONS[0])
        cfg.OVERLAPY=int(cfg.TESSELLATIONPOLYGONS[1])


    op.STAGGERED2D()


#    i1=np.dot(cfg.Ueven,np.conjugate(cfg.Ueven))
#    i2=np.dot(cfg.Uodd,np.conjugate(cfg.Uodd))
#    io.savetxt("A.dat",i1,float,'%1.1f')
#    io.savetxt("B.dat",i2,float,'%1.1f')
    
    cfg.DISTANCE_VECTOR_SIZE=cfg.GRAPHSIZE
    
    io.savetxt("HIPERWALK_TEMP_PSI.dat",cfg.STATE,float,'%1.16f')

    nb.runCore_STAGGERED2D()
    
    cfg.STATE=nb.neblina_state_to_vector("NEBLINA_TEMP_final_state.dat")
    probabilities=nb.neblina_distribution_to_vector("NEBLINA_TEMP_final_distribution.dat")

    output = open("final_distribution.dat",'w')
    output1 = open("final_state.dat",'w')
    output.write("POS X \t POS Y \t PROBABILITY\n")          
    output1.write("POS X \t POS Y \t Re(amplitude)\t \t Im(amplitude)\n")   



    if cfg.GRAPHTYPE=="TORUS":
        for x in range(cfg.RANGEX[0],cfg.RANGEX[1],1):
            for y in range(cfg.RANGEY[0],cfg.RANGEY[1],1):
                index=(x)*(cfg.RANGEX[1])+(y)
#                print index, x, y
                output.write("%d\t %d\t %1.16f\n"%(x,y,probabilities[index]))
                output1.write("%d\t %d\t %1.16f \t %1.16f\n"%(x,y,cfg.STATE[index].real,cfg.STATE[index].imag))         
        output.close()
        output1.close()

    elif cfg.GRAPHTYPE=="LATTICE":
        for x in range(cfg.RANGEX[0],cfg.RANGEX[1]+1,1):
            for y in range(cfg.RANGEY[0],cfg.RANGEY[1]+1,1):
                index=(x-cfg.RANGEX[0])*cfg.SIZEY+(y-cfg.RANGEY[0])    
                output.write("%d\t %d\t %1.16f\n"%(x,y,probabilities[index]))
                output1.write("%d\t %d\t %1.16f \t %1.16f\n"%(x,y,cfg.STATE[index].real,cfg.STATE[index].imag))         
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
        modelVector=testmode.create_STAGGERED2D_test_vector()
        returnNeblina=nb.neblina_distribution_to_vector("NEBLINA_TEMP_final_distribution.dat")
        if np.linalg.norm(modelVector-returnNeblina,np.inf) == float(0):
            return 1
        else:
            return 0
                
    return 1
