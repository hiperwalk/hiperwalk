# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 04:56:43 2014

@author: aaron
"""

import os
import subprocess
import numpy as np
import config as cfg

    
def runCore_DTQW1D():
    if cfg.TEST_MODE:
        os.system("neblina -id %d HIPERWALK_TEMP_DTQW1D.nbl %d %d %d %d %d %d %d >> /dev/null "%(cfg.HARDWAREID,cfg.STEPS,cfg.STATESIZE,cfg.GRAPHSIZE,cfg.DISTANCE_VECTOR_SIZE,cfg.SAVE_STATES_MULTIPLE_OF_N,cfg.ANIMATION,cfg.STATESIZE/2))
    else:
        os.system("neblina -id %d HIPERWALK_TEMP_DTQW1D.nbl %d %d %d %d %d %d %d"%(cfg.HARDWAREID,cfg.STEPS,cfg.STATESIZE,cfg.GRAPHSIZE,cfg.DISTANCE_VECTOR_SIZE,cfg.SAVE_STATES_MULTIPLE_OF_N,cfg.ANIMATION,cfg.STATESIZE/2))
        
def runCore_DTQW2D():
    if cfg.TEST_MODE:
        os.system("neblina -id %d HIPERWALK_TEMP_DTQW2D.nbl %d %d %d %d %d %d %d >> /dev/null "%(cfg.HARDWAREID,cfg.STEPS,cfg.STATESIZE,cfg.GRAPHSIZE,cfg.DISTANCE_VECTOR_SIZE,cfg.SAVE_STATES_MULTIPLE_OF_N,cfg.ANIMATION,cfg.STATESIZE/4))
    else:
        os.system("neblina -id %d HIPERWALK_TEMP_DTQW2D.nbl %d %d %d %d %d %d %d"%(cfg.HARDWAREID,cfg.STEPS,cfg.STATESIZE,cfg.GRAPHSIZE,cfg.DISTANCE_VECTOR_SIZE,cfg.SAVE_STATES_MULTIPLE_OF_N,cfg.ANIMATION,cfg.STATESIZE/4))

def runCore_STAGGERED1D():
    if cfg.TEST_MODE:
        os.system("neblina -id %d HIPERWALK_TEMP_STAGGERED1D.nbl %d %d %d %d %d %d >> /dev/null "%(cfg.HARDWAREID,cfg.STEPS,cfg.STATESIZE,cfg.GRAPHSIZE,cfg.DISTANCE_VECTOR_SIZE,cfg.SAVE_STATES_MULTIPLE_OF_N,cfg.ANIMATION))
    else:
        os.system("neblina -id %d HIPERWALK_TEMP_STAGGERED1D.nbl %d %d %d %d %d %d"%(cfg.HARDWAREID,cfg.STEPS,cfg.STATESIZE,cfg.GRAPHSIZE,cfg.DISTANCE_VECTOR_SIZE,cfg.SAVE_STATES_MULTIPLE_OF_N,cfg.ANIMATION))

def runCore_STAGGERED2D():
    if cfg.TEST_MODE:
        os.system("neblina -id %d HIPERWALK_TEMP_STAGGERED2D.nbl %d %d %d %d %d %d >> /dev/null "%(cfg.HARDWAREID,cfg.STEPS,cfg.STATESIZE,cfg.GRAPHSIZE,cfg.DISTANCE_VECTOR_SIZE,cfg.SAVE_STATES_MULTIPLE_OF_N,cfg.ANIMATION))
    else:
        os.system("neblina -id %d HIPERWALK_TEMP_STAGGERED2D.nbl %d %d %d %d %d %d"%(cfg.HARDWAREID,cfg.STEPS,cfg.STATESIZE,cfg.GRAPHSIZE,cfg.DISTANCE_VECTOR_SIZE,cfg.SAVE_STATES_MULTIPLE_OF_N,cfg.ANIMATION))


def runCore_CUSTOM():   
    a=" ".join([str(x) for x in cfg.CUSTON_OPERATORS_NAME])
    b=" ".join([str(x) for x in cfg.CUSTOM_LABELS_NAME])
    computeStates = 0
    
    if cfg.ADJMATRIX_PATH != "@NON_INICIALIZED@":
         computeStates = 1    
    

    if cfg.TEST_MODE:
        os.system("neblina -id %d HIPERWALK_TEMP_CUSTOM.nbl %d %d %d %d %d %d %d %d %d %s %s %s >> /dev/null"%(int(cfg.HARDWAREID),int(cfg.STEPS),int(cfg.SAVE_STATES_MULTIPLE_OF_N),int(cfg.SAVE_PROBS_MULTIPLE_OF_N), computeStates, int(cfg.ANIMATION),int(cfg.STATESIZE),int(cfg.GRAPHSIZE),int(cfg.CUSTOM_UNITARY_COUNTER),int(cfg.GRAPHDIMENSION),cfg.CUSTOM_INITIALSTATE_NAME,a,b))
    else:
        os.system("neblina -id %d HIPERWALK_TEMP_CUSTOM.nbl %d %d %d %d %d %d %d %d %d %s %s %s"%(int(cfg.HARDWAREID),int(cfg.STEPS),int(cfg.SAVE_STATES_MULTIPLE_OF_N),int(cfg.SAVE_PROBS_MULTIPLE_OF_N), computeStates, int(cfg.ANIMATION),int(cfg.STATESIZE),int(cfg.GRAPHSIZE),int(cfg.CUSTOM_UNITARY_COUNTER),int(cfg.GRAPHDIMENSION),cfg.CUSTOM_INITIALSTATE_NAME,a,b))
    
    
    
def generating_DTQW1D_NBL():
    output = open("HIPERWALK_TEMP_WALK.nbl",'w')    
    
    output.write("-- core.nbl\n")
    output.write("using io\n")
    output.write("using math\n")
    output.write("using time\n")
    
    output.write("def stdvr(OUTPUT_FILE, STEPS,PSI, DISTANCES, DISTANCES_SQUARE,ANIMATION,GRAPHSIZE)\n")
    output.write("    t0 = micro()\n")
    
    output.write("    p = vec_conj( PSI )\n")
#    output.write("    <<< p\n")
    output.write("    PROBABILITY_VECTOR  = vec_add_off( p,len(p)/2  )    \n")
#    output.write("   >>> PROBABILITY_VECTOR \n")
    output.write("    if ANIMATION == 1    \n")
    output.write("        f = open( \"NEBLINA_TEMP_PROB\" + tostr(STEPS, 5) + \".dat\", \"w\" )\n")
#    output.write("	 <<< PROBABILITY_VECTOR\n")
    output.write("        print( f, PROBABILITY_VECTOR )\n")
#    output.write("        >>> PROBABILITY_VECTOR\n")
    output.write("    end\n")
    output.write("\n")
    output.write("    tmp = vec_prod( DISTANCES_SQUARE, PROBABILITY_VECTOR )\n")
    output.write("\n")
    output.write("    s1  = vec_sum( tmp )    \n")
    output.write("    tmp = vec_prod( DISTANCES, PROBABILITY_VECTOR )\n")
    output.write("    mean = vec_sum( tmp )\n")
    output.write("    stddev = sqrt(abs(s1-mean*mean))\n")
    output.write("    println( OUTPUT_FILE, STEPS  + \"               \" + mean + \"               \" + s1 + \"               \" + stddev )\n")
    output.write("    t1 = micro()\n")
    output.write("--    println( t1 - t0 )\n")
    output.write("end\n")
    
    output.write("def main()\n")
#    
    output.write("fmtdouble( 16 )\n")
    output.write("   STATESIZE = %d\n"%cfg.STATESIZE)
#    
    output.write("   GRAPHSIZE= %d\n"%cfg.GRAPHSIZE)
#    
    output.write("   DISTANCE_VECTOR_SIZE = %d\n"%cfg.DISTANCE_VECTOR_SIZE)
#    
    output.write("   STEPS = %d\n"%cfg.STEPS)
#    
    output.write("   ANIMATION = %d\n"%cfg.ANIMATION)
#    
    output.write("   SAVE_STATES_MULTIPLE_OF_N=%d\n"%cfg.SAVE_STATES_MULTIPLE_OF_N)
#    
    output.write("   file_CtI = open( \"HIPERWALK_TEMP_COIN_TENSOR_IDENTITY_1D.dat\", \"r\" )\n")
    output.write("   CtI = sparse complex[STATESIZE,STATESIZE]\n")
    output.write("   read( file_CtI, CtI )\n")
#    
    output.write("   file_S = open( \"HIPERWALK_TEMP_SHIFT_OPERATOR_1D.dat\", \"r\" )\n")
    output.write("   S = sparse complex[STATESIZE,STATESIZE]\n")
    output.write("   read( file_S, S )\n")
#    
    output.write("   file_psi = open( \"HIPERWALK_TEMP_PSI.dat\", \"r\" )\n")
    output.write("   PSI = complex[STATESIZE]\n")
    output.write("   read( file_psi, PSI )\n")
#    
    output.write("   file_D = open(\"HIPERWALK_TEMP_DISTANCE_VECTOR.dat\", \"r\" ) \n")
    output.write("   DISTANCES_VECTOR = float[DISTANCE_VECTOR_SIZE ]\n")
    output.write("   read( file_D, DISTANCES_VECTOR )\n")
#    
#    output.write("   >>> DISTANCES_VECTOR  \n")
    output.write("   DISTANCES_SQUARE = vec_prod( DISTANCES_VECTOR,DISTANCES_VECTOR)\n")
#    output.write("   >>> CtI\n")
#    output.write("   >>> S\n")
#    output.write("   >>> PSI\n")
#
    output.write("   stdv_file = open(\"statistics.dat\", \"w\" ) \n")
    output.write("   println( stdv_file, \"#STEP               Mean                                      2nd Moment                                  Standard deviation\" )\n")
    
    output.write("   stdvr(stdv_file, 0, PSI,  DISTANCES_VECTOR  , DISTANCES_SQUARE, ANIMATION,GRAPHSIZE)\n")
#    
#
    output.write("   for t = 1 : STEPS\n")
    output.write("           PSI = mat_mulvec( CtI, PSI )\n")
    output.write("           PSI = mat_mulvec( S, PSI )\n")
#
#
    if cfg.ALLSTATES:
#        output.write("           <<< PSI\n")
        output.write("           if SAVE_STATES_MULTIPLE_OF_N > 0 \n")
        output.write("               a=t % SAVE_STATES_MULTIPLE_OF_N \n")
        output.write("               if a == 0 \n")
        output.write("                   f = open( \"wavefunction-\" + tostr(t) + \".dat\", \"w\" )\n")
        output.write("                   print( f, PSI )\n")
        output.write("                   stdvr(stdv_file, t, PSI,  DISTANCES_VECTOR  , DISTANCES_SQUARE, ANIMATION,GRAPHSIZE)\n")
        output.write("               end\n")
        output.write("           end\n")
#        output.write("           if SAVE_STATES_MULTIPLE_OF_N == 0 \n")
#        output.write("               f = open( \"wavefunction-\" + tostr(t) + \".dat\", \"w\" )\n")        
#        output.write("               print( f, PSI )\n")
#
#        output.write("           end\n")
#        output.write("           >>> PSI\n")
    else:
        output.write("           stdvr(stdv_file, t, PSI,  DISTANCES_VECTOR  , DISTANCES_SQUARE, ANIMATION,GRAPHSIZE)\n")

    output.write("   end  \n")
#    
    output.write("  PROBABILITY_VECTOR = vec_conj( PSI )\n")
    output.write("   <<< PSI, PROBABILITY_VECTOR \n")
    
    
    output.write("   final_prob = vec_add_off( PROBABILITY_VECTOR , len(PROBABILITY_VECTOR)/2 )\n")
    output.write("   println( \"[Neblina] Statistics file: .............. statistics.dat\" )\n")
    output.write("   final_state_neblina = open(\"NEBLINA_TEMP_final_state.dat\", \"w\")\n")
    output.write("   println( final_state_neblina, PSI )\n")
    output.write("   final_distribution_neblina = open(\"NEBLINA_TEMP_final_distribution.dat\", \"w\")\n")
    output.write("   println( final_distribution_neblina, final_prob )\n")
    
    output.write("   println( \"[Neblina] Wave Function final : .............. final_state.dat\" )\n")
    output.write("   println(\"[Neblina] Done!\")\n")
    output.write("end\n")
    output.close()

    

    




def generating_DTQW2D_NBL():

    output = open("HIPERWALK_TEMP_WALK.nbl",'w')    
    output.write(" -- core.nbl \n")
    output.write(" using io \n")
    output.write(" using std \n")
    output.write(" using math \n")
    output.write(" using time \n")

    output.write(" def stdv( OUTPUT_FILE, psi, x, x2, y, y2, ANIMATION, t ) \n")
    output.write("  \n")
    output.write("     prob = vec_conj( psi ) \n")
    output.write("     prob = vec_add_off( prob, len(prob)/4 ) \n")
    output.write("      \n")
    output.write("     if ANIMATION == 1 \n")
    output.write("         f = open( \"NEBLINA_TEMP_PROB\" + tostr(t, 5) + \".dat\", \"w\" ) \n")
    output.write(" 	    print( f, prob )     \n")
    output.write("     end \n")
    output.write("  \n")
    output.write("     vX = vec_prod( x, prob ) \n")
    output.write("     vY = vec_prod( y, prob ) \n")
    output.write("      \n")
    output.write("     vXX = vec_prod( x2, prob ) \n")
    output.write("     vYY = vec_prod( y2, prob ) \n")
    output.write("      \n")
    output.write("     sX = vec_sum( vX ) \n")
    output.write("     sY = vec_sum( vY ) \n")
    output.write("      \n")
    output.write("     sXX = vec_sum( vXX ) \n")
    output.write("     sYY = vec_sum( vYY ) \n")
    output.write("      \n")
    output.write("     varX = sXX - sX*sX \n")
    output.write("     varY = sYY - sY*sY \n")
    output.write("      \n")
    output.write("     println( OUTPUT_FILE, t  + \"            \" + sX + \"               \" + sY + \"                \" + varX + \"                \" + varY + \"              \" + sqrt( varX + varY ) ) \n")
#    output.write("     println( OUTPUT_FILE, t   \"         \"  sX  \"          \"  sY  \"          \"  varX  \"          \"  varY  \"          \"  sqrt( varX  varY ) ) \n")
    output.write("      \n")
    output.write(" end \n")
    output.write("  \n")

    output.write(" def main() \n")
    output.write("    fmtdouble( 16 ) \n")
    output.write("   STATESIZE = %d\n"%cfg.STATESIZE)
    output.write("   GRAPHSIZE= %d\n"%cfg.GRAPHSIZE)   
    output.write("   DISTANCE_VECTOR_SIZE = %d\n"%cfg.DISTANCE_VECTOR_SIZE)
    output.write("   STEPS = %d\n"%cfg.STEPS)
    output.write("   ANIMATION = %d\n"%cfg.ANIMATION)
    output.write("   SAVE_STATES_MULTIPLE_OF_N=%d\n"%cfg.SAVE_STATES_MULTIPLE_OF_N)
    output.write("    file_CtI = open( \"HIPERWALK_TEMP_COIN_TENSOR_IDENTITY_OPERATOR_2D.dat\", \"r\" ) \n")
    output.write("    CtI = sparse complex[STATESIZE,STATESIZE] \n")
    output.write("    read( file_CtI, CtI ) \n")
    output.write("    file_S = open( \"HIPERWALK_TEMP_SHIFT_OPERATOR_2D.dat\", \"r\" ) \n")
    output.write("    S = sparse complex[STATESIZE,STATESIZE] \n")
    output.write("    read( file_S, S ) \n")
    output.write("    file_psi = open( \"HIPERWALK_TEMP_PSI.dat\", \"r\" ) \n")
    output.write("    PSI = complex[STATESIZE] \n")
    output.write("    read( file_psi, PSI ) \n")
    output.write("     \n")
    output.write("    file_X = open(\"HIPERWALK_TEMP_X.dat\", \"r\" )  \n")
    output.write("    file_Y = open(\"HIPERWALK_TEMP_Y.dat\", \"r\" ) \n")
    output.write("    X = float[DISTANCE_VECTOR_SIZE] \n")
    output.write("    Y = float[DISTANCE_VECTOR_SIZE] \n")
    output.write("    read( file_X, X ) \n")
    output.write("    read( file_Y, Y ) \n")
    output.write("     \n")
    output.write("    X2 = vec_prod( X, X ) \n")
    output.write("    Y2 = vec_prod( Y, Y ) \n")
    output.write("     \n")
    output.write("    stdv_file = open(\"statistics.dat\", \"w\" )  \n")
    output.write("    println( stdv_file, \"#STEP     Mean(x)                                         Mean(y)                                                Var(x)                                                Var(y)                                                Standard deviation\" ) \n")
    output.write("    stdv( stdv_file, PSI, X, X2, Y, Y2, ANIMATION, 0 ) \n")
    output.write("    for t = 1 : STEPS \n")
    output.write("            PSI = mat_mulvec( CtI, PSI ) \n")
    output.write("            PSI = mat_mulvec( S, PSI ) \n")
    if cfg.ALLSTATES:
        output.write("            if SAVE_STATES_MULTIPLE_OF_N > 0  \n")
        output.write("                a=t % SAVE_STATES_MULTIPLE_OF_N  \n")
        output.write("                if a == 0  \n")
        output.write("                    f = open( \"wavefunction-\" + tostr(t) + \".dat\", \"w\" ) \n")
#        output.write("                    f = open( \"wavefunction-\"  tostr(t)  \".dat\", \"w\" ) \n")
        output.write("                    print( f, PSI ) \n")
        output.write("                stdv( stdv_file, PSI, X, X2, Y, Y2, ANIMATION, t ) \n")
        output.write("                end \n")

        output.write("            end \n")

        

    else:
        output.write("            stdv( stdv_file, PSI, X, X2, Y, Y2, ANIMATION, t ) \n")
    output.write("    end   \n")
    output.write("    PROBABILITY_VECTOR = vec_conj( PSI ) \n")
    output.write("    final_prob = vec_add_off( PROBABILITY_VECTOR , len(PROBABILITY_VECTOR)/4 ) \n")
    output.write("    println( \"[Neblina] Statistics file: .............. statistics.dat\" ) \n")
    output.write("    final_state_neblina = open(\"NEBLINA_TEMP_final_state.dat\", \"w\") \n")
    output.write("    println( final_state_neblina, PSI ) \n")
    output.write("    final_distribution_neblina = open(\"NEBLINA_TEMP_final_distribution.dat\", \"w\") \n")
    output.write("    println( final_distribution_neblina, final_prob ) \n")
    output.write("    println( \"[Neblina] Wave Function final : .............. final_state.dat\" ) \n")
    output.write("    println(\"[Neblina] Done!\") \n")
    output.write(" end \n")
    output.close()
    



   


    
def generating_STAGGERED1D_NBL():
    
    output = open("HIPERWALK_TEMP_WALK.nbl",'w')    
    
    ##
    ## Defining statistics
    ##    
    output.write("-- core.nbl\n")
    output.write("using io\n")
    output.write("using math\n")
    output.write("using time\n")
    output.write("def stdvr(OUTPUT_FILE, STEPS,PSI, DISTANCES, DISTANCES_SQUARE,ANIMATION,GRAPHSIZE)\n")
    output.write("    t0 = micro()\n")
    output.write("    PROBABILITY_VECTOR = vec_conj( PSI )\n")
    output.write("    if ANIMATION == 1    \n")
    output.write("        f = open( \"NEBLINA_TEMP_PROB\" + tostr(STEPS, 5) + \".dat\", \"w\" )\n")
    output.write("        print( f, PROBABILITY_VECTOR )\n")
    output.write("    end\n")
    
    output.write("    tmp = vec_prod( DISTANCES_SQUARE, PROBABILITY_VECTOR )\n")
    
    output.write("    s1  = vec_sum( tmp )    \n")
    output.write("    tmp = vec_prod( DISTANCES, PROBABILITY_VECTOR )\n")
    output.write("    mean = vec_sum( tmp )\n")
    output.write("    stddev = sqrt(abs(s1-mean*mean))\n")
    output.write("    println( OUTPUT_FILE, STEPS  + \"         \" + mean + \"          \" + s1 + \"          \" + stddev )\n")
    output.write("    t1 = micro()\n")
    output.write("--    println( t1 - t0 )\n")
    output.write("end\n")
    
    ##
    ## Defining main
    ##
    output.write("def main()\n")
    output.write("fmtdouble( 16 )\n")
    # Reading variables
    


    output.write("   STATESIZE = %d\n"%cfg.STATESIZE)
#    
    output.write("   STEPS = %d\n"%cfg.STEPS)
#    
    output.write("   ANIMATION = %d\n"%cfg.ANIMATION)

    output.write("   SAVE_STATES_MULTIPLE_OF_N=%d\n"%cfg.SAVE_STATES_MULTIPLE_OF_N)

    output.write("   file_Ue = open( \"HIPERWALK_TEMP_STAGGERED_EVEN_OPERATOR_1D.dat\", \"r\" )\n")
    output.write("   Ue = sparse complex[STATESIZE,STATESIZE]\n")
    output.write("   read( file_Ue, Ue )\n")
    output.write("   file_Uo = open( \"HIPERWALK_TEMP_STAGGERED_ODD_OPERATOR_1D.dat\", \"r\" )\n")
    output.write("   Uo = sparse complex[STATESIZE,STATESIZE]\n")
    output.write("   read( file_Uo, Uo )\n")
    output.write("   file_psi = open( \"HIPERWALK_TEMP_PSI.dat\", \"r\" )\n")
    output.write("   PSI = complex[STATESIZE]\n")
    output.write("   read( file_psi, PSI )\n")
    output.write("   file_D = open(\"HIPERWALK_TEMP_RANGE_1D.dat\", \"r\" ) \n")
    output.write("   DISTANCES_VECTOR = float[STATESIZE]\n")
    output.write("   read( file_D, DISTANCES_VECTOR )\n")

    # Sending variables to device
    output.write("   >>> DISTANCES_VECTOR  \n")
    output.write("   DISTANCES_SQUARE = vec_prod( DISTANCES_VECTOR,DISTANCES_VECTOR)\n")
    output.write("   stdv_file = open(\"statistics.dat\", \"w\" ) \n")
    output.write("   println( stdv_file, \"#STEP     Mean          2nd Moment    Standard deviation\" )\n")
    output.write("   stdvr(stdv_file, 0, PSI,  DISTANCES_VECTOR  , DISTANCES_SQUARE, ANIMATION,STATESIZE)\n")
    # Markov chain
    output.write("   for t = 1 : STEPS\n")
    output.write("           PSI = mat_mulvec( Ue, PSI )\n")
    output.write("           PSI = mat_mulvec( Uo, PSI )\n")

    if cfg.ALLSTATES:
        output.write("           if SAVE_STATES_MULTIPLE_OF_N > 0 \n")        
        output.write("               a=t % SAVE_STATES_MULTIPLE_OF_N \n")
        output.write("               if a == 0 \n")
        output.write("                   f = open( \"wavefunction-\" + tostr(t) + \".dat\", \"w\" )\n")
        output.write("                   print( f, PSI )\n")
        output.write("               end\n")
        output.write("           end\n")
        output.write("           if SAVE_STATES_MULTIPLE_OF_N == 0 \n")
        output.write("               f = open( \"wavefunction-\" + tostr(t) + \".dat\", \"w\" )\n")        
        output.write("               print( f, PSI )\n")
        output.write("           end\n")
    
    output.write("           stdvr(stdv_file, t, PSI,  DISTANCES_VECTOR  , DISTANCES_SQUARE, ANIMATION,STATESIZE)\n")
    output.write("   end  \n")
    
    output.write("   PROBABILITY_VECTOR = vec_conj( PSI )\n")
#    output.write("   final_prob = vec_add_off( PROBABILITY_VECTOR , len(PROBABILITY_VECTOR)/2 )\n")
#    output.write("   final_prob = vec_conj( PSI )\n")    
    output.write("   println( \"[Neblina] Statistics file: .............. statistics.dat\" )\n")
    output.write("   final_state_neblina = open(\"NEBLINA_final_state.dat\", \"w\")\n")
    output.write("   println( final_state_neblina, PSI )\n")
    output.write("   final_distribution_neblina = open(\"NEBLINA_final_distribution.dat\", \"w\")\n")
    output.write("   println( final_distribution_neblina, PROBABILITY_VECTOR )\n")
    output.write("   println( \"[Neblina] Wave Function final : .............. final_state.dat\" )\n")
    output.write("   println(\"[Neblina] Done!\")\n ")
    output.write("end")
    
    output.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

def neblina_state_to_vector(fileIn):
    qtdeValues = 0  
    for line in open(fileIn):  
        if line == "\n":  
            continue  
        if line.startswith("#"):  
            continue  
        qtdeValues = qtdeValues + 1  

    vector=np.zeros((qtdeValues,1),dtype=complex)
    actual = 0  
    
    for line in open(fileIn):  
        if line == "\n":  
            continue  
        if line.startswith("#"):  
            continue          
        line = line.split()  
        a=float(line[0])
        b=1J*float(line[1])
        vector[actual] = a+b
        actual = actual + 1  
    return vector


def neblina_distribution_to_vector(fileIn):
    qtdeValues = 0  
    for line in open(fileIn):  
        if line == "\n":  
            continue  
        if line.startswith("#"):  
            continue  
        qtdeValues = qtdeValues + 1  

    vector=np.zeros((qtdeValues,1),dtype=float)
    actual = 0  
    
    for line in open(fileIn):  
        if line == "\n":  
            continue  
        if line.startswith("#"):  
            continue          
        line = line.split()  
        a=float(line[0])
        vector[actual] = a
        actual = actual + 1  
    return vector






def generating_CUSTOM_NBL():
    output = open("walk.nbl",'w')    
    
    output.write("-- core.nbl\n")
    output.write("using io\n")
    output.write("using math\n")
    output.write("using time\n")
    
    output.write("def main()\n")
 
    output.write("   fmtdouble( 16 )\n")

    output.write("   STATESIZE = %d\n"%cfg.STATESIZE)
    
    output.write("   STEPS = %d\n"%cfg.STEPS)
    
    output.write("   SAVE_STATES_MULTIPLE_OF_N=%d\n"%cfg.SAVE_STATES_MULTIPLE_OF_N)

    counter=0
    for i in cfg.CUSTON_OPERATORS_NAME:
        output.write("   file_U%d = open( \"%s\", \"r\" )\n"%(counter,i))
        output.write("   U%d = sparse complex[STATESIZE,STATESIZE]\n"%counter)
        output.write("   read( file_U%d, U%d )\n"%(counter,counter))
        counter=counter+1
    
    output.write("   file_psi = open( \"%s\", \"r\" )\n"%cfg.CUSTOM_INITIALSTATE_NAME)
    output.write("   PSI = complex[STATESIZE]\n")
    output.write("   read( file_psi, PSI )\n")


#    for i in range(counter):
#        output.write("   >>> U%d\n"%i)    
#    output.write("   >>> PSI\n")   
    
    output.write("   for t = 1 : STEPS\n")

    for i in range(counter):
            output.write("           PSI = mat_mulvec( U%d, PSI )\n"%i)

    if cfg.ALLSTATES:
#        output.write("           <<< PSI\n")
        output.write("           if SAVE_STATES_MULTIPLE_OF_N > 0 \n")        
        output.write("               a=t % SAVE_STATES_MULTIPLE_OF_N \n")
        output.write("               if a == 0 \n")
        output.write("                   f = open( \"wavefunction-\" + tostr(t) + \".dat\", \"w\" )\n")
        output.write("                   print( f, PSI )\n")
        output.write("               end\n")
        output.write("           end\n")


#        output.write("           >>> PSI\n")
        
    output.write("   end  \n")
#    output.write("   <<< PSI\n")
    output.write("   final = open(\"final_state.dat\", \"w\")\n")
    output.write("   println( final, PSI)\n")
    output.write("   println(\"[Neblina] Done!\")\n")
    output.write("end\n")
    output.close()

