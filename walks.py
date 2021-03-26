# -*- coding: utf-8 -*-
#"""
#Created on Sun Jun 22 20:00:00 2014
#
#@author: aaron
#"""

import sys
import ioFunctions as io            ### Module for write on disk functions.
import config as cfg                ### Module for global variables for Quantum Walk.
import parsing as par               ### Module for parsing functions for the input file and setting variables
import time                         ### Library for the runtime.
import dtqw1d
import dtqw2d
import staggered1d
import staggered2d
import custom
import os
import distance as dist


### Checking existing file or TEST_MODE

def walk(inputFile):

    ### Acquiring the initial time
    t0=time.time()    
    
#    ###PreParsing
    par.preParsing(inputFile)
    
    ### Parsing the input file
    par.parsingFile(inputFile)

    ### Setting work path
    io.directory_Creation()
    
    ### Generating some files for Quantum Walk
    io.initializeFiles()
    
    if cfg.ADJMATRIX_PATH != "@NON_INICIALIZED@":
        dist.generateDistance()
        
        
    if cfg.WALK=="DTQW1D":
        cfg.RETURN_SIMULATION_FLAG=dtqw1d.run()
    
    elif cfg.WALK=="DTQW2D":
        cfg.RETURN_SIMULATION_FLAG=dtqw2d.run()
    
    elif cfg.WALK=="STAGGERED1D":
        cfg.RETURN_SIMULATION_FLAG=staggered1d.run()

    elif cfg.WALK=="STAGGERED2D":
        cfg.RETURN_SIMULATION_FLAG=staggered2d.run()
    
    elif cfg.WALK=="SZEGEDY":
        cfg.RETURN_SIMULATION_FLAG=szegedy.run()    
    
    elif cfg.WALK=="CUSTOM":
        cfg.RETURN_SIMULATION_FLAG=custom.run()
         
    ### Cleaning temporary files
    if not cfg.DEBUG:
        io.cleaningTemporaryFiles()

    
    os.chdir(cfg.OLD_DIRECTORY)
    
    
    
    ### End runtime
    t1=time.time()

    print("[HIPERWALK] Runtime: ",t1-t0)

    return cfg.RETURN_SIMULATION_FLAG
    
