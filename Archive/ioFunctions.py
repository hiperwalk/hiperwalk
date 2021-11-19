# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 05:27:04 2014

@author: aaron
"""


import numpy as np
import config as cfg
import os


def savetxt(string,array,arrayType,formatFloat):
    if arrayType == int:
        np.savetxt(str(string),array.view(int),fmt='%d')
    else:
        np.savetxt(str(string),array.view(float),fmt=formatFloat)
        

def initializeFiles():
    if cfg.WALK=="DTQW1D" or cfg.WALK=="STAGGERED1D":
        range1D()
        
    elif cfg.WALK=="DTQW2D" or cfg.WALK=="STAGGERED2D":
        range2D()
        distancesVector_2D()
        
def range1D():
    output = open("HIPERWALK_TEMP_RANGE_1D.dat",'w')    
       
    for x in  np.arange(cfg.RANGEX[0],cfg.RANGEX[1]+1,1):
        output.write("%d\n"%(x))
    output.close()


def range2D(): 
    output = open("HIPERWALK_TEMP_RANGE_2D.dat",'w')    
       
    for x in  np.arange(cfg.RANGEX[0],cfg.RANGEX[1]+1,1):
        for y in np.arange(cfg.RANGEY[0],cfg.RANGEY[1]+1,1):
            output.write("%d\t%d\t\n"%(x,y))
    output.close()
    
def distancesVector_2D():
    X = open("HIPERWALK_TEMP_X.dat", "w" ) 
    Y = open("HIPERWALK_TEMP_Y.dat", "w" ) 
  
    
    for x in range( cfg.RANGEX[0],cfg.RANGEX[1]+1 ):
        for y in range( cfg.RANGEY[0],cfg.RANGEY[1]+1 ):
            X.write("%d\n"%(x))
            Y.write("%d\n"%(y))
    X.close()
    Y.close()
    
def directory_Creation():
    cfg.OLD_DIRECTORY=os.getcwd() 
#    os.system("rm -rf *.pyc")
    if cfg.DIRECTORY!='' and cfg.DIRECTORY:
        
        if not os.path.exists(cfg.DIRECTORY):
            try:
                os.mkdir("%s"%cfg.DIRECTORY)
            except OSError: # this would be "except OSError, e:" before Python 2.6
                print("[HIPERWALK] Could not creat folder %s."%cfg.DIRECTORY)
                exit(-2)

#            os.mkdir("%s"%cfg.DIRECTORY)

        if cfg.WALK=="DTQW1D":
            os.system("cp %s/hiperwalk/dtqw1d.nbl %s/HIPERWALK_TEMP_DTQW1D.nbl"%(cfg.INSTALL_DIR, cfg.DIRECTORY))
        elif cfg.WALK=="DTQW2D":
            os.system("cp %s/hiperwalk/dtqw2d.nbl %s/HIPERWALK_TEMP_DTQW2D.nbl"%(cfg.INSTALL_DIR, cfg.DIRECTORY))
        elif cfg.WALK=="STAGGERED1D":
            os.system("cp %s/hiperwalk/staggered1d.nbl %s/HIPERWALK_TEMP_STAGGERED1D.nbl"%(cfg.INSTALL_DIR, cfg.DIRECTORY))
        elif cfg.WALK=="STAGGERED2D":
            os.system("cp %s/hiperwalk/staggered2d.nbl %s/HIPERWALK_TEMP_STAGGERED2D.nbl"%(cfg.INSTALL_DIR, cfg.DIRECTORY))
        elif cfg.WALK=="CUSTOM":
            os.system("cp %s/hiperwalk/custom.nbl %s/HIPERWALK_TEMP_CUSTOM.nbl"%(cfg.INSTALL_DIR, cfg.DIRECTORY))
        os.chdir("%s"%cfg.DIRECTORY)
        
        
    else:
        print("[HIPERWALK] Error at DIRECTORY BLOCK, missing value!")
        exit(-1)
        
def cleaningTemporaryFiles():
    os.system("rm -rf HIPERWALK_TEMP*")
    os.system("rm -rf NEBLINA_TEMP*")

    if not cfg.TEST_MODE:
        os.chdir(cfg.OLD_DIRECTORY)

def remove(filename):
    try:
        os.remove(filename)
    except OSError: # this would be "except OSError, e:" before Python 2.6
        pass

    
def test_mode():
#    a=str(os.getcwd())
#    print("Teste %s"%a)

    os.chdir(os.environ['HOME'])
    if not os.path.exists("HIPERWALK_TEST_DIRECTORY"):
        os.mkdir("HIPERWALK_TEST_DIRECTORY")
#    os.system("cp *.in HIPERWALK_TEMP_DIRECTORY")
#    os.system("cp psi0.dat HIPERWALK_TEMP_DIRECTORY")
#    os.system("cp u0.dat HIPERWALK_TEMP_DIRECTORY")
#    os.system("cp u1.dat HIPERWALK_TEMP_DIRECTORY")
    os.chdir("HIPERWALK_TEST_DIRECTORY")


def remnove_test_mode_folder():
    os.chdir(os.environ['HOME'])
    os.system("rm -rf HIPERWALK_TEST_DIRECTORY")
