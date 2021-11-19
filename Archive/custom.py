# -*- coding: utf-8 -*-
"""

Created on Mon Oct 27 04:06:41 2014

@author: aaron
"""
import neblina as nb
import testmode
import config as cfg
import numpy as np

def run():
    print("[Hiperwalk] CUSTOM WALK.")
    nb.runCore_CUSTOM()

    
    if cfg.TEST_MODE:
        modelVector=testmode.create_CUSTOM_test_vector()
        returnNeblina=nb.neblina_distribution_to_vector("final_distribution.dat")
        if np.linalg.norm(modelVector-returnNeblina,np.inf) == float(0):
            return 1
        else:
            return 0
            
            
    return 1