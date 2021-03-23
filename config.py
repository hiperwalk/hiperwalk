# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 00:47:37 2014

@author: aaron
Module containing the global variables.
"""
import numpy as np
import config as cfg


def reset():
    ##
    ##  General Variables
    ##
    cfg.WALK = ""
    cfg.DIRECTORY = "HIPERWALK_DEFAULT_FOLDER"
    cfg.STEPS = 0
    cfg.GRAPHSIZE = 0
    cfg.STATE_COMPONENTS = []
    cfg.STATE = []
    cfg.STATESIZE = 0
    cfg.SIMULATIONTYPE = "PARALLEL"
    cfg.GRAPHTYPE = ""
    cfg.SIZEX = 0
    cfg.SIZEY = 0
    cfg.RANGEX = []
    cfg.RANGEY = []
    cfg.COINOPERATORNAME = ""
    cfg.ALLSTATES = 0
    cfg.SAVE_STATES_MULTIPLE_OF_N = 0
    cfg.ALLPROBS = 0
    cfg.SAVE_PROBS_MULTIPLE_OF_N = 0

    ##
    ##  Statistics
    ##
    cfg.DISTANCES_VECTOR = []
    cfg.DISTANCES_SQUARE = []
    cfg.DISTANCES_VECTOR_X = []
    cfg.DISTANCES_VECTOR_Y = []
    cfg.DISTANCES_VECTOR_SQUARE_X = []
    cfg.DISTANCES_VECTOR_SQUARE_Y = []

    ##
    ##  Coined Walk
    ##
    cfg.COINVECTOR = []
    cfg.COINOPERATOR = np.array([], dtype=complex)
    cfg.SHIFTOPERATOR = []
    cfg.COINTENSORIDENTITY = []
    cfg.COINVECTORDIMENSION = 0

    ###
    ### DTQW2D
    ###
    cfg.LATTYPE = "DIAGONAL"
    cfg.TORUSSIZE = np.array([], dtype=int)

    ###  STAGGERED Walk
    ###
    cfg.Ueven = []
    cfg.Uodd = []
    cfg.STAGGERED_COEFICIENTS = []
    cfg.TESSELLATIONPOLYGONS = []
    cfg.TOTAL_PATCHES_IN_X = 0
    cfg.TOTAL_PATCHES_IN_Y = 0
    cfg.OVERLAP = 0
    cfg.OVERLAPX = 0
    cfg.OVERLAPY = 0
    cfg.NUMBER_OF_COEFICIENTS = 0
    ###
    ### Custom Walk
    ###
    cfg.CUSTON_OPERATORS_NAME = []
    cfg.CUSTOM_INITIALSTATE_NAME = []
    cfg.CUSTOM_UNITARY_COUNTER = 0
    cfg.CUSTOM_LABELS_NAME = []
    cfg.CUSTOM_LABELS_NAME_COUNTER = 0
    cfg.GRAPHDIMENSION = 1
    cfg.ADJMATRIX_PATH = "@NON_INICIALIZED@"

    ###
    ###   Neblina
    ###
    cfg.HARDWAREID = 0

    ###
    ###   Gnuplot
    ###
    cfg.GNUPLOT = 0
    cfg.ANIMATION = 0
    cfg.PLOTTING_ZEROS = 0
    cfg.DELAY = 20

    ###
    ###   Debug
    ###
    cfg.DEBUG = 0
    cfg.QWALK_RETURN = []
    cfg.RETURN_SIMULATION_FLAG = 1
    #    cfg.TEST_MODE=0
    cfg.OLD_DIRECTORY = ""

    cfg.NUMBER_OF_TESSELLATIONS = 0
    cfg.TESSELLATIONGEOMETRY = []
    cfg.TESSELLATIONDISPLACEMENT = []
    cfg.TESSELLATIONINITIALDISPLACEMENT = []
    cfg.TESSELLATIONCOEFICIENTS = []

    cfg.VARIANCE = 0


##
##  General Variables
##
INSTALL_DIR = "/usr/local"
WALK = ""
DIRECTORY = "HIPERWALK_DEFAULT_FOLDER"
STEPS = 0
GRAPHSIZE = 0
STATE_COMPONENTS = []
STATE = []
STATESIZE = 0
SIMULATIONTYPE = "PARALLEL"
GRAPHTYPE = ""
SIZEX = 0
SIZEY = 0
RANGEX = []
RANGEY = []
COINOPERATORNAME = ""
ALLSTATES = 0
SAVE_STATES_MULTIPLE_OF_N = 0
ALLPROBS = 0
SAVE_PROBS_MULTIPLE_OF_N = 0

##
##  Statistics
##
DISTANCES_VECTOR = []
DISTANCES_SQUARE = []
DISTANCES_VECTOR_X = []
DISTANCES_VECTOR_Y = []
DISTANCES_VECTOR_SQUARE_X = []
DISTANCES_VECTOR_SQUARE_Y = []

##
##  Coined Walk
##
COINVECTOR = []
COINOPERATOR = np.array([], dtype=complex)
SHIFTOPERATOR = []
COINTENSORIDENTITY = []
COINVECTORDIMENSION = 0

###
### DTQW2D
###
LATTYPE = "DIAGONAL"
TORUSSIZE = np.array([], dtype=int)

###  Coinless Walk
###
Ueven = []
Uodd = []
STAGGERED_COEFICIENTS = []
TESSELLATIONPOLYGONS = []
TOTAL_PATCHES_IN_X = 0
TOTAL_PATCHES_IN_Y = 0
OVERLAP = 0
OVERLAPX = 0
OVERLAPY = 0
NUMBER_OF_COEFICIENTS = 0
###
### Custom Walk
###
CUSTON_OPERATORS_NAME = []
CUSTOM_INITIALSTATE_NAME = []
CUSTOM_UNITARY_COUNTER = 0
CUSTOM_LABELS_NAME = []
CUSTOM_LABELS_NAME_COUNTER = 0
GRAPHDIMENSION = 1
ADJMATRIX_PATH = "@NON_INICIALIZED@"
###
###   Neblina
###
HARDWAREID = 0

###
###   Gnuplot
###
GNUPLOT = 0
ANIMATION = 0
PLOTTING_ZEROS = 0
DELAY = 20

###
###   Debug
###
DEBUG = 0
QWALK_RETURN = []
RETURN_SIMULATION_FLAG = 1
TEST_MODE = 0
OLD_DIRECTORY = ""

###
###  Novo Coinless
###
NUMBER_OF_TESSELLATIONS = 0
TESSELLATIONGEOMETRY = []
TESSELLATIONDISPLACEMENT = []
TESSELLATIONINITIALDISPLACEMENT = []
TESSELLATIONCOEFICIENTS = []

VARIANCE = 0
