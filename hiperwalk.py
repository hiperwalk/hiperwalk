#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 14:29:17 2015

@author: Aaron Leao
         aaron@lncc.br
"""
import sys
import os
import walks
import testmode

tam = len(sys.argv)

if tam>=2:
    string=(str(sys.argv[1]))
    if string =="-test":
        testmode.run()
    
    elif not os.path.isfile( string ):
            print("[Hiperwalk] File '%s' not found."%(string))
            exit(-1)
    else:
        returnValue=walks.walk(string)
        
else:
    print("[Hiperwalk] Missing input file.")
    exit(0)
