#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 14:57:06 2021

@author: dwinge
"""
import numpy as np

def example_gammas() :
    # System parameters
    Cinh = 200 # nf/cm^2
    Rinh = 1e-3 # Ohm cm^2
    Cexc = Cinh
    Rexc = Rinh
    Cstore = 100 # nf/cm^2
    Rstore = 1e-2 # Ohm cm^2
    RCLED = 0.2 # ns
    
    # System frequencies
    g11 = 1./Cinh/Rinh # ns^-1 # GHz
    g22 = 1./Cexc/Rexc # 
    g13 = 1./Cstore/Rinh
    g23 = 1./Cstore/Rexc
    g33 = 1./Cstore/Rstore
    gled = 1./RCLED

    return np.array([g11,g22,g13,g23,g33,gled])

def square_pulse(t, tlims):
    tmp = 0.
    for ttuple in tlims:
        tmp += float((t >= ttuple[0])*(t < ttuple[1]))
    return tmp

def transistorIV (Vgate) :
    return Vgate