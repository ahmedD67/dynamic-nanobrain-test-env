#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 13:20:21 2022

Many of the methods have been adapted from Stone et al. 2017

@author: dwinge
"""
import numpy as np
import pandas as pd
import os

DATA_PATH='data'

def generate_filename(T_outbound, T_inbound, N, **kwargs):
    filename = 'out{0}_in{1}_N{2}'.format(str(T_outbound),
                                                str(T_inbound),
                                                str(N))
    for k, v in kwargs.iteritems():
        filename += '_' + k + str(v)
    return filename + '.npz'

def generate_figurename(description,T_outbound, T_inbound, N, **kwargs):
    filename = '{3}_out{0}_in{1}_N{2}'.format(str(T_outbound),
                                                str(T_inbound),
                                                str(N),
                                                description)
    for k, v in kwargs.iteritems():
        filename += '_' + k + str(v)
    return filename + '.png'

def load_dataset(T_outbound, T_inbound, N, **kwargs):
    filename = generate_filename(T_outbound, T_inbound, N,
                                 **kwargs)
    OUT, INB = pd.read_pickle(os.path.join(DATA_PATH, filename))
    
    return OUT, INB


def save_dataset(OUT, INB, T_outbound, T_inbound, N,
                 **kwargs):
    filename = generate_filename(T_outbound, T_inbound, N,
                                 **kwargs)
        
    (OUT, INB).to_pickle(os.path.join(DATA_PATH, filename))
    

def generate_dataset(T_outbound=1500, T_inbound=1500, N=1000,
                     save=True, **kwargs):
    try:
        OUT, INB = load_dataset(T_outbound, T_inbound, N,
                                           **kwargs)
    except:
        # Old code
        #T = T_outbound + T_inbound
        #H = np.empty([N, T+1])
        #V = np.empty([N, T+1, 2])  # TODO(tomish) why is this shape larger?
        
        l_OUT = [] # list of DataFrames
        l_INB = [] 
                          
        for i in range(N):
            out_res, inb_res, out_travel, inb_travel = run_trial(
                    T_outbound=T_outbound,
                    T_inbound=T_inbound,
                    **kwargs)
            # Generate plots here
            # Save the routes to the lists
            l_OUT.append(out_travel)
            l_INB.append(inb_travel)
        
        # Combine the DataFrame to a big one
        key_sequence = np.arange(0,N)
        OUT = pd.concat(l_OUT,keys=key_sequence)
        INB = pd.concat(l_INB,keys=key_sequence)
        
        if save:
            save_dataset(OUT, INB, T_outbound, T_inbound, N,
                         **kwargs)
    return OUT, INB