#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 11:25:20 2021

@author: dwinge
"""

import numpy as np

# %% Setup the analyze function that runs N trials for each of the args specified in 
# kwargs, effectively iteration through a list of arguments. 

# Note on relative imports, we have to load these things on the main package 
# level otherwise the relative imports won't work
import dynamicnanobrain.beesim.trialflight as trials 
import dynamicnanobrain.beesim.beeplotter as beeplotter

def analyse(N, param_dict):
    """Iterates through param dictionary, running batches of trials according to the param dictionary"""
    min_dists =[]
    min_dist_stds = []
    search_dists=[]
    search_dist_stds=[]
    
    for i in range(param_dict['n']):
        kwargs = {}
        for k, v in param_dict.items():
            if k != 'n' and v[i] != None:
                kwargs[k] = v[i]
                        
        OUT, INB = trials.generate_dataset(N=N, **kwargs)
        
        if 'T_outbound' in kwargs:
            T_outbound = kwargs['T_outbound']
        else:
            T_outbound = 1500
        if 'T_inbound' in kwargs:
            T_inbound = kwargs['T_inbound']
        else:
            T_inbound = 1500

        # Closest position to nest within return part. Use my new one 
        min_d, min_d_sigma, search_d, search_d_sigma= trials.analyze_inbound(INB, T_outbound,T_inbound)

        min_dists.append(min_d)
        min_dist_stds.append(min_d_sigma)
        search_dists.append(search_d)
        search_dist_stds.append(search_d_sigma)
    
    return min_dists, min_dist_stds, search_dists, search_dist_stds

#%% Here we specify the arguments of interest and generate the bulk of data
# needed for the analysis. In addition to saving the minimal distance to the 
# nest during the inbound flight, we also keep track of the size of the search
# pattern, perhaps this will prove interesting. 

N = 5 # number of trials for each parameter to test
N_dists = 3 # number of logarithmic distance steps
#distances = np.round(10 ** np.linspace(1, 4, N_dists)).astype('int')
distances = np.round(10 ** np.linspace(1, 3, N_dists)).astype('int')

# List the parameter values of interest
memupdate_vals = [0.001, 0.005, 0.0025, 0.0050, 0.01]
memupdate_vals = [0.0005,0.001]#, 0.005, 0.0025, 0.0050, 0.01]

# Specify the dict of parameters
#param_dicts = [{'n':N_dists, 'T_outbound': distances, 'T_inbound': distances}]
param_dicts = [{'n':N_dists, 'memupdate': [mem]*N_dists, 'T_outbound': distances, 'T_inbound': distances} for mem in memupdate_vals]
#param_dicts.append({'n':N_dists, 'T_outbound': distances, 'T_inbound': distances, 'random_homing':[True]*N_dists})

min_dists_l = []
min_dist_stds_l = []
search_dists_l=[]
search_dist_stds=[]
    
for param_dict in param_dicts:
    min_dists, min_dist_stds , search_dist, search_dist_std = analyse(N, param_dict)
    min_dists_l.append(min_dists)
    min_dist_stds_l.append(min_dist_stds)
    search_dists_l.append(search_dist)
    search_dist_stds.append(search_dist_std)
    
#%% Produce a plot of the success of the specific parameters over distance. 
# A label for the parameter can be sent just after the parameter values
fig, ax = beeplotter.plot_distance_v_param(min_dists_l, min_dist_stds_l, distances, memupdate_vals, 'Memory update')
fig.show()

#%% Produce a plot showing the search pattern width. Here we adjust the 
# ylabel using an optional variable
fig, ax = beeplotter.plot_distance_v_param(search_dists_l, search_dist_stds, distances, memupdate_vals, 'Memory update', ylabel='Search pattern width')
fig.show()

#%% Single flight can be generated like this
T=100
OUT, INB = trials.generate_dataset(T,T,1)

# Output is after statistical analysis (mean and std)
min_dist, min_dist_std, search_dist, search_dist_std = trials.analyze_inbound(INB,T,T)

