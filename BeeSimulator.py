#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 11:25:20 2021

@author: dwinge
"""

import numpy as np
import matplotlib.pyplot as plt
# %%
#from pathintegration import *
import pathintegration.trials as path_trials
import pathintegration.plotter as path_plotter
import pathintegration.cx_basic as path_cxbasic
import pathintegration.cx_rate as path_cxrate

T=500
headings, velocity = path_trials.generate_route(T=T, vary_speed=True)

path_plotter.plot_route(headings, velocity, T, 0)

tb1 = np.zeros(path_cxbasic.N_TB1)
memory = np.zeros(path_cxbasic.N_CPU4)
my_cx = path_cxrate.CXRate(noise=0.0) # use default settings

# Generate signals cl1, tn1 and tn2 for all time steps
cl1 = np.zeros((T,path_cxbasic.N_CL1))
tn1 = np.zeros((T,path_cxbasic.N_TN1))
tn2 = np.zeros((T,path_cxbasic.N_TN2))

for k in range(T) :
    tl2, cl1[k], tb1, tn1[k], tn2[k], memory, cpu4, cpu1, motor = path_trials.update_cells(headings[k], velocity[k], tb1, memory, my_cx)

# %% Plot the speed signals and cl1 signals

fig, axs = plt.subplots(4,1)

axs[0].plot(tn1[:,0],label='TN1_L')
axs[0].plot(tn1[:,1],label='TN1_R')
axs[0].legend()

axs[1].plot(tn2[:,0],label='TN2_L')
axs[1].plot(tn2[:,1],label='TN2_R')
axs[1].legend()

axs[2].pcolormesh(cl1.T, vmin=0, vmax=1,
                  cmap='viridis', rasterized=True)

axs[3].plot(velocity[:,0],label='vx')
axs[3].plot(velocity[:,1],label='vy')
axs[3].legend()

