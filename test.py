#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 15:56:00 2022

@author: dwinge
"""

import numpy as np
import matplotlib.pyplot as plt

a=0

def hej(t) :
    return a

print(f'Evaluating hej as {hej(1)}')

for k in range(0,10) :
    a+=1
    print(f'Evaluating hej as {hej(1)}')
    
heading = 0

def get_flow(heading, v, pref_angle=np.pi/4):
    head_arr = np.array([[np.sin(heading + pref_angle),
                          np.cos(heading + pref_angle)],
                         [np.sin(heading - pref_angle),
                          np.cos(heading - pref_angle)]])
    return np.einsum('ijk,jk->ik',head_arr,v)

def tn2_activity(t,heading, velocity) :
    tn2 = get_flow(heading, velocity)
    # scale by the standard current factor
    return np.clip(tn2,0,1)

def rotate(theta, r):
    """Return new heading after a rotation around Z axis."""
    return (theta + r + np.pi) % (2.0 * np.pi) - np.pi


theta = np.linspace(-np.pi,np.pi)
v = 0.15 * np.array([np.sin(theta), np.cos(theta)])
noise=0.05
v += np.random.normal(scale=noise, size=v.shape)
tn2_output = tn2_activity(0,theta,v)
#tn2_output = get_flow(heading, v)

pref_angle=np.pi/4
heading_arr = np.array([[np.sin(theta + pref_angle),
                         np.cos(theta + pref_angle)],
                        [np.sin(theta - pref_angle),
                         np.cos(theta - pref_angle)]])

#%%
plt.plot(theta,tn2_output[0],label='L')
plt.plot(theta,tn2_output[1],label='R')
plt.legend()
plt.show()

#%% 
plt.plot(theta,v[0],label='vx')
plt.plot(theta,v[1],label='vy')
plt.legend()
plt.grid(True)
plt.show()