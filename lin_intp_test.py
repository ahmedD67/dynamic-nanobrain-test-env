#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 13:42:51 2021

@author: dwinge
"""
import numpy as np
import matplotlib.pyplot as plt
def lin_intp(x,f) :
    # specify interval
    x0 = f[0,0]
    x1 = f[1,0]
    p1 = (x-x0)/(x1-x0) # fraction to take of index 1
    p0 = 1-p1 # fraction to take from index 1
    return p0*f[0]+p1*f[1]
        
x_sin = np.linspace(0,2)
y_sin = np.sin(x_sin)
N = len(x_sin)

f_sin = np.hstack((x_sin.reshape(N,1),y_sin.reshape(N,1)))

find_x = 0.0

search_k = 1

while find_x > x_sin[search_k] :
    search_k += 1
    
find_y = lin_intp(find_x,f_sin[search_k-1:search_k+1])

fig, ax = plt.subplots()

ax.plot(x_sin,y_sin)
ax.plot(find_y[0],find_y[1],'kx')

plt.show()