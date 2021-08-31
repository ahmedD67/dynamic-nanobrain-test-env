#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 15:18:44 2021

@author: dwinge
"""

import numpy as np

# TODO: This part should be updated with an algorithm takes a time-step also
# in input currents to check if the time-step needs to be limited due to a 
# future pulse. For this the input currents need to updated 'ahead of time'
def evolve(t, layers, dVmax, dtmax) :
    # First we calculate the derivative of the internal variables to get dt
    max_dV = 0.
    for layer in layers.values() :
        if layer.layer_type == 'hidden' :
            # Calculate the derivative of the voltages
            dV = layer.get_dV(t)
            max_dV_layer = np.abs(dV).max()
            # Save the largest dV we find
            if max_dV_layer > max_dV :
                max_dV = max_dV_layer
          
    # Choose time-step accordingly
    if max_dV > 0 :
        dt = dVmax/max_dV
    else :
        dt = dtmax
        
    # Check also against maximum timestep
    dt = min(dt, dtmax)
    
    return dt
            
def update(dt, t, layers, weights, unity_coeff=1.0) :   
    # Time updating sequence
    # Update first all voltages V and reset currents in matrices B
    for layer in layers.values() :
        if layer.layer_type == 'hidden' :
            layer.update_V(dt)
            layer.update_I(dt) 
        
        if layer.layer_type == 'input' :
            layer.update_C(t)
            
        layer.reset_B()
    
    # Now we rewrite the currents B according to the weight rules
    for w in weights.values() :
        from_idx = w.from_layer
        to_idx = w.to_layer
        if layers[from_idx].layer_type == 'hidden' :
            # Inner matrix product with currents from D
            C = np.einsum('i,j->ij',w.D,layers[from_idx].P)
            # At this point we normalize with the unity coupling coefficient
            C *= unity_coeff
        else :
            C = layers[from_idx].C
       
        
        layers[to_idx].update_B(w,C)
    
    # Time step complete
    return 0