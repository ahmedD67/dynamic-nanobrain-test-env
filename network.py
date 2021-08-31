#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 14:08:46 2021

Notation:
    Capital letters A,B,C... denote matrices 
    Capital letters N,M,P... denote total number of something
    
    Internal voltage variables sets V as 3xN with number of nodes N.
    Internal dynamics governed by A of 3x3 so dV/dt = A @ V
    Input curents supplied as B = 3xN from B = WC where C are output currents
    from connecting nodes and W are weights
    We use dim W = M x N x P, for mapping from P nodes onto N nodes with M channels
    with C as M x P with the rule to leave the channel index to the left.
    This way we can do tensor updates with 
    B(ij) = sum(k) W (ijk) C (ik) 
    using the numpy routine einsum

@author: dwinge
"""

import numpy as np

# Global variable
NV = 3 # internal voltage degrees of freedom

class Layer :
    
    # Base class for all layers
    def __init__(self, N, layer_type):
        self.N = N
        self.layer_type = layer_type
        
    def get_node_name(self,node_idx,layer_idx=1) :
        if self.layer_type=='hidden' :
            # Name hidden layers using sequence (defaults to 'H')
            letters = 'HKLMN'
            letter = letters[layer_idx-1]
        else :
            # if not hidden layer, name either 'I' or 'O'
            letter = self.layer_type[0].upper()
        idx = str(node_idx)
        return letter+idx
    
    def get_names(self,layer_idx=1) :
        names = []
        for n in range(self.N) :
            names.append(self.get_node_name(n,layer_idx))
        return names

class HiddenLayer(Layer) :
    
    def __init__(self, N, output_channel, inhibition_channel, excitation_channel, 
                 gammas=np.zeros(6), func_Vgate_to_Isd=None, func_eta_LED=None, V_thres=1.2):
        
        Layer.__init__(self, N, layer_type='hidden')
        # Connections to the outside world
        self.out_channel = output_channel
        self.inh_channel = inhibition_channel
        self.exc_channel = excitation_channel
        # Phsyics variables
        self.gammas = gammas
        self.func_Vgate_to_Isd = func_Vgate_to_Isd
        self.func_eta_LED = func_eta_LED
        self.A = self.calc_A(*gammas[:-1]) # exclude gled
        # Set up internal variables
        self.V = np.zeros((NV,self.N))
        self.B = np.zeros_like(self.V)
        self.dV= np.zeros_like(self.V)
        # I is the current through the LED
        self.I = np.zeros(self.N)
        # Power is the outputted light, in units of current
        self.P = np.zeros(self.N)
        self.ISD = np.zeros_like(self.I)
        self.V_thres = V_thres
         
    # System matrix
    def calc_A(self,g11,g22,g13,g23,g33) :
        gsum = g13+g23+g33
        Amat = np.array([[-g11, 0, g11],
                         [0, -g22, g22],
                         [g13, g23, -gsum]]) # rows: inh, exc, gate
    
        return Amat    
    
    # Preferred way to set gammas as it updates A
    def set_gammas(self, gammas) :
        self.gammas = gammas
        self.A = self.calc_A(*gammas[:-1]) # exclude gled
        
    # Provide derivative
    def get_dV(self, t) :
         self.dV = self.A @ self.V + self.B
         return self.dV
        
    def update_V(self, dt) :
        self.V += dt*self.dV
        # Voltage clipping
        #if self.V > self.V_thres :
        #    self.V = self.V_thres
        # Second try at voltage clipping
        mask = (self.V > self.V_thres)
        self.V[mask] = self.V_thres
        mask = (self.V < -1*self.V_thres)
        self.V[mask] = -1*self.V_thres
    
    def update_I(self, dt) :
        # Get the source drain current from the transistor IV
        self.ISD = self.func_Vgate_to_Isd(self.V[2])
        self.I += dt*self.gammas[-1]*(self.ISD-self.I)
        # Convert current to power through efficiency function
        self.P = self.I*self.func_eta_LED(self.I)
    
    def reset_B(self) :
        self.B[:,:] = 0
    
    def reset_I(self):
        self.I[:] = 0
        
    def reset_V(self):
        self.V = np.zeros((NV,self.N))
    
    # Here we need an input mask to choose the communciation channels
    def update_B(self, weights, C) :
        E = weights.E
        W = weights.W
        BM = np.einsum('ijk,ik->ij',W,C)
        self.B += E @ BM
        # Normalize with the correct capactiances and units to get units of V/ns
        self.B[0,:] *= 1eX/Cinh
        self.B[1,:] *= 1eX/Cexc
    
# Inherits Layer
class InputLayer(Layer) :
    
    # These dictonaries hold function handles and arguments
    input_func_handles={}
    input_func_args={}
    
    def __init__(self, input_channels):
        Layer.__init__(self, len(input_channels), layer_type='input')
        self.channels = input_channels
        self.C = np.zeros((self.N,self.N))
        self.I = np.zeros(self.N)

    def set_input_func(self, channel, func_handle, func_args) :
        self.input_func_handles[channel] = func_handle
        self.input_func_args[channel] = func_args
        
    def get_input_current(self, t) :
        # Work on class object self.I instead of an arbitrary thing
        for key in self.channels :
            try :
                self.I[self.channels[key]] = self.input_func_handles[key](t,self.input_func_args[key])
            except :
                pass
            
        return self.I
        
    def update_C(self,t) :
        # Create a matrix out of the input currents
        self.C = np.diag(self.get_input_current(t))
    
    # This class has no B
    def reset_B(self) :
        return None
    
# Inherits Layer    
class OutputLayer(Layer) :
    
    def __init__(self, output_channels) :
        Layer.__init__(self, len(output_channels), layer_type='output')
    
    def update_B (self, weights, C) :
        # B is automatically allocated using this procedure
        W = weights.W
        self.B += np.einsum('ijk,ik->ij',W,C)
    
    def reset_B(self) :
        # This construction hopefully avoids reallocating memory
        try :
            self.B[:,:] = 0
        except :
            self.B = 0
    
# Connect layers and create a weight matrix
def connect_layers(down, up, layers, channels) :
    
    class Weights :
        
        def __init__(self, down, up, layers, channels) :
            # Should not assume correct ordering here
            self.from_layer = down
            self.to_layer = up
            self.channels = channels
            self.M = len(self.channels)
            
            # Initialize matrices
            L0 = layers[down]
            L1 = layers[up]
            self.W = np.zeros((self.M,L1.N,L0.N))
            
            # Check for connetions to hidden layers
            if L1.layer_type == 'hidden' :
                self.initialize_E(L1)
            if L0.layer_type == 'hidden' :
                self.initialize_D(L0)
                
            
        # Define an explicit connection
        def connect_nodes(self, from_node, to_node, channel, weight=1.0)  :   
            self.W[self.channels[channel],to_node,from_node] = weight
            return 0
        
        def initialize_E(self,L1) :
            # These channels are specific to L1 and could be reversed
            # compared to other nodes!
            inh_channel = self.channels[L1.inh_channel]
            exc_channel = self.channels[L1.exc_channel]
            self.E = np.zeros((NV,self.M))
            self.E[0,inh_channel]=-1. # inhibiting channel!
            self.E[1,exc_channel]=1.
        
        def initialize_D(self, L0) :
            # Construct the proper D for an hidden layer here
            self.D = np.zeros(self.M, dtype=float)
            out_channel = L0.out_channel # a key like 'pos'
            self.D[self.channels[out_channel]] = 1.
        
        def get_edges(self) :
            edges = {}
            for key in self.channels:
                edge_list = []
                m = self.channels[key]
                for down in range(len(self.W[0,0,:])) : # column loop (from)
                    for up in range(len(self.W[0,:,0])) : # row loop (to)
                        weight = self.W[m,up,down]
                        if weight > 0. :
                            edge_list.append((down,up,weight))
                edges[key] = edge_list
                    
            # Returns a dictionary over the edges of each channel
            return edges
        
        def print_W(self, *args):
            def print_key_W(key,W) :
                with np.printoptions(precision=2, suppress=True):
                    print('{0}:\n{1}'.format(key,W))
            if len(args) > 0:
                for key in args :        
                    print_key_W(key,self.W[self.channels[key],:,:])
            else :
                for key in channels :
                    print_key_W(key,self.W[self.channels[key],:,:])
                      
        def set_W(self, key, W) :
            self.W[self.channels[key],:,:] = W
            
    return Weights(down,up,layers,channels)
    
def reset(layers) :
    for key in layers :
        layers[key].reset_B()
        if layers[key].layer_type == 'hidden' :
            layers[key].reset_I()
            layers[key].reset_V()
    