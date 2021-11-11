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
                 device=None, Vthres=1.2) :
        
        Layer.__init__(self, N, layer_type='hidden')
        # Connections to the outside world
        self.out_channel = output_channel
        # These can be multiple, care taken in the constructing of weights
        self.inh_channel = inhibition_channel
        self.exc_channel = excitation_channel

        # Set up internal variables
        self.V = np.zeros((NV,self.N))
        self.B = np.zeros_like(self.V)
        self.dV= np.zeros_like(self.V)
        # I is the current through the LED
        self.I = np.zeros(self.N)
        # Power is the outputted light, in units of current
        self.P = np.zeros(self.N)
        self.ISD = np.zeros_like(self.I)
        self.Vthres = Vthres

        # Device object hold A, for example
        self.device=device

    def assign_device(self, device) :
        self.device=device
        
    # Provide derivative
    def get_dV(self, t) :     
        self.dV = self.device.A @ self.V + self.B
        return self.dV
        
    def update_V(self, dt) :
        self.V += dt*self.dV
        # Voltage clipping
        #if self.V > self.Vthres :
        #    self.V = self.Vthres
        # Second try at voltage clipping
        mask = (self.V > self.Vthres)
        self.V[mask] = self.Vthres
        mask = (self.V < -1*self.Vthres)
        self.V[mask] = -1*self.Vthres
    
    def update_I(self, dt) :
        # Get the source drain current from the transistor IV
        self.ISD = self.device.transistorIV(self.V[2])
        self.I += dt*self.device.gammas[-1]*(self.ISD-self.I)
        # Convert current to power through efficiency function
        self.P = self.I*self.device.eta_ABC(self.I)
    
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
        B_to_add = E @ BM 
        # Normalize with the correct capactiances and units to get units of V/ns
        # This could be inserted into E, 5 lines above
        B_to_add[0,:] *= 1e-18/self.device.p_dict['Cinh']
        B_to_add[1,:] *= 1e-18/self.device.p_dict['Cexc']
        self.B += B_to_add
    
# Inherits Layer
class InputLayer(Layer) :
        
    def __init__(self, input_channels):
        Layer.__init__(self, len(input_channels), layer_type='input')
        self.channels = input_channels
        self.C = np.zeros((self.N,self.N))
        self.I = np.zeros(self.N)
        # These dictonaries hold function handles and arguments
        self.input_func_handles={}
        self.input_func_args={}

    def set_input_func(self, channel, func_handle, func_args=None) :
        self.input_func_handles[channel] = func_handle
        self.input_func_args[channel] = func_args
        
    def get_input_current(self, t) :
        # Work on class object self.I instead of an arbitrary thing
        for key in self.channels :
            try :
                if self.input_func_args[key] is not None:
                    self.I[self.channels[key]] = self.input_func_handles[key](t,*self.input_func_args[key])
                else :
                    self.I[self.channels[key]] = self.input_func_handles[key](t)
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
    
    def __init__(self, output_channels, teacher_delay=None) :
        Layer.__init__(self, len(output_channels), layer_type='output')
        self.channels = output_channels
        self.C = np.zeros((self.N,self.N))
        self.B = np.zeros_like(self.C)
        self.I = np.zeros(self.N)
        # These dictonaries hold function handles and arguments
        self.output_func_handles={}
        self.output_func_args={}
        self.teacher_delay=teacher_delay

    def set_teacher_delay(self,delay) :
        # introduce a class variable to handle time delays
        self.teacher_delay=delay
        self.search_teacher=0
        # t and then B11, B12..
        self.teacher_memory=np.zeros((len(self.B.flatten())+1))
    
    def set_output_func(self, channel, func_handle, func_args=None) :
        self.output_func_handles[channel] = func_handle
        self.output_func_args[channel] = func_args
        
    def get_output_current(self, t) :
        # Work on class object self.I instead of an arbitrary thing
        for key in self.channels :
            try :
                if self.output_func_args[key] is not None:
                    self.I[self.channels[key]] = self.output_func_handles[key](t,*self.output_func_args[key])
                else :
                    self.I[self.channels[key]] = self.output_func_handles[key](t)
            except :
                pass
            
        return self.I
    
    def update_B (self, weights, C) :
        # B is automatically allocated using this procedure
        W = weights.W
        # This automatically yields current in nA
        self.B += np.einsum('ijk,ik->ij',W,C)
    
    def reset_B(self) :
        # This construction hopefully avoids reallocating memory
        try :
            self.B[:,:] = 0
        except :
            self.B = 0
            
    def lin_intp(self,x,f) :
        # specify interval
        x0 = f[0,0]
        x1 = f[1,0]
        p1 = (x-x0)/(x1-x0) # fraction to take of index 1
        p0 = 1-p1 # fraction to take from index 1
        return p0*f[0]+p1*f[1]
            
    def update_C_from_B(self,t) :
        # Here it is possible to add a nonlinear function as well
        # Need to take the delay into account here
        if self.teacher_delay is not None :
            # First, we need to store B with correct time-stamp
            self.teacher_memory=np.vstack((self.teacher_memory,np.concatenate((np.array([t]),self.B.flatten()))))
            # Now we choose the correct memory point to choose from
            t_find = max(t-self.teacher_delay,0) # starts at 0
            # Search for point just before t
            while t_find > self.teacher_memory[self.search_teacher,0] :
                self.search_teacher += 1
            
            if self.search_teacher > 0 :
                teacher_signal = self.lin_intp(t_find,
                                               self.teacher_memory[self.search_teacher-1:self.search_teacher+1])
            else : 
                teacher_signal = self.teacher_memory[0]
            
            B_for_update = teacher_signal[1:].reshape((self.N,self.N))
        else :
            B_for_update = self.B
            
        self.C = np.copy(B_for_update)
        
    def update_C(self,t) :
        # Create a matrix out of the output currents
        self.C = np.diag(self.get_output_current(t))
        
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
            
            # Check for connections to hidden layers
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
            # Now with support for multiple inhibitory and excitatory channels
            try :
                inh_channel = [self.channels[channel] for channel in L1.inh_channel]
            except : # if inh_channel is not an iterable
                inh_channel = self.channels[L1.inh_channel] # old version
            try :
                exc_channel = [self.channels[channel] for channel in L1.exc_channel]
            except :
                exc_channel = self.channels[L1.exc_channel] # old version
                
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
            
        def ask_W(self) :
            print(f'Set weights by set_W(channel key, array of size M x N \nwith {self.W[0,:,:].shape}')
            
    return Weights(down,up,layers,channels)
    
def reset(layers) :
    for key in layers :
        layers[key].reset_B()
        if layers[key].layer_type == 'hidden' :
            layers[key].reset_I()
            layers[key].reset_V()
    