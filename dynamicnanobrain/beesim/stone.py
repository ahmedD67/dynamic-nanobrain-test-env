#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 15:53:48 2022

@author: dwinge
"""
import time

import numpy as np
import pandas as pd
from ..core import networker as nw
from ..core import plotter
from ..core import logger
from ..core import timemarching as tm


N_TB1=8
N_CL1=16
N_CPU4=16
N_Pontine=16
N_TN2=2
N_CPU1a=14
N_CPU1b=2

default_update_m = 0.25

# By Tom Stone (Stone et al Curr Biology 2017)
def gen_tb_tb_weights(weight=1.):
    """Weight matrix to map inhibitory connections from TB1 to other neurons"""
    W = np.zeros([N_TB1, N_TB1])
    sinusoid = -(np.cos(np.linspace(0, 2*np.pi, N_TB1, endpoint=False)) - 1)/2
    for i in range(N_TB1):
        values = np.roll(sinusoid, i)
        W[i, :] = values
    return weight * W

class TravelLog :
    
    # This should store the velocity and heading at each instance. Might as well
    # store also the position
    def __init__(self) :
        self.list_data = []
        self.column_labels = self.columnnames()
        
    def columnnames(self) :
        names=['Time','x','y','vx','vy','heading']
        return names
        
    def add_tstep(self,t,x,v,heading,motor) :
        row = [t]
        # position
        pos = x.flatten(order='F').tolist()
        row += pos
        # velocity
        vel = v.flatten(order='F').tolist()
        row += vel
        row += [heading] # skipped motor here, plus needs to be a list to be appended
        self.list_data.append(row)
        
    def get_timelog(self) :
        # Convert to pandas data frame
        return pd.DataFrame(self.list_data, columns=self.column_labels)
        
class StoneNetwork :
    
    def __init__(self,name='StoneNetwork', input_scaling=1.0, tb1_c=0.33, 
                 mem_update_h=0.0025) :
        self.name = name
        self.input_scaling=input_scaling
        self.tb1_c=tb1_c
        self.mem_update_h = mem_update_h
        self.layers, self.weights = self.initialize_nw()
        # Initialize a dictionary for the devices
        self.devices = {}

        
    def initialize_nw(self) :
        # Defining layers with custom labels        
        layers = {}
        # input layers
        layers['CL1'] = nw.InputLayer(N=N_CL1)
        layers['TN2'] = nw.InputLayer(N=N_TN2)
        # ring layers
        layers['TB1'] = nw.HiddenLayer(N=N_TB1, output_channel='green', inhibition_channel='green', excitation_channel='blue')
        layers['CPU4'] = nw.HiddenLayer(N=N_CPU4, output_channel='orange', inhibition_channel='green', excitation_channel='brown')
        layers['Pontine'] = nw.HiddenLayer(N=N_Pontine,output_channel='green',inhibition_channel='white',excitation_channel='orange')
        layers['CPU1a'] = nw.HiddenLayer(N=N_CPU1a, output_channel='black', inhibition_channel='green', excitation_channel='orange')
        layers['CPU1b'] = nw.HiddenLayer(N=N_CPU1b, output_channel='purple', inhibition_channel='green', excitation_channel='orange')
        
        weights = {}
        # Set up the connections
        weights['CL1->TB1']=nw.connect_layers('CL1', 'TB1', layers, channel='blue')
        W = np.tile(np.diag([1.0]*N_TB1),2)
        W *= (1.-self.tb1_c) # scaling factor to weigh TB1 and CL1 input to TB1
        weights['CL1->TB1'].set_W(W)
        
        weights['TB1->TB1']=nw.connect_layers('TB1','TB1',layers,channel='green')
        W = gen_tb_tb_weights(weight=self.tb1_c)
        weights['TB1->TB1'].set_W(W)
        
        weights['TB1->CPU4']=nw.connect_layers('TB1', 'CPU4', layers, channel='green')
        W = np.tile(np.diag([1.0]*N_TB1),(2,1)) 
        weights['TB1->CPU4'].set_W(W)
        # The memory is updated with a lower weight
        weights['TB1->CPU4'].scale_W(self.mem_update_h)
        
        weights['TB1->CPU1a']=nw.connect_layers('TB1', 'CPU1a', layers, channel='green')
        W = np.tile(np.diag([1.0]*N_TB1),(2,1))
        weights['TB1->CPU1a'].set_W(W[1:-1])
        
        weights['TB1->CPU1b']=nw.connect_layers('TB1', 'CPU1b', layers, channel='green')
        W = np.zeros((2,N_TB1))
        W[0,-1] = 1.0
        W[1,0]  = 1.0
        weights['TB1->CPU1b'].set_W(W)
        
        weights['TN2->CPU4']=nw.connect_layers('TN2', 'CPU4', layers, channel='brown')
        W = np.concatenate((np.tile(np.array([1,0]),(N_CPU4//2,1)),np.tile(np.array([0,1]),(N_CPU4//2,1)))) 
        weights['TN2->CPU4'].set_W(W)
        # The memory is updated with a lower weight
        weights['TN2->CPU4'].scale_W(self.mem_update_h)
        
        weights['CPU4->CPU1a']=nw.connect_layers('CPU4', 'CPU1a', layers, channel='orange')
        W =   np.array([[0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], #2
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], #15
                        ])
        weights['CPU4->CPU1a'].set_W(W)
        
        weights['CPU4->CPU1b']=nw.connect_layers('CPU4', 'CPU1b', layers, channel='orange')
        W = np.zeros((2,N_CPU4))
        W[0,0]=1.0
        W[-1,-1]=1.0
        weights['CPU4->CPU1b'].set_W(W)
        
        weights['CPU4->Pontine']=nw.connect_layers('CPU4', 'Pontine', layers, channel='orange')
        W = np.diag([1.0]*N_CPU4)
        weights['CPU4->Pontine'].set_W(W)
        
        weights['Pontine->CPU1a']=nw.connect_layers('Pontine', 'CPU1a', layers, channel='green')
        W =   np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], #2
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #15])
                        ],dtype=float)
        weights['Pontine->CPU1a'].set_W(W)
        
        weights['Pontine->CPU1b']=nw.connect_layers('Pontine', 'CPU1b', layers, channel='green')
        W = np.zeros((2,N_Pontine))
        W[0,4] = 1.0
        W[1,11]= 1.0
        weights['Pontine->CPU1b'].set_W(W)
        
        return layers, weights
        
    def show_network(self, savefig=False,layout='spring',**kwargs) :
        pos = plotter.visualize_network(self.layers, self.weights, 
                                        #exclude_nodes={0:['I1','I2'],3:['O1','O2']},
                                        node_size=100,
                                        layout=layout, 
                                        show_edge_labels=False,
                                        savefig=savefig,
                                        **kwargs)
        
        return pos
        
    def show_weights(self, **kwargs) :
        fig, ax = plotter.plot_weights(self.weights, **kwargs)
        return fig, ax
        
    def assign_device(self, device_dict, unity_key) :
        for key in device_dict :
            self.layers[key].assign_device(device_dict[key])
            self.devices[key] = device_dict[key]
        unity_device = device_dict[unity_key]
        self.unity_coeff, self.Imax = unity_device.inverse_gain_coefficient(unity_device.eta_ABC, self.layers[unity_key].Vthres)

    def show_devices(self, Vleak_dict, **kwargs) :
        # Send the class own device dict
        plotter.plot_devices(self.devices, Vleak_dict, **kwargs)
        
    def assign_memory(self, key, device) :
        self.layers[key].assign_device(device)
        self.mem_coeff, self.mem_Imax = device.inverse_gain_coefficient(device.eta_ABC, self.layers[key].Vthres)
        return self.mem_coeff, self.mem_Imax
    
    def specify_inputs(self,key,input_handle) :
        self.layers[key].set_input_vector_func(func_handle=input_handle(self.Imax*self.input_scaling))
        
    def updateheading(self,heading,layers,update_m=default_update_m) :
        # sum left and right hand turning neurons
        r_right = sum(layers['CPU1a'].P[:7]) + layers['CPU1b'].P[1]
        r_left  = sum(layers['CPU1a'].P[7:]) + layers['CPU1b'].P[0]
        return update_m*(r_right-r_left)*self.unity_coeff
        
    def evolve(self,T,reset=True,t0=0.,inbound=False,savestep=1.0,
               initial_pos=(0,0),initial_vel=(0,0),initial_heading=0,
               a=0.1, drag=0.15, updateheading_m=default_update_m,
               printdiag=False) : 
               
        # Start time is t
        t=t0
        # These parameters are used to determine an appropriate time step each update
        dtmax = 0.1 # ns 
        dVmax = 0.01 # V # I loosen this a little bit now
        # To sample result over a fixed time-step, use savetime
        savetime = max(savestep,dtmax) + t0
        
        if reset :
            # option to keep reservoir in its current state
            nw.reset(self.layers)
            
        if not inbound :
            # Initialize activity in the memory cells
            self.layers['CPU4'].V[:,:]=self.layers['CPU4'].Vthres/2
            
        # Create a log over the dynamic data
        time_log = logger.Logger(self.layers,feedback=True) # might need some flags
        # write zero point
        time_log.add_tstep(t, self.layers, self.unity_coeff)
        
        if inbound :
            # Create a travel log for the return journey
            travel_log = TravelLog()
            v = np.array(initial_vel)
            x = np.array(initial_pos)
            heading = initial_heading
            travel_log.add_tstep(t, x, v, heading, 0.0) # motor=0 for duplcate first step
            
            # Allow the agent to determine its speed through TN2 layer
            def get_flow(heading, v, pref_angle=np.pi/4):
                head_arr = np.array([[np.sin(heading + pref_angle),
                                      np.cos(heading + pref_angle)],
                                     [np.sin(heading - pref_angle),
                                      np.cos(heading - pref_angle)]])
                return head_arr @ v

            def tn2_activity(t) :
                tn2 = get_flow(heading, v)
                # scale by the standard current factor
                return np.clip(tn2,0,1)*self.Imax*self.input_scaling
            
            # Specify the output function
            self.layers['TN2'].set_input_vector_func(tn2_activity)
            
            # Allow the agent to determine its heading through CL1 layer
            TL_angles = np.tile(np.arange(0,8),2)*np.pi/4
            
            from scipy.special import expit
                
            ### NOT SURE THAT THIS IS THE BEST PLACE TO PUT IT
            # From Stone Curr Biol 2017
            # TUNED PARAMETERS:
            tl2_slope_tuned = 6.8
            tl2_bias_tuned = 3.0   
            cl1_slope_tuned = 3.0
            cl1_bias_tuned = -0.5
            
            def noisy_sigmoid(v, slope=1.0, bias=0.5, noise=0.01):
                """Takes a vector v as input, puts through sigmoid and
                adds Gaussian noise. Results are clipped to return rate
                between 0 and 1"""
                sig = expit(v * slope - bias)
                if noise > 0:
                    sig += np.random.normal(scale=noise, size=len(v))
                return np.clip(sig, 0, 1)
            ###
            
            def tl2_activity(heading):
                output = np.cos(heading - TL_angles)
                return noisy_sigmoid(output, tl2_slope_tuned, tl2_bias_tuned, noise=0.0)
            
            def cl1_activity(t):
                output = tl2_activity(heading)
                sig = noisy_sigmoid(-output, cl1_slope_tuned, cl1_bias_tuned, noise=0.0)
                # scale by the standard current factor
                return sig*self.Imax*self.input_scaling
            
            def rotate(theta, r):
                """Return new heading after a rotation around Z axis."""
                return (theta + r + np.pi) % (2.0 * np.pi) - np.pi

            # Specify the output function
            self.layers['CL1'].set_input_vector_func(cl1_activity)
        
        start = time.time()
        
        while t < T:
            # evolve by calculating derivatives, provides dt
            dt = tm.evolve(t, self.layers, dVmax, dtmax )
        
            # update with explicit Euler using dt
            # supplying the unity_coeff here to scale the weights
            tm.update(dt, t, self.layers, self.weights, self.unity_coeff, t0)
            
            if inbound :
                # update agent heading
                motor = self.updateheading(heading,self.layers,updateheading_m)
                # use a dt here since we do many turns per time unit
                heading = rotate(heading,motor*dt)
                # velocity using drag force and acceleration
                v[0] += np.sin(heading)*a*dt
                v[1] += np.cos(heading)*a*dt
                v -= v*drag*dt
                x += v*dt
                if printdiag :
                    print(f'Heading is: {heading}, velocity is {v}')
                            
            
            t += dt
            # Log the progress
            if t > savetime :
                # Put log update here to have (more or less) fixed sample rate
                # Now this is only to check progress
                print(f'Time at t={t} ns') 
                savetime += savestep         
                time_log.add_tstep(t, self.layers, self.unity_coeff)
                
                if inbound :
                    travel_log.add_tstep(t,x,v,heading,motor)
        
        end = time.time()
        print('Time used:',end-start)
        
        # This is a large pandas data frame of all system variables
        result = time_log.get_timelog()
        
        if inbound :
            travel = travel_log.get_timelog()
            return result, travel
        else :
            return result