#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 09:10:12 2021

@author: dwinge
"""
import numpy as np
import network as nw
import plotter

class EchoStateNetwork :
    
    def __init__(self,N,input_handle=None,bias_handle=None,teacher_handle=None,
                 input_scaling=1.0,bias_scaling=1.0,teacher_scaling=1.0,
                 spectral_radius=1.0, sparsity=0.9, timescale=0.5,
                 device=None,silent=False,savefig=False,seed=42) :
        
        self.N = N
        self.input_handle = input_handle
        self.bias_handle = bias_handle
        self.teacher_handle = teacher_handle
        self.sparsity = sparsity
        self.timescale = timescale
        self.silent = False
        self.device = device
        self.savefig = savefig
        self.seed = seed
                
        self.specify_network(spectral_radius,input_scaling,bias_scaling,teacher_scaling)
        # Not used anymore
        #self.layers, self.weights = self.initialize_nw(self.N,self.sparsity)
        # Show the network when first created
        self.show_network(self.savefig)
        
        # If the handles are stored from the start, this can be given as an
        # internal update if layers are redefined.
        self.specify_inputs(input_handle,bias_handle,teacher_handle)

    def _unscale_teacher(self, teacher_scaled):
        """inverse operation of the _scale_teacher method."""
        return teacher_scaled / self.teacher_scaling
    
    def update_network(self,spectral_radius,input_scaling=None,
                        bias_scaling=None,
                        teacher_scaling=None) :
        
        self.spectral_radius = spectral_radius
        
        if input_scaling is not None :
            self.input_scaling = input_scaling
        if teacher_scaling is not None :
            self.teacher_scaling = teacher_scaling   
        if bias_scaling is not None :
             self.bias_scaling = bias_scaling   
             
        self.layers, self.weights, self.channels = self.initialize_nw(self.N,self.sparsity)
        # Reassign the device
        if self.device is not None :
            self.assign_device(self.device)
        # Reassign the input and output handles
        self.specify_inputs(self.input_handle,self.bias_handle,self.teacher_handle)
        self.set_delay(self.timescale)
    
    def specify_network(self,spectral_radius,input_scaling=None,
                        bias_scaling=None,
                        teacher_scaling=None) :
        
        self.spectral_radius = spectral_radius
        
        if input_scaling is not None :
            self.input_scaling = input_scaling
        if teacher_scaling is not None :
            self.teacher_scaling = teacher_scaling   
        if bias_scaling is not None :
             self.bias_scaling = bias_scaling   
             
        # instead of a return statement we do this to be able to call it from
        # outside the class scope easily
        self.layers, self.weights, self.channels = self.initialize_nw(self.N,self.sparsity)
  
        
    def initialize_nw(self,N,sparsity,initialize_output=False) :
        # We will use a standard set of channels for now    
        channel_list = ['blue','red','green']
        # Automatically generate the object that handles them
        channels = {channel_list[v] : v for v in range(len(channel_list))}
        
        layers = {} 
        # An input layer automatically creates on node for each channel that we define
        layers[0] = nw.InputLayer(input_channels=channels)
        # Two intertwined hidden layers, one excitatiory, one inhibitory
        layers[1] = nw.HiddenLayer(N//2, output_channel='blue',excitation_channel=('blue','green'),inhibition_channel='red')
        layers[2] = nw.HiddenLayer(N//2, output_channel='red' ,excitation_channel=('blue','green'),inhibition_channel='red')
        # Output layer
        layers[3] = nw.OutputLayer(output_channels=channels) 
        
        # Define the overall connectivity
        weights = {}
        # The syntax is connect_layers(from_layer, to_layer, layers, channels)
        # Connections into the reservoir from input layer
        weights['inp->hd0'] = nw.connect_layers(0, 1, layers, channels)
        weights['inp->hd1'] = nw.connect_layers(0, 2, layers, channels)
        # Connections between reservoir nodes
        weights['hd0->hd1'] = nw.connect_layers(1, 2, layers, channels)
        weights['hd1->hd0'] = nw.connect_layers(2, 1, layers, channels)
        # Intralayer connections
        weights['hd0->hd0'] = nw.connect_layers(1, 1, layers, channels)
        weights['hd1->hd1'] = nw.connect_layers(2, 2, layers, channels)
        # Connections back into reservoir from output
        weights['out->hd0'] = nw.connect_layers(3, 1, layers, channels)
        weights['out->hd1'] = nw.connect_layers(3, 2, layers, channels)
        
        # Initiate the weights randomly
        rng = np.random.RandomState(self.seed)
        # Some parameters, fixed for now, kill off red weights
        Win_scale = {'blue':1.0,'red':0.0,'green':1}
        Wfb_scale = 1.0

        # Input weights to all of the input units
        W_in = rng.rand(N, len(channels))

        # Put each weight column in a specific channel
        for key in channels :
            k = channels[key]
            W_key = np.zeros_like(W_in)
            W_key[:,k] = W_in[:,k]
            weights['inp->hd0'].set_W(key,Win_scale[key]*W_key[:N//2]) # first half
            weights['inp->hd1'].set_W(key,Win_scale[key]*W_key[N//2:]) # second half
        
        # Now initiate reservoir weights
        W_partition = {'hd0->hd0':(0,N//2,0,N//2), # ++
                       'hd0->hd1':(N//2,N,0,N//2), # +- 
                       'hd1->hd1':(N//2,N,N//2,N), # --
                       'hd1->hd0':(0,N//2,N//2,N)} # -+

        # Generate a large matrix of values for the whole reservoir 
        W_res = rng.rand(N, N)  # all positive [0,1)
        # Rightmost half side of this matrix will effectively be negative weights (-- and -+)
        W_res[:,N//2:] *= -1 

        # Delete the fraction of connections given by sparsity:
        W_res[rng.rand(*W_res.shape) < sparsity] = 0
        # Delete any remaining diagonal elements (we can't have those)
        for k in range(0,N) :
            W_res[k,k] = 0.
            
        radius = np.max(np.abs(np.linalg.eigvals(W_res)))
        # rescale them to reach the requested spectral radius:
        W_res = W_res * (self.spectral_radius / radius)
        # Now shift the signs again for the implementation
        W_res[:,N//2:] *= -1 

        for k, connection in enumerate(W_partition) :
            key = channel_list[k//2]
            A,B,C,D = W_partition[connection]
            weights[connection].set_W(key,W_res[A:B,C:D])

        # Output weights from reservoir to the output units
        if initialize_output :
            W_out = rng.rand(len(channels),N)
    
            # Put each weight column in a specific channel
            for key in channels :
                # Green is the bias and not forwarded to output
                if key != 'green' :
                    k = channels[key]
                    W_key = np.zeros_like(W_out)
                    W_key[k] = W_out[k]
                    weights['hd0->out'].set_W(key,W_key[:,:N//2])
                    weights['hd1->out'].set_W(key,W_key[:,N//2:])

        # Feedback weights from output nodes back into reservoir 
        # These are random. The weights from the extended network state are trained.
        W_feedb = rng.rand(N,len(channels))

        # Put each weight column in a specific channel
        for key in channels :
            # AT THIS POINT, WE USE ONLY POSITIVE SIGNAL AT OUTPUT
            if key == 'blue' :
                k = channels[key]
                W_key = np.zeros_like(W_feedb)
                # Pick a column
                W_key[:,k] = W_feedb[:,k]
                weights['out->hd0'].set_W(key,Wfb_scale*W_key[:N//2,:])
                weights['out->hd1'].set_W(key,Wfb_scale*W_key[N//2:,:])
        
        return layers, weights, channels
        
    def add_trained_weights(self, W_out, bias_scaling=None) :
        # First we define the weights to add
        self.weights['hd0->out'] = nw.connect_layers(1, 3, self.layers, self.channels)
        # We don't add the inhibition layer for now
        #self.weights['hd1->out'] = nw.connect_layers(2, 3, self.layers, self.channels)
        self.weights['inp->out'] = nw.connect_layers(0, 3, self.layers, self.channels)
        
        W_res_out = np.zeros(self.weights['hd0->out'].ask_W(silent=True))
        # Set the weights for the blue output
        W_res_out[self.channels['blue']]=W_out[0,:self.N//2]
        self.weights['hd0->out'].set_W('blue', W_res_out)
        # Update the B value correspondingly, as the are newly connected
        C = np.einsum('i,j->ij',self.weights['hd0->out'].D,self.layers[1].P)
        # At this point we normalize with the unity coupling coefficient
        C *= self.unity_coeff
        # Add the corresponding currents to Output layer
        self.layers[3].update_B(self.weights['hd0->out'],C)
            
        W_inout = np.zeros(self.weights['inp->out'].ask_W(silent=True))
        # Set the weights for the input-output coupling
        W_inout[self.channels['blue']]=W_out[0,self.N//2:] # 3 last elements
        self.weights['inp->out'].set_W('blue', W_inout)
        # Also here we add the currents to B
        C = self.layers[0].C
        self.layers[3].update_B(self.weights['inp->out'],C)
        
        if bias_scaling is not None :
            self.weights['inp->hd0'].set_W('green',self.weights['inp->hd0'].W[self.channels['green'],:,:]*bias_scaling)
            self.weights['inp->hd1'].set_W('green',self.weights['inp->hd1'].W[self.channels['green'],:,:]*bias_scaling)
        
        
    def show_network(self, savefig=False,layout='spring') :
        plotter.visualize_network(self.layers, self.weights, 
                                  exclude_nodes={0:['I1','I2'],3:['O1','O2']},
                                  node_size=100,
                                  layout=layout, 
                                  show_edge_labels=False,
                                  savefig=savefig)
            
    def produce_movie(self,movie_series) :
        plotter.movie_maker(movie_series,self.layers, self.weights, 
                            exclude_nodes={0:['I1','I2'],3:['O1','O2']},
                            node_size=100,
                            layout='spring', 
                            show_edge_labels=False)
            
    def assign_device(self, device) :
        self.layers[1].assign_device(device)
        self.layers[2].assign_device(device)
        self.unity_coeff, self.Imax = device.inverse_gain_coefficient(device.eta_ABC, self.layers[1].Vthres)
        
    def specify_inputs(self,input_handle,bias_handle,teacher_handle) :
        if input_handle is not None :
            self.layers[0].set_input_func(channel='blue',
                                          func_handle=input_handle(self.Imax*self.input_scaling))    
            self.input_handle = input_handle
        
        if bias_handle is not None :
            self.layers[0].set_input_func(channel='green',
                                          func_handle=bias_handle(self.Imax*self.bias_scaling))
            self.bias_handle = bias_handle
            
        if teacher_handle is not None :
            self.layers[3].set_output_func(channel='blue',
                                           func_handle=teacher_handle(self.Imax*self.teacher_scaling))
            self.teacher_handle = teacher_handle
            
    def set_delay(self, delay,nsave=None) :
        # This is the timescale of the system
        self.timescale=delay
        self.layers[3].set_teacher_delay(self.timescale,nsave)
        
    def evolve(self,T,reset=True,t0=0.,teacher_forcing=False) : 
        import logger
        import timemarching as tm
        import time
        
        # Start time is t
        t=t0
        # These parameters are used to determine an appropriate time step each update
        dtmax = 0.5 # ns 
        dVmax = 0.01 # V # I loosen this a little bit now
        # To sample result over a fixed time-step, use savetime
        savestep = 0.5 # ns
        savetime = max(savestep,dtmax)
        
        if reset :
            # option to keep reservoir in its current state
            nw.reset(self.layers)
            
        # Create a log over the dynamic data
        time_log = logger.Logger(self.layers,self.channels,feedback=True) # might need some flags
        # write zero point
        time_log.add_tstep(t, self.layers, self.unity_coeff)
        
        start = time.time()
        
        while t < T:
            # evolve by calculating derivatives, provides dt
            dt = tm.evolve(t, self.layers, dVmax, dtmax )
        
            # update with explicit Euler using dt
            # supplying the unity_coeff here to scale the weights
            tm.update(dt, t, self.layers, self.weights, self.unity_coeff, t0, teacher_forcing)
            
            t += dt
            # Log the progress
            if t > savetime :
                # Put log update here to have (more or less) fixed sample rate
                # Now this is only to check progress
                print(f'Time at t={t} ns') 
                savetime += savestep
            
            time_log.add_tstep(t, self.layers, self.unity_coeff)
        
        end = time.time()
        print('Time used:',end-start)
        
        # This is a large pandas data frame of all system variables
        result = time_log.get_timelog()
        
        return result
    
    def interp_columns(self,result,tseries,header_exp=None,columns=None,regex=None,return_df=False) :
        # TODO: Could be tidied up a bit with the regex
        from scipy.interpolate import interp1d 
        # Extract time column
        tcol = result['Time']
        # Extract all Pout for the states
        if header_exp is not None :
            headers = [c for c in result.columns if header_exp in c]
            df = result[headers]
        elif columns is not None :
            headers= columns
            df = result[headers]
        elif regex is not None :
            df = result.filter(regex=regex)
        else :
            df = result
            
        # Create interpolation function
        df_interp = interp1d(tcol,df,axis=0)
        
        if return_df :
            import pandas as pd
            df_new = pd.DataFrame(df_interp(tseries),columns=df.columns)
            try :
                # Try to add the time column as well
                df_new.insert(0,'Time',tseries)
            except ValueError :
                # If it is already there we end up here
                pass
            
        else :
            df_new = df_interp(tseries)
        
        return df_new
        
    def harvest_states(self,T,t0=0.,reset=True) :
        # First we evolve to T in time from optional t0
        result = self.evolve(T,t0=t0,teacher_forcing=True,reset=reset)
        # Now we fit the output weights using ridge regression
        if not self.silent:
            print("harvesting states...")

        # Secondly, we employ a discrete sampling of the signals
        tseries = np.arange(t0,T,step=self.timescale,dtype=float)
        # States
        states_series = self.interp_columns(result,tseries,regex='H.\d?-Pout')
        # Input signals
        inputs_columns = [c for c in result.columns if ('I0' in c) or ('I1' in c) or ('I2' in c)]
        inputs_series = self.interp_columns(result,tseries,columns=inputs_columns)
        # Teacher signal
        teacher_series = self.interp_columns(result,tseries,header_exp='O0-Pinp-blue')

        # Now we formulate extended states including the input signal
        extended_states = np.hstack((states_series, inputs_series))
        
        print('Voltages at the last point (H,K):\n', self.layers[1].V)
        print(self.layers[2].V)   
        print('Currents out from H:', self.layers[1].P)
        
        
        return tseries, extended_states, teacher_series

    def fit(self, states, target, beta=100, regularization=True) :
        # we'll disregard the first few states:
        transient = min(int(target.shape[0] / 10), 100)
        
        if regularization:
            # Use regularization parameter beta to find output weights
            Nx = states.shape[1]
            # Part to invert
            X = states[transient:].T @ states[transient:] + beta * np.diag([1]*Nx)
            # Part including the target
            YX = target[transient:].T @ states[transient:]
            # Final expression
            W_out = YX @ np.linalg.inv(X)
        else :
            # Solve for W_out using direct pseudoinverse
            W_out = np.dot(np.linalg.pinv(states[transient:, :]),
                           target[transient:, :]).T
        
   
        # Now we need to specify some weights from the input and reservoir unit
        print('The following weights were found:\n', W_out)
        # apply learned weights to the collected states:
        if self.bias_scaling == 0. :
            self.add_trained_weights(W_out,0)
        else :
            self.add_trained_weights(W_out)

        # Generate the prediction for the traning data
        pred_train = self._unscale_teacher(np.dot(states, 
                                                  W_out.T))
        
        error = np.sqrt(np.mean((pred_train - target)**2))/self.Imax
        
        if not self.silent:
            print('Training error:', error)
            
        return pred_train, error
        
    def predict(self,t0,T,output_all=False) :
        # Assume here that we continue on from the state of the reservoir
        if not self.silent:
            print("predicting...")
                
        # First we evolve to T in time from optional t0, without resetting 
        result = self.evolve(T,t0=t0,reset=False)
        
        # Secondly, we employ a discrete sampling of the signals
        tseries = np.arange(t0,T,step=self.timescale,dtype=float)
        # States
        output_series = self.interp_columns(result,tseries,header_exp='O0-Pout-blue')
        movie_series = self.interp_columns(result,tseries,regex='Pout',return_df=True)
        unscaled_output = self._unscale_teacher(output_series)
        plot_series = self.interp_columns(result,tseries,return_df=True)
        
        if output_all :
            return tseries, unscaled_output, movie_series, plot_series
        else :
            return tseries, unscaled_output, movie_series
        