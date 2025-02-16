#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 14:08:46 2021
Major revision of the matrix algrebra on March 8 2022.

Notation:
    Capital letters A,B,C... denote matrices 
    Capital letters N,M,P... denote total number of something
    
    Internal voltage variables sets V as 3xN with number of nodes N.
    Internal dynamics governed by A of 3x3 so dV/dt = A @ V
    Input curents supplied as B = 3xN from B = WC where C are output currents
    from connecting nodes and W are weights of dim W = N x P
    for mapping from P nodes onto N nodes. 
    with C having dimensions of simply dim P.
    This way we can do all matrix multiplications by for examle
    B = W @ C which equals B(i) = sum(j) W(ij)C(j) 
    
    Each W is contained in a class
    object that keeps track of the communication channel (e.g. wavelength)
    and the connecting layers.

@author: dwinge
"""
import pandas as pd
import numpy as np

# Global variable
NV = 3 # internal voltage degrees of freedom

class Layer :
    keys = ['Rinh','Rexc','RLED','Rstore','Cinh','Cexc','CLED','Cstore','Cgate','Vt','m','I_Vt','vt','Lg','AB','CB']
    units = ['Ohm']*4 + ['F']*4 + ['F/cm'] + ['V','dim. less','nA','cm/s','um','uA','1/uA']
    kT = 0.02585
    """ Base class for the other layers"""
    def __init__(self, N, layer_type, path_to_file):
        self.N = N
        self.layer_type = layer_type
        self.p_dict = {}
        self.p_units = {}
        self.read_parameter_file(path_to_file,self.p_dict, self.p_units)
        # Transistor linear slope in nA/V
        self.linslope=self.p_dict['Cgate']*self.p_dict['vt']*1e9 # nA/V
        self.gammas = self.calc_gammas()
        self.A = self.calc_A()

    def read_parameter_file(self, path_to_file, p_dict, p_units) :
        # Read file and then loop through and assign to dict
        params = np.loadtxt(path_to_file)
        for k, key in enumerate(self.keys):
            p_dict[key] = params[k]
            p_units[key] = self.units[k]
        
    def calc_A(self,Rstore=None,Cstore=None) :
        # Calcualte A for the special case of a distribution of Rstore and Cstore, 
        # or for the object self.gammas
        if (Rstore is not None) and (Cstore is not None) :
            new_gammas = self.calc_gammas(Rstore,Cstore)
            return self.A_mat(*new_gammas[:-1])
        elif Rstore is not None :
            new_gammas = self.calc_gammas(Rstore)
            return self.A_mat(*new_gammas[:-1])
        else :
            return self.A_mat(*self.gammas[:-1])    

    def A_mat(self,g11,g22,g13,g23,g33) :
        gsum = g13+g23+g33
        A = np.array([[-g11, 0, g11],
                      [0, -g22, g22],
                      [g13, g23, -gsum]])

    # Supply the gammas needed for time stepping
    def calc_gammas(self,Rstore=None,Cstore=None) :
        # Sum the memory and gate capacitance, convert Lg in um to cm
        if Cstore is not None:
            Cmem = self.calc_Cmem(Cstore)
        else :
            Cmem = self.calc_Cmem()
        # System frequencies
        g11 = 1e-9/self.p_dict['Cinh']/self.p_dict['Rinh'] # ns^-1 # GHz
        g22 = 1e-9/self.p_dict['Cexc']/self.p_dict['Rexc'] # ns^-1 # GHz
        g13 = 1e-9/Cmem/self.p_dict['Rinh'] # ns^-1 # GHz
        g23 = 1e-9/Cmem/self.p_dict['Rexc'] # ns^-1 # GHz
        if Rstore is not None :
            g33 = 1e-9/Cmem/Rstore
        else :
            g33 = 1e-9/Cmem/self.p_dict['Rstore'] # ns^-1 # GHz
        gled = 1e-9/self.p_dict['CLED']/self.p_dict['RLED'] # ns^-1 # GHz

        return np.array([g11,g22,g13,g23,g33,gled])
    
    def calc_Cmem(self,Cstore=None) :
        # Sum the memory and gate capacitance, convert Lg in um to cm
        if Cstore is None :
            Cmem = self.p_dict['Cstore'] + self.p_dict['Cgate']*self.p_dict['Lg']*1e-4 
        else :
            Cmem = Cstore + self.p_dict['Cgate']*self.p_dict['Lg']*1e-4  
        return Cmem

    # TODO: This does not work perfectly for 2 input layers (first hidden layer 'K')
    def get_node_name(self,node_idx,layer_idx=1) :
        if type(layer_idx) is int :
            # Name using the standard convention
            if self.layer_type=='hidden' :
                # Name hidden layers using sequence (defaults to 'H')
                letters = 'HKLMN'
                letter = letters[layer_idx-1]
            elif self.layer_type=='input' :
                # Name input layers using sequence (defaults to 'I')
                letters = 'IJ'
                letter = letters[layer_idx]
            else :
                # if not hidden or input layer
                letter = self.layer_type[0].upper()
        else :
            # Use the layer key as a name
            letter = layer_idx + '_' # now this is a label instead
                
        idx = str(node_idx)
        return letter+idx
    
    def get_names(self,layer_idx=1) :
        names = []
        for n in range(self.N) :
            names.append(self.get_node_name(n,layer_idx))
        return names
    
    def reset(self) :
        return None
    
    def reset_B(self) :
        return None
    
    def setup_gain (self) :
        # Setup matrix
        A = self.A_mat(*self.gammas[:-1])
        #print('System matrtix A=\n',A)   
        # Find the eigenvalues
        v, w = np.linalg.eig(A)
    
        # Add our LED system as a pole as well
        gled = self.gammas[-1]
        v_add = np.concatenate((v,np.array([-gled])))
          
        return v_add
    
    def gain(self, s, eigvals, vsat=1e7, Vt_prime=0.) :
        eigvals = self.setup_gain()
        # This function needs to change if Vt_prime is not zero
        if Vt_prime > 0. :
            print('Values of Vt_prime not equal zero not supported')
            return 0
        
        # Sort the input
        g11, g22,g13,g23,g33,gled = self.gammas
        # Get the necessary capacitance values
        Cexc = self.p_dict['Cexc'] 
        Cinh = self.p_dict['Cinh'] 
        Cgate = self.p_dict['Cgate']
        
        # Denoninator of expression (shared between G11, G12)
        denom = np.ones(len(s),dtype=complex)
        for l in eigvals :
            denom *= s-l

        # Constant prefactor (shared between G11, G12)
        prefactor = gled*vsat*Cgate
        # gammas are in inverse ns so units have to be adjusted
        # (in the scope of the total expression)
        prefactor *= 1e-9 # ns to s
        
        # Specifying the actual terms
        G11 = g23*(s+g22)/Cexc * prefactor/denom
        G12 = -g13*(s+g11)/Cinh * prefactor/denom  
        
        return G11, G12
    
    def transistorIV_example (self, Vstart=-0.5, Vend=1.0) :
        # Generate a sample transistor IV, generates data in uA
        NV = 200
        
        Vgate = np.linspace(Vstart, Vend, NV)
        I = self.transistorIV(Vgate)*1e-3
        data = np.array([Vgate, I])
        
        df = pd.DataFrame(data.transpose(),
                          columns=['Vgate','Current'])
        
        return df
    
    def eta_example(self, handle) :

        I = 10**np.linspace(-2,2,num=50) # in uA
        eta = handle(I*1e3) # ABC expects nA
        if len(np.atleast_1d(eta)) < len(I) :
            # eta is a single value
            eta = eta*np.ones_like(I)
            
        data = np.array([I, eta])
        
        df = pd.DataFrame(data.transpose(),
                          columns=['Current (uA)','eta, IQE'])
        
        return df
class HiddenLayer(Layer) :
    
    def __init__(self, N, output_channel, inhibition_channel, excitation_channel, 
                 device=None, Vthres=1.2, multiA=False) :
        """
        Constructor for hidden layers.
    
        Parameters
        ----------
        N : integer
            number of nodes.
        output_channel : str or tuple of str
            output channel label.
        inhibition_channel : str or tuple of str
            inibition channel label.
        excitation_channel : str
            excitation channel label.
        device : physics.Device object, optional
            Assign a device to the nodes. The default is None.
        Vthres : float, optional
            Clipping voltage for the exciting/inhibiting voltage. The default is 1.2 V.
    
        Returns
        -------
        None.
    
        """
        
        Layer.__init__(self, N, layer_type='hidden', path_to_file=device)
        # Connections to the outside world
        self.out_channel = output_channel
        # These can be multiple, care taken in the constructing of weights
        self.inh_channel = inhibition_channel
        self.exc_channel = excitation_channel
        self.channel_map = {inhibition_channel:0, excitation_channel:1}

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
        # Sequence of transistor threshold voltages, initialized to None
        self.Vt_vec = None 
        # Device object hold A, for example
        self.multiA=multiA
        self.Bscale=np.diag([1e-18/self.p_dict['Cinh'],
                             1e-18/self.p_dict['Cexc'],
                             0.])

    def inverse_gain_coefficient(self):
        # Remember nA
        Rsum=self.p_dict['Rstore']+self.p_dict['Rexc']
        max_Vg = self.Vthres*self.p_dict['Rstore']/Rsum
        Iexc = self.Vthres/Rsum*1e9 # nA
        Isd = self.transistorIV(max_Vg) 
        Iout = self.eta_ABC(Isd)*Isd
        return Iexc/Iout, Iexc

    
    def read_parameter_file(self, path_to_file, p_dict, p_units) :
        # Read file and then loop through and assign to dict
        params = np.loadtxt(path_to_file)
        for k, key in enumerate(self.keys):
            p_dict[key] = params[k]
            p_units[key] = self.units[k]
        
    def A_mat(self,g11,g22,g13,g23,g33) :
        gsum = g13+g23+g33
        A = np.array([[-g11, 0, g11],
                      [0, -g22, g22],
                      [g13, g23, -gsum]])
    
        return A
    
    def calc_Cmem(self,Cstore=None) :
        # Sum the memory and gate capacitance, convert Lg in um to cm
        if Cstore is None :
            Cmem = self.p_dict['Cstore'] + self.p_dict['Cgate']*self.p_dict['Lg']*1e-4 
        else :
            Cmem = Cstore + self.p_dict['Cgate']*self.p_dict['Lg']*1e-4  
        return Cmem

    # Supply the gammas needed for time stepping
    def calc_gammas(self,Rstore=None,Cstore=None) :
        # Sum the memory and gate capacitance, convert Lg in um to cm
        if Cstore is not None:
            Cmem = self.calc_Cmem(Cstore)
        else :
            Cmem = self.calc_Cmem()
        # System frequencies
        g11 = 1e-9/self.p_dict['Cinh']/self.p_dict['Rinh'] # ns^-1 # GHz
        g22 = 1e-9/self.p_dict['Cexc']/self.p_dict['Rexc'] # ns^-1 # GHz
        g13 = 1e-9/Cmem/self.p_dict['Rinh'] # ns^-1 # GHz
        g23 = 1e-9/Cmem/self.p_dict['Rexc'] # ns^-1 # GHz
        if Rstore is not None :
            g33 = 1e-9/Cmem/Rstore
        else :
            g33 = 1e-9/Cmem/self.p_dict['Rstore'] # ns^-1 # GHz
        gled = 1e-9/self.p_dict['CLED']/self.p_dict['RLED'] # ns^-1 # GHz

        return np.array([g11,g22,g13,g23,g33,gled])


    def calc_A(self,Rstore=None,Cstore=None):
        # Calcualte A for the special case of a distribution of Rstore and Cstore, 
        # or for the object self.gammas
        if (Rstore is not None) and (Cstore is not None) :
            new_gammas = self.calc_gammas(Rstore,Cstore)
            return self.A_mat(*new_gammas[:-1])
        elif Rstore is not None :
            new_gammas = self.calc_gammas(Rstore)
            return self.A_mat(*new_gammas[:-1])
        else :
            return self.A_mat(*self.gammas[:-1])    

    def generate_uniform_Adist(self, scale) :
        """Here we do another variation of the memory constants"""
        if self.p_dict is None :
            print("Please first assign a device before generating Adist")
        else :
            A = np.zeros((self.N,3,3))
            R_ref = self.p_dict['Rstore']
            C_ref = self.p_dict['Cstore']
            rng = np.random.RandomState()
            scale_RC_dist = np.sqrt(rng.uniform(1.0,scale**2,size=self.N))
            for k in range(0,self.N) :
                Rstore = R_ref*scale_RC_dist[k]
                Cstore = C_ref*scale_RC_dist[k]
                A[k] = self.calc_A(Rstore, Cstore)
            self.Adist = A
    
    def generate_exp_Adist(self, mean) :
        if self.p_dict is None :
            print("Please first assign a device before generating Adist")
        else :
            A = np.zeros((self.N,3,3))
            R_ref = self.p_dict['Rstore']
            C_ref = self.p_dict['Cstore']
            rng = np.random.RandomState()
            scale_RC_dist = np.sqrt(rng.exponential(scale=mean,size=self.N))
            #scale_RC_dist = np.sqrt(rng.uniform(1.0,scale**2,size=self.N))
            for k in range(0,self.N) :
                Rstore = R_ref*(1+scale_RC_dist[k])
                Cstore = C_ref*(1+scale_RC_dist[k])
                A[k] = self.calc_A(Rstore, Cstore)
            self.Adist = A
            
    def generate_poisson_Adist(self, mean) :
        if self.p_dict is None :
            print("Please first assign a device before generating Adist")
        else :
            A = np.zeros((self.N,3,3))
            R_ref = self.p_dict['Rstore']
            C_ref = self.p_dict['Cstore']
            rng = np.random.RandomState()
            scale_RC_dist = np.sqrt(rng.poisson(scale=mean,size=self.N))
            #scale_RC_dist = np.sqrt(rng.uniform(1.0,scale**2,size=self.N))
            for k in range(0,self.N) :
                Rstore = R_ref*(1+scale_RC_dist[k])
                Cstore = C_ref*(1+scale_RC_dist[k])
                A[k] = self.calc_A(Rstore, Cstore)
            self.Adist = A
                
    def generate_Adist(self,noise=0.1,p_label='Rstore') :
        """At the moment we do only variance in Rstore."""
        if self.p_dict is None :
            print("Please first assign a device before generating Adist")
        else :
            A = np.zeros((self.N,3,3))
            p_ref = self.p_dict[p_label] 
            rng = np.random.RandomState()
            Rstore_dist = rng.normal(loc=p_ref, scale=noise*p_ref,size=self.N)
            # Clip to avoid negative values
            Rstore_dist = np.clip(Rstore_dist, p_ref*0.01,np.inf)
            for k in range(0,self.N) :
                A[k] = self.calc_A(Rstore_dist[k])
            self.Adist = A
        
    def specify_Vt(self,Vts) :
        self.Vt_vec = Vts
        
    def get_dV(self, t) :     
        """ Calculate the time derivative."""
        if not self.multiA :
            self.dV = self.A @ self.V + self.Bscale @ self.B
        else :
            self.dV = np.einsum('jik,kj->ij',self.Adist,self.V) + self.Bscale @ self.B
            
        return self.dV
        
    def update_V(self, dt) :
        """ Using a fixed dt, update the voltages."""
        self.V += dt*self.dV
        # Count voltage overshoots
        overshoots = (self.V<-self.Vthres)*(self.V>self.Vthres)
        N = np.sum(overshoots)
        # Voltage clipping (third try)
        self.V = np.clip(self.V,-self.Vthres,self.Vthres)
        return N
    
    def eta_ABC(self,I) :
        # calculate efficiency from ABC model
        # ABC model is in uA, so multiply I by 1e-3
        return self.ABC(I*1e-3,self.p_dict['AB'],self.p_dict['CB']) 
    
    def ABC(self,I,AB,CB) :
        # ABC model adapted to currents
        eta = I/(AB + I + CB*I**2)
        return eta
    
    # Supply transistor functionality, the method transistorIV can be ported 
    def Id_sub(self,Vg,Vt,mask) :
        return self.p_dict['I_Vt']*np.exp((Vg-Vt[mask])/self.p_dict['m']/self.kT)

    def Id_sat(self,Vg,Vt,mask) :
        # Invert the mask for this one
        return self.p_dict['I_Vt'] + self.linslope*(Vg-Vt[mask==False])
    
    def Id_sub_0(self,Vg,Vt) :
        return self.p_dict['I_Vt']*np.exp((Vg-Vt)/self.p_dict['m']/self.kT)

    def Id_sat_0(self,Vg,Vt) :
        return self.p_dict['I_Vt'] + self.linslope*(Vg-Vt)

    def transistorIV(self,Vg,Vt_vec=None) :
        """Reads gate voltage and calculated transistor current based on 
        parameters in p_dict. Can take a vector of individual threshold currents 
        to introduce fluctuations in the system. Returns current in nA."""

        if Vt_vec is None :
            Vt = self.p_dict['Vt'] 
            return np.piecewise(Vg, [Vg<Vt, Vg>=Vt], [self.Id_sub_0, self.Id_sat_0],Vt) 
        else :
            Vt = Vt_vec
            # This should work even when Vt is an array
            return np.piecewise(Vg, [Vg<Vt, Vg>=Vt], [self.Id_sub, self.Id_sat],Vt,Vg<Vt)   
    

    def update_I(self, dt) :
        """ Using a fixed dt, update the voltages."""
        # Get the source drain current from the transistor IV
        self.ISD = self.transistorIV(self.V[2],self.Vt_vec)
        self.I += dt*self.gammas[-1]*(self.ISD-self.I)
        # Convert current to power through efficiency function
        self.P = self.I*self.eta_ABC(self.I)
    
    def reset_B(self) :
        """ Set elements of matrix B to 0"""
        self.B[:,:] = 0
    
    def reset_I(self):
        """ Set the currents to 0."""
        self.I[:] = 0
        
    def reset_V(self):
        """ Set all voltages to 0"""
        self.V = np.zeros((NV,self.N))
    
    # Here we need an input mask to choose the communciation channels
    def update_B(self, weights, C) :
        """
        Incremental update to the matrix B that handles all the inputs. 

        Parameters
        ----------
        weights : network.Weights object
            The connecting weights.
        C : numpy ndarray
            Output values from the connected nodes.

        Returns
        -------
        None.

        """        
        if weights.channel==self.inh_channel :
            # Make sure we get the correct sign
            self.B[self.channel_map[weights.channel]] -= weights.W @ C
        else :
            self.B[self.channel_map[weights.channel]] += weights.W @ C
                
        
        #E = weights.E
        #W = weights.W
        #BM = np.einsum('ijk,ik->ij',W,C)
        ## B is scaled at a later point in get_dV
        #self.B += E @ BM 
        
    def reset(self) :
        """ Reset layer values."""
        self.reset_B()
        self.reset_I()
        self.reset_V()
    
# Inherits Layer
class InputLayer(Layer) :
        
    def __init__(self, N, path_to_file):
        """
        Constructor for an input layer. Input data set via set_input_func
        method.

        Parameters
        ----------
        N : int
            number of nodes of the input layer

        Returns
        -------
        None.

        """
        Layer.__init__(self, N, layer_type='input', path_to_file=path_to_file)
        
        #self.channels = input_channels
        #self.M = len(input_channels)
        #self.start_idx = [int(sum(self.node_structure[:k])) for k in range(self.M+1)]
        self.C = np.zeros((self.N))
        # These dictonaries hold function handleks and arguments
        self.input_func_handles={}
        self.input_func_args={}

    def set_input_vector_func(self, func_handle, func_args=None) :
        """
        Assign a continous function handle to get the input current at time t
        For an input layer of N>1 this function will have to be vector valued.

        Parameters
        ----------
        func_handle : handle
            function handle to provide continous signal.
        func_args : tuple, optional
            Additional function arguments. Enter single arguments as single 
            tuple (a,). The default is None.

        Returns
        -------
        None.

        """
        self.input_func_handles['v']= func_handle
        self.input_func_args['v'] = func_args
        
    def set_input_func_per_node(self,node,func_handle, func_args=None):
        
        self.input_func_handles[node] = func_handle
        self.input_func_args[node] = func_args
        
    def update_C(self, t) :
        """ Using the specified handles, give the input current at time t."""
        # Work on class object self.I instead of an arbitrary thing
        if 'v' in self.input_func_handles:
            # In this case, the input values are set by a vector valued function
            try :
                if self.input_func_args['v'] is not None:
                    self.C = self.input_func_handles['v'](t,*self.input_func_args['v'])
                else :
                    self.C = self.input_func_handles['v'](t)
            except :
                pass
            # Fix single node issues
            self.C = np.atleast_1d(self.C)
            
        else :
            # Otherwise we loop through the nodes to search for functions
            for key in self.input_func_handles :
                try :
                    if self.input_func_args[key] is not None:
                        self.C[key] = self.input_func_handles[key](t,*self.input_func_args[key])
                    else :
                        self.C[key] = self.input_func_handles[key](t)
                except :
                    pass
            
        return self.C
    
    
# Inherits Layer    
class OutputLayer(Layer) :
    
    def __init__(self, N, teacher_delay=None, nsave=800, path_to_file='') :
        """
        Constructor for an input layer. 
        
        """
        Layer.__init__(self, N, layer_type='output', path_to_file=path_to_file)
        #self.channels = output_channels
        # TODO: Perhaps self.I is not needed anymore, compare input layer
        self.C = np.zeros(self.N)
        self.B = np.zeros_like(self.C)
        self.I = np.zeros(self.N)
        # These dictonaries hold function handles and arguments
        self.output_func_handles={}
        self.output_func_args={}
        self.teacher_delay=teacher_delay
        self.nsave=nsave

    def set_teacher_delay(self,delay,nsave=None) :
        """
        Configure the saving of previous points at the output 

        Parameters
        ----------
        delay : float
            Delay in units of ns.
        nsave : int, optional
            Change number of time steps stored in memory. The default is None.

        Returns
        -------
        None.

        """
        # introduce a class variable to handle time delays
        self.teacher_delay=delay
        # Give possibility to override nsave here
        if nsave is not None:
            self.nsave=nsave
        self.reset_teacher_memory(self.nsave)
        
    def reset_teacher_memory(self,nsave) :
        """ Reset the memory."""
        # create a np array of fixed size to handle the memory structure
        self.teacher_memory=np.zeros((nsave,len(self.B.flatten())+1))
    
    def set_output_func(self, func_handle, func_args=None) :
        """
        Assign a continous function handle to get the input current at time t
        For N>1 this will have to be vector valued.

        Parameters
        ----------
        func_handle : handle
            function handle to provide continous signal.
        func_args : tuple, optional
            Additional function arguments. Enter single arguments as single 
            tuple (a,). The default is None.

        Returns
        -------
        None.

        """
        self.output_func_handles = func_handle
        self.output_func_args = func_args
        
    def get_output_current(self, t) :
        """ Using the specified handles, give the input current at time t."""
        # Work on class object self.I instead of an arbitrary thing
        try :
            if self.output_func_args is not None:
                self.I = self.output_func_handles(t,*self.output_func_args)
            else :
                self.I = self.output_func_handles(t)
        except :
            pass
        
        return self.I
    
    def update_B (self, weights, C) :
        """ Update B using the weights and input."""
        # B is automatically allocated using this procedure
        W = weights.W
        # This automatically yields current in nA
        self.B += W @ C
    
    def reset_B(self) :
        """ Set the B matrix to 0."""
        # This construction hopefully avoids reallocating memory
        try :
            self.B[:] = 0
        except :
            self.B = 0
            
    def lin_intp(self,x,f) :
        """ Helper function to interpolate data points."""
        # specify interval, xs is the endpoints
        x0 = f[0,0]
        x1 = f[1,0]
        p1 = (x-x0)/(x1-x0) # fraction to take of index 1
        p0 = 1-p1 # fraction to take from index 1
        # Appy these to the deque object
        return p0*f[0]+p1*f[1]
            
    def update_C_from_B(self,t,t0=0) :
        """
        Generates the output matrix C from the history of activations of this
        layer. Can start at a different t0 than 0.

        Parameters
        ----------
        t : float
            Time where output is calculated.
        t0 : float, optional
            Starting time for the saved history. The default is 0.

        Returns
        -------
        None.

        """
        # Here it is possible to add a nonlinear function as well
        # Need to take the delay into account here
        if self.teacher_delay is not None :
            # First, we update the memory with correct time-stamp B
            # Shift memory "upwards"
            self.teacher_memory[:self.nsave-1]=self.teacher_memory[1:]
            # Write new value
            self.teacher_memory[-1]=np.concatenate((np.array([t]),self.B.flatten()))
            # Now we choose the correct memory point to choose from
            t_find = max(t-self.teacher_delay,t0) # starts at t0
            # Search for point just before t, start from end of the deque
            idx_t=-1 
            while t_find < self.teacher_memory[idx_t,0] :
                idx_t -= 1
            
            if t_find > t0 :
                # Catching an exception here when idx_t=-2
                end = idx_t+2 if idx_t+2<0 else None
                teacher_signal = self.lin_intp(t_find,
                                               self.teacher_memory[idx_t:end])
            else : 
                # last entry, think this is ok
                teacher_signal = self.teacher_memory[idx_t]
            
            B_for_update = teacher_signal[1:].reshape(self.N)
        else :
            B_for_update = self.B
            
        self.C = np.copy(B_for_update)
        
    def update_C(self,t) :
        """ Cast the input of the channels as a matrix."""
        self.C = self.get_output_current(t)
        
    def reset(self) :
        """ Reset function."""
        self.reset_B()
        self.reset_teacher_memory(self.nsave)
        
# Connect layers and create a weight matrix
def connect_layers(down, up, layers, channel) :
    """
    Function to connect two layers.

    Parameters
    ----------
    down : int
        from layer index.
    up : TYPE
        to layer index.
    layers : dict
        dictionary containing the layer keys and indicies.
    channel : str
        channel label

    Returns
    -------
    class object
        Weight object holding all relevant matrices and couplings.

    """
    class Weights :
        
        # TODO: How about an connect_all keyword here to avoid boilerplate
        def __init__(self, down, up, layers, channel, connect_all=False) :
            """ Contructor function for the Weight class. """
            # Should not assume correct ordering here
            self.from_layer = down
            self.to_layer = up
            self.channel = channel
            #self.M = len(self.channels)
            
            # Initialize matrices
            L0 = layers[down]
            L1 = layers[up]
            self.W = np.zeros((L1.N,L0.N),dtype=float)
            if connect_all :
                self.W[:,:]=1.0
                       
        # Define an explicit connection
        def connect_nodes(self, from_node, to_node, weight=1.0)  : 
            """
            Node specific method to set node to node weights.

            Parameters
            ----------
            from_node : int
                Node index.
            to_node : int
                Node index.
            weight : float, optional
                Scaling of the weight. The default is 1.0.

            Returns
            -------
            None

            """
            self.W[to_node,from_node] = weight
                        
        def get_edges(self) :
            """ Returns a dict of all internode connections. Each channel is 
            represented by a key."""
            edges = {}
            edge_list = []

            for down in range(len(self.W[0,:])) : # column loop (from)
                for up in range(len(self.W[:,0])) : # row loop (to)
                    weight = self.W[up,down]
                    if weight > 0. :
                        edge_list.append((down,up,weight))
            edges[self.channel] = edge_list
                    
            # Returns a dictionary over the edges of each channel
            return edges
        
        def print_W(self, *args):
            """ Print the weights."""
            def print_W(key,W) :
                with np.printoptions(precision=4, suppress=True):
                    print('{0}:\n{1}'.format(key,W))
                    
            print_W(self.channel,self.W)
                      
        def set_W(self, W) :
            """ Set weight matrix manually."""
            self.W = W
            
        def scale_W(self, scale) :
            """ Scale weight matrix manually."""
            self.W = self.W * scale
            
        def ask_W(self,silent=False) :
            """ Print the shape of the weight matrix.""" 
            if not silent : print(f'Set weights by set_W(array of size M x N \nwith {self.W[0,:,:].shape}')
            return self.W.shape
        
    return Weights(down,up,layers,channel)
    
def reset(layers) :
    """ Resets all layer values to 0."""
    for key in layers :
        # all layer types have this function
        layers[key].reset()

    
