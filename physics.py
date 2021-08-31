#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 14:57:06 2021

@author: dwinge
"""
import numpy as np
import pandas as pd

# Parameters for transistor function
m = 1.7214255535965923
Vg_prime=0.132675 # V
Id_Vt = 941.4351939816188 # nA
lin_slope = 44087.273087156165 # nA/V
kT = 0.025851 # V
Isub_Vg_prime = 1961.9088696825586 # nA/V
Vt = 0.1 # V

# Soemthing to think about
# I would probably like an object holding the node physical parameters in
# a better fashion. At the moment there is no way of for example to plot
# the gain-bandwidth product as a function of Rstore. 
# A proper way would be to write an input file holding the parameters that is 
# read in, with the possibility to adjust individual parameters

# def example_gammas(node_type='propagator', Lg=150e-7) :
#     # Gate length as default parameter with units in cm
    
#     #### Old system parameters, now defined from absolute values
#     #Cinh = 200 # nf/cm^2
#     #Rinh = 1e-3 # Ohm cm^2
#     #Cexc = Cinh
#     #Rexc = Rinh
#     #Cstore = 100 # nf/cm^2
#     #Rstore = 1e-2 # Ohm cm^2
#     #RCLED = 0.2 # ns
    
#     Cexc, Cinh, Cgate, Cmem, CLED = example_capacitances()
#     if node_type=='memory' :
#         Rexc, Rinh, Rstore, RLED = example_resistances_memory()
#     else :
#         Rexc, Rinh, Rstore, RLED = example_resistances()
    
#     Cstore = Cmem + Cgate*Lg # Sum the memory and gate capacitance
#     # System frequencies
#     g11 = 1e-9/Cinh/Rinh # ns^-1 # GHz
#     g22 = 1e-9/Cexc/Rexc # 
#     g13 = 1e-9/Cstore/Rinh
#     g23 = 1e-9/Cstore/Rexc
#     g33 = 1e-9/Cstore/Rstore
#     gled = 1e-9/CLED/RLED

#     return np.array([g11,g22,g13,g23,g33,gled])

# def example_resistances() :
    
#     Rinh = 6.667e5 # Ohm
#     Rexc = Rinh 
#     RLED = 6.667e6 # Ohm
#     Rstore = 2.0e6 # Ohm
    
#     return np.array([Rexc, Rinh, Rstore, RLED])
  
# def example_resistances_memory() :
    
#     Rinh = 6.667e5 # Ohm
#     Rexc = Rinh 
#     RLED = 6.667e6 # Ohm
#     Rstore = 2.0e7 # Ohm
    
#     return np.array([Rexc, Rinh, Rstore, RLED])
      

# def example_capacitances() :
    
#     Cgate = 0.06e-15 # F
#     Cgate = 4.41e-12 # F/cm
#     Cinh = 0.3e-15   # F
#     Cexc = Cinh
#     Cstore = 0.06e-15 # F, equal to the gate capacitance
#     CLED = 0.03e-15  # F

#    return np.array([Cexc, Cinh,Cgate,Cstore,CLED])

# def unity_coupling_coefficient(s=1e-3, gammas, capacitances) :
#     # Get the unity couping coefficient from the system
#     sample_s = np.array([s])
#     # Setup the gain function
#     gammas = example_gammas()
#     capacitances = example_capacitances()
#     eigvals = setup_gain(gammas)
#     # Sample the gain function
#     G11_at_s, _ = gain(sample_s,eigvals,gammas,capacitances)

#     return G11_at_s.real[0]**-1

# def gain(s, eigvals, gammas, capacitances, vsat=1e7, Vt_prime=0.) :
#     # This function needs to change if Vt_prime is not zero
#     if Vt_prime > 0. :
#         print('Values of Vt_prime not equal zero not supported')
#         return 0
    
#     # Sort the input
#     g11, g22,g13,g23,g33,gled = gammas
#     Cexc, Cinh, Cgate, Cmem, CLED = capacitances
    
#     # Denoninator of expression (shared between G11, G12)
#     denom = np.ones(len(s),dtype=complex)
#     for l in eigvals :
#         denom *= s-l
#     # The LED dynamics are included in eigvals
#     #denom *= s+gled
    
#     # Constant prefactor (shared between G11, G12)
#     prefactor = gled*vsat*Cgate
#     # gammas are in inverse ns so units have to be adjusted
#     # (in the scope of the total expression)
#     prefactor *= 1e-9 # ns to s
    
#     # Specifying the actual terms
#     G11 = g23*(s+g22)/Cexc * prefactor/denom
#     G12 = -g13*(s+g11)/Cinh * prefactor/denom  
    
#     return G11, G12
    
# def setup_gain (gammas) :
    
#     # System matrix setup
#     def A_mat(g11,g22,g13,g23,g33) :
#         gsum = g13+g23+g33
#         A = np.array([[-g11, 0, g11],
#                       [0, -g22, g22],
#                       [g13, g23, -gsum]])

#         return A


#     # Setup matrix
#     #A = A_mat(g11,g22,g13,g23,g33)
#     A = A_mat(*gammas[:-1])
#     #print('System matrtix A=\n',A)   
#     # Find the eigenvalues
#     v, w = np.linalg.eig(A)

#     # Add our LED system as a pole as well
#     gled = gammas[-1]
#     v_add = np.concatenate((v,np.array([-gled])))
      
#     return v_add

def square_pulse(t, args):
    tmp = 0.
    # Unpack here
    tlims, amplitude = args
    for ttuple in tlims:
        tmp += float((t >= ttuple[0])*(t < ttuple[1]))
    return tmp*amplitude

def constant(t, amplitude) :
    return amplitude


# Write a method that returns a function, need a class somewhere as we might
# need multiple instances of this. PErhaps it returns a class instead. 
# Could be a device class with the set of parameters as specified in the file. 

class Device:
    # Class variable, shared by all instances
    keys = ['Rinh','Rexc','RLED','Rstore','Cinh','Cexc','CLED','Cstore','Cgate','Vt','m','I_Vt','vt','Lg','AB','CB']
    units = ['Ohm']*4 + ['F']*4 + ['F/cm'] + ['V','dim. less','nA','cm/s','um','uA','1/uA']
    kT = 0.02585
    
    def __init__(self, path_to_file) :
        # Read in the specified parameter file
        self.p_dict = {}
        self.p_units = {}
        self.read_parameter_file(path_to_file,self.p_dict, self.p_units)
        # Transistor linear slope in nA/V
        self.linslope=self.p_dict['Cgate']*self.p_dict['vt']*1e9 # nA/V
                                 
    def read_parameter_file(self, path_to_file, p_dict, p_units) :
        # Read file and then loop through and assign to dict
        params = np.loadtxt(path_to_file)
        for k, key in enumerate(self.keys):
            p_dict[key] = params[k]
            p_units[key] = self.units[k]

    def set_parameter(self,key,value) :
        self.p_dict[key]=value
    
    def print_parameter(self, key):
        print(f'The parameter {key}={self.p_dict[key]} {self.p_units[key]}')
              
    # Supply transistor functionality, the method transistorIV can be ported 
    def Id_sub(self,Vg) :
        return self.p_dict['I_Vt']*np.exp((Vg-self.p_dict['Vt'])/self.p_dict['m']/self.kT)

    def Id_sat(self,Vg) :
        return self.p_dict['I_Vt'] + self.linslope*(Vg-self.p_dict['Vt'])

    def transistorIV(self,Vg) :
        # Returns current in nA
        Vt = self.p_dict['Vt']
        return np.piecewise(Vg, [Vg<Vt, Vg>=Vt], [self.Id_sub, self.Id_sat])   
    
    def transistorIV_example (self, Vstart=-0.5, Vend=1.0) :
        # Generate a sample transistor IV
        NV = 200
        
        Vgate = np.linspace(Vstart, Vend, NV)
        I = self.transistorIV(Vgate)*1e-3
        data = np.array([Vgate, I])
        
        df = pd.DataFrame(data.transpose(),
                          columns=['Vgate','Current'])
        
        return df

    # Supply the gammas needed for time stepping
    def gammas(self) :
        # Sum the memory and gate capacitance, convert Lg in um to cm
        Cmem = self.p_dict['Cstore'] + self.p_dict['Cgate']*self.p_dict['Lg']*1e-4 
        # System frequencies
        g11 = 1e-9/self.p_dict['Cinh']/self.p_dict['Rinh'] # ns^-1 # GHz
        g22 = 1e-9/self.p_dict['Cexc']/self.p_dict['Rexc'] # ns^-1 # GHz
        g13 = 1e-9/Cmem/self.p_dict['Rinh'] # ns^-1 # GHz
        g23 = 1e-9/Cmem/self.p_dict['Rexc'] # ns^-1 # GHz
        g33 = 1e-9/Cmem/self.p_dict['Rstore'] # ns^-1 # GHz
        gled = 1e-9/self.p_dict['CLED']/self.p_dict['RLED'] # ns^-1 # GHz

        return np.array([g11,g22,g13,g23,g33,gled])


    # Transfer function calculations
    def setup_gain (self,gammas) :
        
        # System matrix setup
        def A_mat(g11,g22,g13,g23,g33) :
            gsum = g13+g23+g33
            A = np.array([[-g11, 0, g11],
                          [0, -g22, g22],
                          [g13, g23, -gsum]])
    
            return A
    
    
        # Setup matrix
        #A = A_mat(g11,g22,g13,g23,g33)
        A = A_mat(*gammas[:-1])
        #print('System matrtix A=\n',A)   
        # Find the eigenvalues
        v, w = np.linalg.eig(A)
    
        # Add our LED system as a pole as well
        gled = gammas[-1]
        v_add = np.concatenate((v,np.array([-gled])))
          
        return v_add
    
    def gain(self, s, eigvals, gammas, vsat=1e7, Vt_prime=0.) :
        # This function needs to change if Vt_prime is not zero
        if Vt_prime > 0. :
            print('Values of Vt_prime not equal zero not supported')
            return 0
        
        # Sort the input
        g11, g22,g13,g23,g33,gled = gammas
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
    
    def unity_coupling_coefficient(self, eta_handle, s=1e-3) :
        # Get the unity couping coefficient from the system
        sample_s = np.array([s])
        # Setup the gain function with system gammas
        gammas = self.gammas()
        eigvals = self.setup_gain(gammas)
        # Sample the gain function
        G11_at_s, _ = self.gain(sample_s,eigvals,gammas)
        # Add also the LED efficiency
        eta_max = max(self.eta_example(eta_handle)['eta, IQE'])
        print(f'Found max eta of {eta_max}')
    
        return (eta_max*G11_at_s.real[0])**-1
        
    # LED efficiency physics as well
    def ABC(self,I,AB,CB) :
        # ABC model adapted to currents
        eta = I/(AB + I + CB*I**2)
        return eta
        
    def eta_ABC(self,I) :
        # calculate efficiency from ABC model
        # ABC model is in uA, so multiply I by 1e-3
        return self.ABC(I*1e-3,self.p_dict['AB'],self.p_dict['CB']) 
    
    def eta_unity(self, I) :
        #return np.ones_like(np.atleast_1d(I))
        return 1.0
    
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