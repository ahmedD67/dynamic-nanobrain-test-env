# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

import networkx as nx
import matplotlib.pyplot as plt
import time

# modules specific to this project
from context import network as nw
from context import physics
from context import timemarching as tm
from context import plotter
from context import logger
import esn


Nreservoir = 40
SEED=44
# Get me a network
my_esn = esn.EchoStateNetwork(Nreservoir,seed=SEED)

# %%
# Specify a standard device for the hidden layers
propagator = physics.Device('device_parameters.txt')
my_esn.assign_device(propagator)


# %% [markdown] Setup the input/output for training
# ### Input and outputs to train the network
# Both input and output need to be supplied to the network in order to train it. 
# These are generated as a random sequence of frequecies.
import numpy as np
rng = np.random.RandomState(SEED)

def freqeuncy_step_generator(tend,fmin,fmax,dT,res=10) :
    # determine the size of the sequence
    dt = fmax**-1/res
    N = int(tend/dt)+1 # steps of total time interval
    dN = int(dT/dt) # steps of average period
    # From the info above we can setup our intervals
    n_changepoints = int(N/dN)
    changepoints = np.insert(np.sort(rng.randint(0,N,n_changepoints)),[0,n_changepoints],[0,N])
    # From here on we use the pyESN example code, with some modifications
    const_intervals = list(zip(changepoints,np.roll(changepoints,-1)))[:-1]
    frequency_control = np.zeros((N,1))
    for k, (t0,t1) in enumerate(const_intervals): # enumerate here
        frequency_control[t0:t1] = fmin + (fmax-fmin)* (k % 2)
    # run time update through a sine, while changing the freqeuncy
    frequency_output = np.zeros((N,1))
    z = 0
    for i in range(N):
        z = z + frequency_control[i]*dt
        frequency_output[i] = (np.sin(z) + 1)/2
        
    tseries = np.arange(0,tend,step=dt)
    
    return np.hstack([np.ones((N,1)),frequency_control]),frequency_output,tseries
    
T = 1000 # ns
dT = 50 # ns, average period length of constant frequency
fmin = 0.1 # GHz
fmax = 0.3 # GHz

frequency_input, frequency_output, tseries = freqeuncy_step_generator(T,fmin,fmax,dT) 

print(f'Generated a time series from 0 to {T} ns with {len(tseries)} elements')

# Now we use interpolation to get function handles from these data
from scipy.interpolate import interp1d 
# Everything is scaled by Imax
def teacher_signal(signal_scale) :
    handle = interp1d(tseries,frequency_output*signal_scale,axis=0)
    return handle 

def input_signal(signal_scale) :
    handle = interp1d(tseries,frequency_input[:,1]*signal_scale,axis=0)  
    return handle

def bias_signal(signal_scale) :
    return lambda t : signal_scale

# %% 
# Plot the frequency control and periods together

if True :
    Nmax = 2999
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    
    ax1.plot(tseries[:Nmax],frequency_input[:Nmax,1])
    ax2.plot(tseries[:Nmax],frequency_output[:Nmax])
    #ax3.plot(frequency_output[:1000])
    #ax2.plot(periods[:1000])
    
    plt.show


# %% 
# Specify spectral radius, scaling of input,bias and teacher signal
my_esn.specify_network(1.0,2.0,0.2,1.0)
teacher_scaling=1.0
# Specify device
my_esn.assign_device(propagator)
# Specify explicit signals by handle
my_esn.specify_inputs(input_signal,bias_signal,teacher_signal)

#%% Set the system delay time

my_esn.set_delay(0.5) # units of ns

# %% [markdown]
# ### 6. Evolve in time
# Fit the ESN using teacher forcing for the first part of the time series
# %%
Tfit = 100.
tseries_train, states_train, teacher_train = my_esn.harvest_states(Tfit)

#%% Do the actual fit as a separate step
pred_train, error = my_esn.fit(states_train, teacher_train,beta=10)

# %%
# Setup a copy of the teacher signal here
teacher_handle = teacher_signal(my_esn.Imax*teacher_scaling)
#teacher_train = teacher_handle(tseries_train)

# Plot training example
fig, ax = plt.subplots()

ax.plot(tseries_train[:],pred_train[:])
ax.plot(tseries_train,teacher_train,'--')
ax.plot()

plt.show()

# %%
scl = 2.0
tseries_test, pred_test = my_esn.predict(Tfit,2.0*Tfit)


# %%
teacher_handle = teacher_signal(my_esn.Imax*teacher_scaling)
teacher_test = teacher_handle(tseries_test)

fig, ax = plt.subplots()
ax.plot(tseries_test[:],pred_test[:])
ax.plot(tseries_test,teacher_test,'--')


plt.show()

# %% Valdidation error 

pred_error = np.sqrt(np.mean((pred_train - teacher_test)**2))/my_esn.Imax

