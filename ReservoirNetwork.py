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
import network as nw
import physics
import timemarching as tm
import plotter
import logger


# %% [markdown]
# ### 1. Define the broadcasting channels of the network
# This is done by creating a list of the channel names. The names are arbitrary and can be set by the user, such as 'postive', 'negative' or explicit wavelenghts like '870 nm', '700 nm'. Here I chose the colors 'red' and 'blue', as well as bias with same wavelength as 'blue'.

# %%
channel_list = ['red', 'blue','green']
# Automatically generate the object that handles them
channels = {channel_list[v] : v for v in range(len(channel_list))}

# %% [markdown]
# ### 2. Define the layers
# Define the layers of nodes in terms of how they are connected to the channels. Layers and weights are organized in dictionaries. The input and output layers do not need to be changed, but for the hidden layer we need to specify the number of nodes N and assign the correct channels to the input/output of the node.

# %%
# Create layers ordered from 0 to P organized in a dictionary
layers = {} 
Nreservoir = 20
# An input layer automatically creates on node for each channel that we define
layers[0] = nw.InputLayer(input_channels=channels)
# Forward signal layer
layers[1] = nw.HiddenLayer(Nreservoir//2, output_channel='blue',excitation_channel='blue',inhibition_channel='red')
# Inhibiting memory layer
layers[2] = nw.HiddenLayer(Nreservoir//2, output_channel='red' ,excitation_channel='blue',inhibition_channel='red')
layers[3] = nw.OutputLayer(output_channels=channels) # similar to input layer

# %% [markdown]
# ### 3. Define existing connections between layers
# The weights are set in two steps. 
# First the connetions between layers are defined. This should be done using the keys defined for each layer above, i.e. 0, 1, 2 ... for input, hidden and output layers, respectively. The `connect_layers` function returns a weight matrix object that we store under a chosen key, for example `'inp->hid'`.
# Second, the specific connections on the node-to-node level are specified using the node index in each layer

# %%
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
# Connections to output
weights['hd0->out'] = nw.connect_layers(1, 3, layers, channels)
weights['hd1->out'] = nw.connect_layers(2, 3, layers, channels)
# Connections back into reservoir from output
weights['out->hd0'] = nw.connect_layers(3, 1, layers, channels)
weights['out->hd1'] = nw.connect_layers(3, 2, layers, channels)

# %% [markdown]
# Setup parameters for the network

# %%
sparsity = 0.90
spectral_radius = 1.0
# One number per channel, blue, red, green
input_scaling = [1.0,1.0,0.1]
output_scaling= [1.0,1.0,1.0]
teacher_scaling = [1.0,1.0,1.0]
# Already specified above Nreservoir = 100

# %% [markdown]
# #### Setup the input weights

# %%
# We will generate some random numbers
import numpy as np
rng = np.random.RandomState(42)

# Input weights to all of the input units
W_in = rng.rand(Nreservoir, len(channels))

# Ask the weights object which dimensions to be in
weights['inp->hd0'].ask_W()
# Put each weight column in a specific channel
for key in channels :
    k = channels[key]
    W_key = np.zeros_like(W_in)
    W_key[:,k] = W_in[:,k]
    weights['inp->hd0'].set_W(key,input_scaling[k]*W_key[:Nreservoir//2]) # first half
    weights['inp->hd1'].set_W(key,input_scaling[k]*W_key[Nreservoir//2:]) # second half

# %% [markdown]
# #### Setup the reservoir weights

# %%
W_partition = {'hd0->hd0':(0,Nreservoir//2,0,Nreservoir//2),
               'hd0->hd1':(Nreservoir//2,Nreservoir,0,Nreservoir//2),
               'hd1->hd1':(Nreservoir//2,Nreservoir,Nreservoir//2,Nreservoir),
               'hd1->hd0':(0,Nreservoir//2,Nreservoir//2,Nreservoir)}

# Generate a large matrix of values for each reservoir channel (red and blue)
W_res = rng.rand(2, Nreservoir, Nreservoir)

# Delete the fraction of connections given by sparsity:
W_res[rng.rand(*W_res.shape) < sparsity] = 0
# Delete any remaining diagonal elements
for k in range(0,Nreservoir) :
    W_res[:,k,k] = 0.
    
# Normalize this to have the chosen spectral radius, once per channel
for k in range(0,2) :
    radius = np.max(np.abs(np.linalg.eigvals(W_res[k])))
    # rescale them to reach the requested spectral radius:
    W_res[k] = W_res[k] * (spectral_radius / radius)
    
weights['hd0->hd1'].ask_W()

for connection in W_partition :
    for key in list(channels.keys())[:2] :
        k=channels[key]
        A,B,C,D = W_partition[connection]
        weights[connection].set_W(key,W_res[k,A:B,C:D])



# %% [markdown]
# #### Setup the output weights

# %%
# Output weights from reservoir to the output units
W_out = rng.rand(len(channels),Nreservoir)

# Ask the weights object which dimensions to be in
weights['hd0->out'].ask_W()

# Put each weight column in a specific channel
for key in channels :
    # Greem is the bias and not forwarded to output
    if key != 'green' :
        k = channels[key]
        W_key = np.zeros_like(W_out)
        W_key[k] = W_out[k]
        weights['hd0->out'].set_W(key,output_scaling[k]*W_key[:,:Nreservoir//2])
        weights['hd1->out'].set_W(key,output_scaling[k]*W_key[:,Nreservoir//2:])
        
# Print to make sure these are in matrix form
weights['hd0->out'].print_W()

# %% 
# Input weights from output nodes back into reservoir 
# These are random. The weights from the extended network state are trained.
W_inout = rng.rand(Nreservoir,len(channels))

# Ask the weights object which dimensions to be in
weights['out->hd0'].ask_W()

# Put each weight column in a specific channel
for key in channels :
    # Greem is the bias and not forwarded to output
    if key != 'green' :
        k = channels[key]
        W_key = np.zeros_like(W_inout)
        # Pick a column
        W_key[:,k] = W_inout[:,k]
        weights['out->hd0'].set_W(key,teacher_scaling[k]*W_key[:Nreservoir//2,:])
        weights['out->hd1'].set_W(key,teacher_scaling[k]*W_key[Nreservoir//2:,:])
        
# Print to make sure it looks ok
weights['out->hd0'].print_W()
weights['out->hd1'].print_W()

# %% [markdown]
# ### 4. Visualize the network 
# The `plotter` module supplies functions to visualize the network structure. The nodes are named by the layer type (Input, Hidden or Output) and the index. To supress the printing of weight values on each connection, please supply `show_edge_labels=False`.
#
# #### Available layouts:
# **multipartite**: Standard neural network appearance. Hard to see recurrent couplings within layers.  
# **circular**: Nodes drawn as a circle  
# **shell**: Layers drawn as concetric circles  
# **kamada_kawai**: Optimization to minimize weighted internode distance in graph  
# **spring**: Spring layout which is standard in `networkx` 
#
# #### Shell layout
# This is my current favorite. It is configured to plot the input and output nodes on the outside of the hidden layer circle, in a combined outer concentric circle.

# %%
plotter.visualize_network(layers, weights, exclude_nodes={3:['O2']},node_size=100,layout='shell', show_edge_labels=False)

# %% [markdown]
# ### 5. Specify the physics of the nodes
# Before running any simulations, we need to specify the physics of the hidden layer nodes. Parameters can either be specified directly or coupled from the `physics` module. 

# %%
# Specify a standard device for the hidden layer
propagator = physics.Device('device_parameters.txt')

# %%
# Specify the internal dynamics by supplying the RC constants to the hidden layer (six parameters)
layers[1].assign_device(propagator)
layers[2].assign_device(propagator)
# Tweak the threshold voltage
Vthres=0.5
layers[1].Vthres=Vthres
layers[2].Vthres=Vthres

# Calculate the unity_coeff to scale the weights accordingly
unity_coeff, Imax = propagator.inverse_gain_coefficient(propagator.eta_ABC, Vthres)
print(f'Unity coupling coefficient calculated as unity_coeff={unity_coeff:.4f}')

# %% [markdown] Setup the input/output for training
# ### Input and outputs to train the network
# Both input and output need to be supplied to the network in order to train it. 
# These are generated as a random sequence of frequecies.

def freqeuncy_step_generator(tend,fmin,fmax,dT,res=10) :
    # determine the size of the sequence
    dt = fmax**-1/res
    N = int(tend/dt) # steps of total time interval
    dN = int(dT/dt) # steps of average period
    # From the info above we can setup our intervals
    n_changepoints = int(N/dN)
    changepoints = np.insert(np.sort(rng.randint(0,N,n_changepoints)),[0,n_changepoints],[0,N])
    # From here on we use the pyESN example code, with some modifications
    const_intervals = list(zip(changepoints,np.roll(changepoints,-1)))[:-1]
    frequency_control = np.zeros((N,1))
    for k, (t0,t1) in enumerate(const_intervals): # enumerate here
        frequency_control[t0:t1] = fmin + (fmax-fmin)*rng.rand() 
    # run time update through a sine, while changing the freqeuncy
    frequency_output = np.zeros((N,1))
    z = 0
    for i in range(N):
        z = z + frequency_control[i]*dt
        frequency_output[i] = (np.sin(z) + 1)/2
        
    tseries = np.arange(0,tend,step=dt)
    
    return np.hstack([np.ones((N,1)),frequency_control]),frequency_output,tseries
    
T = 1000 # ns
dT = 20 # ns, average period length of constant frequency
fmin = 0.1 # GHz
fmax = 1.6 # GHz

frequency_input, frequency_output, tseries = freqeuncy_step_generator(T,fmin,fmax,dT) 

# Now we use interpolation to get a function handle from these data
from scipy.interpolate import interp1d 
# Everything is scaled by Imax
signal_scale = Imax/2.
teacher_signal = interp1d(tseries,frequency_output*signal_scale,axis=0)
input_signal = interp1d(tseries,frequency_input[:,1]*signal_scale,axis=0)
bias_signal = lambda t : signal_scale

# %% 
# Plot the frequency control and periods together

fig, (ax1, ax2, ax3) = plt.subplots(1,3)

ax1.plot(tseries[:1000],frequency_input[:1000,1])
ax2.plot(tseries[:1000],frequency_output[:1000])
#ax3.plot(frequency_output[:1000])
#ax2.plot(periods[:1000])

plt.show
# %% 
# Plot the frequency control and periods together

fig, (ax1, ax2, ax3) = plt.subplots(1,3)

ax1.plot(tseries,frequency_input[:,1])
ax2.plot(tseries,frequency_output[:])
#ax3.plot(frequency_output[:1000])
#ax2.plot(periods[:1000])

plt.show

# %% [markdown]
# Here we specify the external signals, bias, input and teacher signal

# %%
# Input layer signal
use_random = False
if (use_random) :
    layers[0].set_input_func(channel='blue',func_handle=input_signal)
    layers[0].set_input_func(channel='green',func_handle=bias_signal)
    layers[3].set_output_func(channel='blue',func_handle=teacher_signal)
# We don't put the teacher's signal in the inhibition channel just yet

# Play around with standard signals
t_blue = [(5.0,6.0), (8.0,8.5), (9.0,9,5), (10.0,11.0), (20.0,21.0), (30.0,31.0)] # 
#t_blue = [(5.0,6.0), (8.0,8.5), (9.0,9,5), (10.0,11.0)] # 

# Scale
signal_scale = 20*Imax
bias_signal = lambda t : signal_scale

# Use the square pulse function and specify which node in the input layer gets which pulse
layers[0].set_input_func(channel='blue',func_handle=physics.square_pulse, func_args=(t_blue, 3.0*Imax))
layers[0].set_input_func(channel='green',func_handle=bias_signal)
#layers[0].set_input_func(channel='red',func_handle=physics.square_pulse, func_args=(t_red, 3.0*Imax))

# %% [markdown]
# ### 6. Evolve in time

# %%
# Start time t, end time T
t = 0.0
T = 100.0 # ns
# To sample result over a fixed time-step, use savetime
savestep = 0.1
savetime = savestep
# These parameters are used to determine an appropriate time step each update
dtmax = 0.1 # ns 
dVmax = 0.005 # V

nw.reset(layers)
# Create a log over the dynamic data
time_log = logger.Logger(layers,channels) # might need some flags

start = time.time()

while t < T:
    # evolve by calculating derivatives, provides dt
    dt = tm.evolve(t, layers, dVmax, dtmax )

    # update with explicit Euler using dt
    # supplying the unity_coeff here to scale the weights
    tm.update(dt, t, layers, weights, unity_coeff, teacher_forcing=True)
    
    t += dt
    # Log the progress
    if t > savetime :
        # Put log update here to have (more or less) fixed sample rate
        # Now this is only to check progress
        print(f'Time at t={t} ns') 
        savetime += savestep
    
    time_log.add_tstep(t, layers, unity_coeff)

end = time.time()
print('Time used:',end-start)

# This is a large pandas data frame of all system variables
result = time_log.get_timelog()

# %% [markdown]
# ### 7. Visualize results
# Plot results specific to certain nodes

# %%
#nodes = ['H0','H1','H2','H3','H4']

nodes = ['H0','H1','H2','H3','H4','H5']
plotter.plot_nodes(result, nodes)

# %%

nodes = ['K0','K1','K2','K3','K4','K5']
plotter.plot_nodes(result, nodes)

# %%

plotter.visualize_dynamic_result(result,'O1-Iout-blue')

plotter.visualize_dynamic_result(result,'O0-Iout-red')

plotter.visualize_dynamic_result(result,'I2-Iout-green')
# %% [markdown]
# For this system it's quite elegant to use the `plot_chainlist` function, taking as arguments a graph object, the source node (I1 for blue) and a target node (O1 for blue)

# %%
# Variable G contains a graph object descibing the network
G = plotter.retrieve_G(layers, weights)
plotter.plot_chainlist(result,G,'I1','K0')

# %% [markdown]
# Plot specific attributes

# %%
attr_list = ['Vgate','Vexc']
plotter.plot_attributes(result, attr_list)

# %% [markdown]
# We can be totally specific if we want. First we list the available columns to choose from

# %%
print(result.columns)

# %%
plotter.visualize_dynamic_result(result, ['I0-Iout-red','I1-Iout-blue'])

# %%
plotter.visualize_dynamic_result(result, ['H0-Iout','H0-Pout','K0-Iout','K0-Pout'])

# %%
plotter.visualize_transistor(propagator.transistorIV,propagator.transistorIV_example())

# %%
plotter.visualize_LED_efficiency(propagator.eta_example(propagator.eta_ABC))

# %%
