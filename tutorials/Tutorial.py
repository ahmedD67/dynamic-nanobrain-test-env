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

# %% [markdown]
# ## Example of network generation and dynamic simulation

# %%
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

import time

# load the modules specific to this project
from context import network as nw
from context import physics
from context import timemarching as tm
from context import plotter
from context import logger


# %% [markdown]
# ### 1. Define the broadcasting channels of the network
# This is done by creating a list of the channel names. The names are arbitrary and can be set by the user, such as 'postive', 'negative' or explicit wavelenghts like '870 nm', '700 nm'. Here I chose the colors 'red' and 'blue'.

# %%
channel_list = ['red', 'blue']
# Automatically generate the object that handles them
channels = {channel_list[v] : v for v in range(len(channel_list))}
# It looks like this
print(channels)

# %% [markdown]
# ### 2. Define the layers
# Define the layers of nodes in terms of how they are connected to the channels. Layers and weights are organized in dictionaries. The input and output layers do not need to be changed, but for the hidden layer we need to specify the number of nodes N and assign the correct channels to the input/output of the node.

# %%
# Create layers ordered from 0 to P organized in a dictionary
layers = {} 
# An input layer automatically creates on node for each channel that we define
layers[0] = nw.InputLayer(N=2)
layers[1] = nw.HiddenLayer(N=1, output_channel='red',excitation_channel='blue',inhibition_channel='red')
layers[2] = nw.OutputLayer(N=1) # for now it can be the same as input

# %% [markdown]
# ### 3. Define existing connections between layers
# The weights are set in two steps. 
# First the connetions between layers are defined. This should be done using the keys defined for each layer above, i.e. 0, 1, 2 ... for input, hidden and output layers, respectively. The `connect_layers` function returns a weight matrix object that we store under a chosen key, for example `'inp->hid'`.
# Second, the specific connections on the node-to-node level are specified using the node index in each layer and the function `connect_nodes` of the weights object.

# %%
# Define the overall connectivity
weights = {}
# The syntax is connect_layers(from_layer, to_layer, layers, channels)
weights['inp->hid(blue)'] = nw.connect_layers(0, 1, layers, channel='blue')
weights['inp->hid(red)'] = nw.connect_layers(0, 1, layers, channel='red')
weights['hid->out'] = nw.connect_layers(1, 2, layers, channel='red')
# Recurrent connections possible
weights['hid->hid'] = nw.connect_layers(1, 1, layers, channel='red')

# Define the specific node-to-node connections in the weight matrices
# The syntax is connect_nodes(from_node, to_node, weight=value in weight matrix (default=1.0))
weights['inp->hid(blue)'].connect_nodes(0 ,0) # channels['blue']=1
weights['inp->hid(red)'].connect_nodes(1 ,0)   # channels['red']=0
weights['hid->out'].connect_nodes(0, 0)

# The explicit weight matrix can be set by
# weights['inp->hid'].W = np.array([])
# where the array should have the shape of weights['inp->hid'].W.shape()

# %% [markdown]
# ### 4. Visualize the network 
# The `plotter` module supplies functions to visualize the network structure. The nodes are named by the layer type (Input, Hidden or Output) and the index.

# %%
plotter.visualize_network(layers, weights)

# %% [markdown]
# ### 5. Specify the physics of the nodes
# Before running any simulations, we need to specify the input currents and the physics of the hidden layer nodes. Parameters can either be specified directly or coupled from the `physics` module. 

# %%
# Define a devices for the hidden layer
device = physics.Device('../parameters/device_parameters.txt')
# Assign it to the hidden layer
layers[1].assign_device(device)
# Get a quick report on what kind of currents that we need
unity_coeff, Imax = device.inverse_gain_coefficient(device.eta_ABC, layers[1].Vthres)
print(f'Unity coupling coefficient calculated as unity_coeff={unity_coeff:.4f}')
print(f'Imax is found to be {Imax} nA')

# %%
# Specify an exciting current square pulse and an inhibition square pulse
t_blue = [(2,3),(5.5,6),(7.5,8)] # 
# and inhibition from the other one
t_red = [(4,5)] # 

# Use the square pulse function and specify which node in the input layer gets which pulse
layers[0].set_input_func_per_node(node=0,func_handle=physics.square_pulse, func_args=(t_blue,Imax))
layers[0].set_input_func_per_node(node=1,func_handle=physics.square_pulse, func_args=(t_red, Imax))

# %% [markdown]
# ### 6. Example dynamic simulation

# %%
# Start time t, end time T
t = 0.0
T = 10.0 # ns
# These parameters are used to determine an appropriate time step each update
dtmax = 0.01 # ns 
dVmax = 0.001 # V

nw.reset(layers)
# Create an instance of Logger to store the data
time_log = logger.Logger(layers,channels) 

start = time.time()

while t < T:
    # evolve by calculating derivatives, provides dt
    dt = tm.evolve(t, layers, dVmax, dtmax )

    # update with explicit Euler using dt
    tm.update(dt, t, layers, weights)
    
    t += dt
    # Log the update
    time_log.add_tstep(t, layers, unity_coeff=unity_coeff)

end = time.time()
print('Time used:',end-start)

res = time_log.get_timelog()

# %% [markdown]
# Visualize results using `plotter` module

# %%
# Plot the results for a node
plotter.plot_nodes(res,['H0'])

# %%
# Plot the linked currents from I1/I0 to O0
G = plotter.retrieve_G(layers, weights)
plotter.plot_chainlist(res,G,'I1','O0')
