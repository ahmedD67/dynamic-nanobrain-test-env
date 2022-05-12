# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Bee flight simulator
# Here we use the modules stone and trialflight to test the capabilites of the physical device. 

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Local imports
from context import dynamicnanobrain
# core modules
import dynamicnanobrain.core.plotter as plotter
import dynamicnanobrain.core.physics as physics
# beesim modules
import dynamicnanobrain.beesim.stone as stone
import dynamicnanobrain.beesim.trialflight as trials
import dynamicnanobrain.beesim.beeplotter as beeplotter


plt.rcParams['figure.dpi'] = 200 # 200 e.g. is really fine, but slower


# %% [markdown]
# Method to setup the network with the desired physics. Here the storage time of the gate electrode is indirectly set by the RC constant via Rstore and Cstore. Changes to the activation functions, i.e. the transistor IVs, are made here. 

# %%
def setup_network(Rs=2e9, manipulate_shift=False, onset_shift=-0.1,
                  cpu_shift=0.05) :
    
    setup_nw = stone.StoneNetwork() 
    # Setup the internal devices
    devices = {}
    devices['TB1']=physics.Device('../parameters/device_parameters.txt')
    devices['CPU4']=physics.Device('../parameters/device_parameters.txt')
    #devices['CPU4'].set_parameter('Cstore',7e-16) # Original is 0.07 10^-15
    devices['CPU4'].set_parameter('Rstore',Rs) # Original 2e6
    devices['CPU4'].print_parameter('Cstore')
    devices['CPU4'].print_parameter('Rstore')
    print(f'Calculate tau_gate={devices["CPU4"].calc_tau_gate()} ns')
    #setup_nw.weights['TB1->CPU4'].print_W()
    devices['CPU1a']=physics.Device('../parameters/device_parameters.txt')
    devices['CPU1b']=physics.Device('../parameters/device_parameters.txt')
    devices['Pontine']=physics.Device('../parameters/device_parameters.txt')

    if manipulate_shift :
        devices["TB1"].p_dict['Vt'] = onset_shift
        devices["CPU4"].p_dict['Vt'] = cpu_shift
        devices["CPU1a"].p_dict['Vt'] = cpu_shift
        devices["CPU1b"].p_dict['Vt'] = cpu_shift
        devices["Pontine"].p_dict['Vt'] = cpu_shift

    # Feed the devices into the network
    setup_nw.assign_device(devices, unity_key='TB1')
    
    return setup_nw


# %% [markdown]
# Create network using the above method. A estimate of the storage time is given.

# %%
my_nw = setup_network()

# %% [markdown]
# #### Available layouts for network visualization:
# **multipartite**: Standard neural network appearance. Hard to see recurrent couplings within layers.  
# **circular**: Nodes drawn as a circle  
# **shell**: Layers drawn as concetric circles  
# **kamada_kawai**: Optimization to minimize weighted internode distance in graph  
# **spring**: Force-directed layout which is standard in `networkx`
#

# %%
# Draw a shell-structure
shell_pos = my_nw.show_network(layout='shell',shell_order=('TB1','CPU4',['CPU1a','CPU1b']),exclude_layers=['CL1','TN2','Pontine'])

# %%
# Including the Pontine cells
spring_pos= my_nw.show_network(exclude_layers=['CL1','TN2'],layout='spring',k=20)

# %%
# Yet another version of a drawing
kamada_pos=my_nw.show_network(exclude_layers=['CL1','TN2','Pontine'],layout='kamada_kawai')

# %%
fig, ax = my_nw.show_weights()

# %% [markdown]
# Check transistors

# %%
Vleak_dict={}
my_nw.show_devices(Vleak_dict, Vstart=-0.7)

# %% [markdown]
# ## Running the model
# At this point we are all set up to do the evolution. Outbound and inbound flight are simulated after each other to test homing behavior.

# %%
Tout=1000
Tinb=1000
out_res, inb_res, out_travel, inb_travel = trials.run_trial(my_nw,Tout,Tinb)

# %% ,arkdown [markdown]
# ## Analyze the result
# First, check the closest position to the nest for the inbound flight. Then make a few plots to illustrate the process.

# %%
min_dist, _, search_width, _ = trials.analyze_inbound(inb_travel,Tout,Tinb)

# %%
import warnings
warnings.filterwarnings('ignore',category=UserWarning) # get rid of some red text...

plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower
# 1. Combined trace plot
comb_res = pd.concat([out_res,inb_res],ignore_index=True)
fig,_ = beeplotter.plot_traces(comb_res, layers=['CL1','TB1','TN2','CPU4','Pontine','CPU1'],attr='Pout',titles=True)

# %%
# Plot the combined traveled route
fig, ax = plt.subplots()
out_travel.plot(x='x',y='y',style='purple',ax=ax,linewidth=0.5, label='Outbound')
inb_travel.plot(x='x',y='y',style='g',ax=ax,linewidth=0.5, label='Inbound')
#path_plotter.plot_route(headings, velocity, Tout, 0,ax=ax)
ax.set_title(f'Closest dist: {min_dist:.1f}, search width: {search_width:.1f}')
ax.annotate('N',(0,0),fontstyle='oblique',fontsize=14)


# %%
# Specific nodes can be especially studied. 
# When plotting in a notebook, use the onecolumn flag as below.
TB1_list = [f'TB1_{idx}' for idx in range(0,8)]
plotter.plot_nodes(out_res, nodes=TB1_list, onecolumn=True)
#CPU4_list = [f'CPU4_{idx}' for idx in range(0,16)]
#plotter.plot_nodes(out_res, nodes=CPU4_list)

# Plot the heading and motor and velocity
#inb_travel.plot(x='Time',y=['heading','motor'],style='-o')
#inb_travel.plot(x='Time',y=['vx','vy'])

# %%
