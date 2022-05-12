#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 13:20:21 2022

Many of the methods have been adapted from Stone et al. 2017

@author: dwinge
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle

# Local imports
import pathintegration.trials as path_trials
import pathintegration.cx_rate as path_cxrate
import pathintegration.cx_basic as path_cxbasic

import stone
from context import physics
from context import plotter
from scipy.interpolate import interp1d

DATA_PATH='data'

def analyze_inbound(df, Tout, Tinb, search_fraction=0.4) :
    
    # 1. Closest distance to nest when homing
    dist = np.sqrt(np.power(df['x'],2)+np.power(df['y'],2))
    try :
        closest_dist_l = [dist[k].min() for k in range(0,dist.index[-1][0]+1)]
    except TypeError : # in case df is a simple DataFrame we revert to old code
        closest_dist_l = dist.min()
    
    # 2. Estimate size of search pattern
    def searchwidth(df,search_fraction, Tout, Tinb) :
        
        search_time = Tout+Tinb*(1-search_fraction)
        search_res = df[df['Time']>search_time]
        
        min_x = search_res['x'].min()
        max_x = search_res['x'].max()    
        min_y = search_res['y'].min()
        max_y = search_res['y'].max()
        
        return (max_x-min_x+max_y-min_y)/2

    try :
        search_dist_l = [searchwidth(df.loc[k],search_fraction,Tout,Tinb) for k in range(0,dist.index[-1][0]+1)]
    except TypeError : # in case df is a simple DataFrame we revert to old code
        search_dist_l = searchwidth(df, search_fraction, Tout, Tinb)
        
    return np.mean(closest_dist_l),np.std(closest_dist_l),np.mean(search_dist_l),np.std(search_dist_l)


# Calculate the position of the agent
def integrate_to_position(headings, velocity) :
    x = np.array([0.,0.]) # starting position
    for k,_ in enumerate(headings) :
        x += velocity[k]
        
    return x

def construct_out_travel(Tout,headings,velocity) :
    # Package this into our DataFrame format
    out_time=np.arange(0,Tout)
    out_travel = pd.DataFrame()
    out_travel['Time'] = out_time
    x = np.zeros((Tout,2))
    x[0] = np.array([0.,0.]) # starting position
    for k in range(1,Tout):
        x[k] = x[k-1] + velocity[k]

    out_travel[['x','y']] = x
    out_travel[['vx','vy']] = velocity
    out_travel['heading']=headings
    
    return out_travel

def run_trial(trial_nw,Tout=1000,Tinb=1500, hupdate=0.00025,
              inputscaling=1.0,noise=0.0, savestep=1.0) :
    
    # Generate route as Tom Stone did
    headings, velocity = path_trials.generate_route(T=Tout, vary_speed=True)
    outbound_end_position = integrate_to_position(headings,velocity)

    out_travel = construct_out_travel(Tout, headings, velocity)
    
    stone_cx = path_cxrate.CXRatePontin(noise=noise) # use default settings
    
    # Generate sig:nals cl1, tn1 and tn2 for all time steps
    tb1 = np.zeros(path_cxbasic.N_TB1)
    memory = np.zeros(path_cxbasic.N_CPU4)
    cl1 = np.zeros((Tout,path_cxbasic.N_CL1))
    tn1 = np.zeros((Tout,path_cxbasic.N_TN1))
    tn2 = np.zeros((Tout,path_cxbasic.N_TN2))
    cpu4 = np.zeros((Tout,path_cxbasic.N_CPU4))
    tl2 =np.zeros((Tout,path_cxbasic.N_TL2))
    cpu1 =np.zeros((Tout,path_cxbasic.N_CPU1))
    
    for k in range(Tout) :
        tl2[k], cl1[k], tb1, tn1[k], tn2[k], memory, cpu4[k], cpu1[k], motor = path_trials.update_cells(headings[k], velocity[k], tb1, memory, stone_cx)

    # Define interpolation functions to feed some pre-produced values into the network
    def cl1_input(scaling) :
        return interp1d(range(Tout), cl1*scaling, axis=0)

    def tn2_input(scaling) :
        return interp1d(range(Tout), tn2*scaling, axis=0)
        
    # Setup the inputs for CL1 and TN1
    trial_nw.specify_inputs('CL1', cl1_input)
    trial_nw.specify_inputs('TN2', tn2_input)

    # Feed the network the correct input signals for the outbound travel
    out_res = trial_nw.evolve(T=Tout,savestep=savestep)
    
    # Let the network navigate the inbound journey
    inb_res, inb_travel = trial_nw.evolve(T=Tout+Tinb,reset=False,t0=Tout,inbound=True,
                                       initial_heading=headings[-1],
                                       initial_pos=outbound_end_position,
                                       initial_vel=velocity[-1],
                                       #savestep=0.1,
                                       updateheading_m=hupdate)

    return out_res, inb_res, out_travel, inb_travel

def setup_network(Rs=2e9, memupdate=0.0025, manipulate_shift=False, onset_shift=-0.1,
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
    setup_nw.weights['TB1->CPU4'].print_W()
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

def generate_filename(T_outbound, T_inbound, N, **kwargs):
    filename = 'out{0}_in{1}_N{2}'.format(str(T_outbound),
                                                str(T_inbound),
                                                str(N))
    for k, v in kwargs.items():
        filename += '_' + k + str(v)
    return filename + '.pkl'

def generate_figurename(T_outbound, T_inbound, N, **kwargs):
    filename = 'out{0}_in{1}_N{2}'.format(str(T_outbound),
                                          str(T_inbound),  
                                          str(N))
    for k, v in kwargs.items():
        filename += '_' + k + str(v)
    return filename

def load_dataset(T_outbound, T_inbound, N, **kwargs):
    filename = generate_filename(T_outbound, T_inbound, N,
                                 **kwargs)
    
    with open(os.path.join(DATA_PATH, filename),'rb') as f :
        # Items read sequentially
        OUT=pickle.load(f)
        INB=pickle.load(f)
        # now it's empty
    
    return OUT, INB


def save_dataset(OUT, INB, T_outbound, T_inbound, N,
                 **kwargs):
    filename = generate_filename(T_outbound, T_inbound, N,
                                 **kwargs)
        
    # save to a pickle file
    with open(os.path.join(DATA_PATH, filename),'wb') as f :
        pickle.dump(OUT,f)
        pickle.dump(INB,f)
        
    #(OUT, INB).to_pickle(os.path.join(DATA_PATH, filename))
    

def one_flight_results(out_res,inb_res,out_travel, inb_travel, sim_name, plot_path='plots'):
    # Create plots
    # 1. Combined trace plot
    comb_res = pd.concat([out_res,inb_res],ignore_index=True)
    fig,_ = plotter.plot_traces(comb_res, layers=['CL1','TB1','TN2','CPU4','Pontine','CPU1'],attr='Pout',titles=True)
    # Here we need to save the figure
    plotter.save_plot(fig,'traces_'+sim_name,plot_path)
       
    # 2. Plot the combined traveled route
    Tout=out_res['Time'].iloc[-1]
    Tinb=inb_res['Time'].iloc[-1]-Tout
    min_dist, _, search_width, _ = analyze_inbound(inb_travel,Tout,Tinb)
    # Plot the combined traveled route
    fig, ax = plt.subplots()
    out_travel.plot(x='x',y='y',style='purple',ax=ax,linewidth=0.5, label='Outbound')
    inb_travel.plot(x='x',y='y',style='g',ax=ax,linewidth=0.5, label='Inbound')
    #path_plotter.plot_route(headings, velocity, Tout, 0,ax=ax)
    ax.set_title(f'Closest dist: {min_dist:.1f}, search width: {search_width:.1f}')
    ax.annotate('N',(0,0),fontstyle='oblique',fontsize=14)
    # Save result again
    plotter.save_plot(fig,'route_'+sim_name,plot_path)
    
    # 3. Close figures
    plt.close('all')

def generate_dataset(T_outbound=1500, T_inbound=1500,N=1000,
                     save=True, make_plots=True, **kwargs):
    try:
        OUT, INB = load_dataset(T_outbound, T_inbound, N,
                                           **kwargs)
    except:

        # Create the correct network
        network_args = ['Rstore','memupdate'] # add more here when needed
        network_kwargs = {k:v for k,v in kwargs.items() if k in network_args}
        # I guess if filtered_kwargs is empty this is an empty call
        trial_nw = setup_network(**network_kwargs)
        # These are the other arguments that go into the run_trial call
        trial_kwargs = {k:v for k,v in kwargs.items() if k not in network_args}
        
        # Create somewhere to store figures
        if make_plots :
            dirname = generate_figurename(T_outbound,T_inbound,N,**kwargs)
            plot_dir = os.path.join('plots',dirname)
            if not os.path.isdir(plot_dir) : # check first for existance
                os.mkdir(plot_dir)
        
        l_OUT = [] # list of DataFrames
        l_INB = [] 
                          
        for i in range(N):
            out_res, inb_res, out_travel, inb_travel = run_trial(
                    trial_nw,
                    Tout=T_outbound,
                    Tinb=T_inbound,
                    **trial_kwargs) # remaining keywords go here
            
            # Generate plots here
            if make_plots :
                plotname=generate_figurename(T_outbound,T_inbound,i,**kwargs)
                one_flight_results(out_res,inb_res,out_travel,inb_travel,
                                   plotname,plot_path=plot_dir)
            # Save the routes to the lists
            l_OUT.append(out_travel)
            l_INB.append(inb_travel)
        
        # Combine the DataFrame to a big one
        key_sequence = np.arange(0,N)
        OUT = pd.concat(l_OUT,keys=key_sequence)
        INB = pd.concat(l_INB,keys=key_sequence)
        
        if save:
            save_dataset(OUT, INB, T_outbound, T_inbound, N,
                         **kwargs)
    return OUT, INB