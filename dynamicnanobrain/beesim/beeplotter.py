#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 11:02:27 2022

@author: dwinge
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

nature_single = 89.0 / 25.4
nature_double = 183.0 / 25.4
nature_full = 247.0 / 25.4

def subplot_trace(target, res, layer, attr, titles) :
    # Get the relevant nodes
    columns = [name for name in res.columns if (attr in name) and (layer in name)]
    
    # If we are plotting CPU1, we permute the one step forward
    if layer == 'CPU1' :
        columns.insert(0,columns.pop(-1))
    
    #print(columns)
    
    if columns[0][3] == 'V' :
        ylabel = 'Voltage (V)' 
    else:
        ylabel = 'Current (nA)'
    
    #TIME, INDEX = np.meshgrid(res['Time'])
    node_idx = [x+1 for x in range(0,len(columns))]
    # Need to copy as assigned by reference
    node_labels = node_idx.copy()
    if layer == 'CPU1' :
        node_labels[0] = 'CPU1b_9'
        node_labels[-1] = 'CPU1b_8'
        
    import numpy as np
    TIME, INDEX = np.meshgrid(res['Time'].values,node_idx)
    # Produce a 2D plot of values over time
    im = target.pcolormesh(TIME,INDEX,res[columns].values.transpose(),
                           cmap='viridis', rasterized=True,
                           shading='auto')
    
    plt.colorbar(im, ax=target, label=ylabel)  
    target.set_yticks(np.array(node_idx))
    target.set_yticklabels(node_labels)
    target.set_ylabel('Node idx')
    target.set_xlabel('Time (ns)')
    if titles :
        target.set_title(layer)
    

def plot_traces(res, layers, attr, onecolumn=False, doublewidth=True,
                time_interval=None, titles=False)    :
           
    Nrows = len(layers)
    Ncols = 1 # Put traces with a shared x-axis
    if doublewidth : 
        nature_width = nature_double 
    else :  
        nature_width = nature_single
        
    fig, axs = plt.subplots(Nrows, Ncols, 
                            figsize=(nature_width*Ncols, 0.5*nature_single*Nrows),
                            sharex=True) 
    
    # Select the approperiate time interval
    if time_interval is not None :
        select_res = res[(res["Time"]>=time_interval[0]) & (res["Time"]<=time_interval[1])]
    else : 
        select_res = res
        
    if Nrows > 1 :
        for k, ax in enumerate(axs.flatten()) :
            subplot_trace(ax, select_res, layers[k], attr, titles)
    else:
        subplot_trace(axs, select_res, layers[0], attr, titles)
        
    plt.subplots_adjust(left=0.124, right=0.9, bottom=0.1, top=0.9, wspace=0.1)
    plt.tight_layout()
    
    return fig, axs

def plot_distance_v_param(min_dists, min_dist_stds, distances, param_vals,
                          param_name,
                          ax=None, label_font_size=11, unit_font_size=10,
                          title=None, x_lim=10000, y_lim=300):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(nature_single, nature_single))

    colors = [cm.viridis(x) for x in np.linspace(0, 1, len(param_vals))]

    for i in range(len(param_vals)):
        noise = param_vals[i]
        mu = min_dists[i]
        sigma = min_dist_stds[i]
        if noise != 'Random':
            ax.semilogx(distances, mu, color=colors[i], label=noise, lw=1);
        else:
            ax.semilogx(distances, mu, color=colors[i], label='Random walk',
                        lw=1);
        ax.fill_between(distances,
                        [m+s for m, s in zip(mu, sigma)],
                        [m-s for m, s in zip(mu, sigma)],
                        facecolor=colors[i], alpha=0.2);

    ax.set_xlim(10, x_lim)
    ax.set_ylim(0, y_lim)
    ax.set_title(title, fontsize=label_font_size)
    ax.tick_params(labelsize=unit_font_size)
    ax.set_xlabel('Route length (steps)', fontsize=label_font_size)
    ax.set_ylabel('Distance (steps)', fontsize=label_font_size)

    handles, labels = ax.get_legend_handles_labels()

    l = ax.legend(handles,
                  labels,
                  loc='best',
                  fontsize=label_font_size,
                  handlelength=0,
                  handletextpad=0,
                  title=f'{param_name}:')
    l.get_title().set_fontsize(label_font_size)
    for i, text in enumerate(l.get_texts()):
        text.set_color(colors[i])
    for handle in l.legendHandles:
        handle.set_visible(False)
    l.draw_frame(False)
    plt.tight_layout()
    return fig, ax