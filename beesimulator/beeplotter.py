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