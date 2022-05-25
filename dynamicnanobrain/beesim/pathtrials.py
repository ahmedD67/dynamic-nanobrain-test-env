#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:15:55 2022
Code taken from Tom Stone's pathintegration package from 2017. 
Adapted here to python 3 standard. 

@author: dwinge
"""
import numpy as np
from scipy.signal import lfilter
from scipy.interpolate import interp1d

# Imports from the code of Tom Stone
import pathintegration.cx_rate as path_cxrate

default_acc = 0.15  # A good value because keeps speed under 1
default_drag = 0.15
    
def get_flow(heading, velocity, pref_angle=np.pi/4):
    A = np.array([[np.sin(heading - pref_angle),
                   np.cos(heading - pref_angle)],
                  [np.sin(heading + pref_angle),
                   np.cos(heading + pref_angle)]])
    return np.dot(A, velocity)


def rotate(theta, r):
    """Return new heading after a rotation around Z axis."""
    return (theta + r + np.pi) % (2.0 * np.pi) - np.pi


def thrust(theta, acceleration):
    """Thrust vector from current heading and acceleration

    theta: clockwise radians around z-axis, where 0 is forward
    acceleration: float where max speed is ....?!?
    """
    return np.array([np.sin(theta), np.cos(theta)]) * acceleration


def get_next_state(heading, velocity, rotation, acceleration, drag=0.5):
    """Get new heading and velocity, based on relative rotation and
    acceleration and linear drag."""
    theta = rotate(heading, rotation)
    v = velocity + thrust(theta, acceleration)
    v -= drag * v
    return theta, v

def generate_route(T=1500, mean_acc=default_acc, drag=default_drag,
                   kappa=100.0, max_acc=default_acc, min_acc=0.0,
                   vary_speed=False):
    """Generate a random outbound route using bee_simulator physics.
    The rotations are drawn randomly from a von mises distribution and smoothed
    to ensure the agent makes more natural turns."""
    # Generate random turns
    mu = 0.0
    vm = np.random.vonmises(mu, kappa, T)
    rotation = lfilter([1.0], [1, -0.4], vm)
    rotation[0] = 0.0

    # Randomly sample some points within acceptable acceleration and
    # interpolate to create smoothly varying speed.
    if vary_speed:
        if T > 200:
            num_key_speeds = int(T / 50)
        else:
            num_key_speeds = 4
        x = np.linspace(0, 1, num_key_speeds)
        y = np.random.random(num_key_speeds) * (max_acc - min_acc) + min_acc
        f = interp1d(x, y, kind='cubic')
        xnew = np.linspace(0, 1, T, endpoint=True)
        acceleration = f(xnew)
    else:
        acceleration = mean_acc * np.ones(T)

    # Get headings and velocity for each step
    headings = np.zeros(T)
    velocity = np.zeros([T, 2])

    for t in range(1, T):
        headings[t], velocity[t, :] = get_next_state(
            heading=headings[t-1], velocity=velocity[t-1, :],
            rotation=rotation[t], acceleration=acceleration[t], drag=drag)
    return headings, velocity

def get_cx_instance(noise) :
    cx = path_cxrate.CXRatePontin(noise=noise) # use default settings
    return cx

def update_cells(heading, velocity, tb1, memory, cx, filtered_steps=0.0):
    """Generate activity for all cells, based on previous activity and current
    motion."""
    # Compass
    tl2 = cx.tl2_output(heading)
    cl1 = cx.cl1_output(tl2)
    tb1 = cx.tb1_output(cl1, tb1)

    # Speed
    flow = cx.get_flow(heading, velocity, filtered_steps)
    tn1 = cx.tn1_output(flow)
    tn2 = cx.tn2_output(flow)

    # Update memory for distance just travelled
    memory = cx.cpu4_update(memory, tb1, tn1, tn2)
    cpu4 = cx.cpu4_output(memory)

    # Steer based on memory and direction
    cpu1 = cx.cpu1_output(tb1, cpu4)
    motor = cx.motor_output(cpu1)
    return tl2, cl1, tb1, tn1, tn2, memory, cpu4, cpu1, motor