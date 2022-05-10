#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 15:56:00 2022

@author: dwinge
"""

import numpy as np
import matplotlib.pyplot as plt

a=0

def hej(t) :
    return a

print(f'Evaluating hej as {hej(1)}')

for k in range(0,10) :
    a+=1
    print(f'Evaluating hej as {hej(1)}')
    
heading = 0

def get_flow(heading, v, pref_angle=np.pi/4):
    head_arr = np.array([[np.sin(heading + pref_angle),
                          np.cos(heading + pref_angle)],
                         [np.sin(heading - pref_angle),
                          np.cos(heading - pref_angle)]])
    return np.einsum('ijk,jk->ik',head_arr,v)

def tn2_activity(t,heading, velocity) :
    tn2 = get_flow(heading, velocity)
    # scale by the standard current factor
    return np.clip(tn2,0,1)

def rotate(theta, r):
    """Return new heading after a rotation around Z axis."""
    return (theta + r + np.pi) % (2.0 * np.pi) - np.pi


theta = np.linspace(-np.pi,np.pi)
v = 0.15 * np.array([np.sin(theta), np.cos(theta)])
noise=0.05
v += np.random.normal(scale=noise, size=v.shape)
tn2_output = tn2_activity(0,theta,v)
#tn2_output = get_flow(heading, v)

pref_angle=np.pi/4
heading_arr = np.array([[np.sin(theta + pref_angle),
                         np.cos(theta + pref_angle)],
                        [np.sin(theta - pref_angle),
                         np.cos(theta - pref_angle)]])

#%%
plt.plot(theta,tn2_output[0],label='L')
plt.plot(theta,tn2_output[1],label='R')
plt.legend()
plt.show()

#%% 
plt.plot(theta,v[0],label='vx')
plt.plot(theta,v[1],label='vy')
plt.legend()
plt.grid(True)
plt.show()

#%% Pandas test
import pandas as pd
import numpy as np

df = pd.DataFrame(
    [["bar", "one"], ["bar", "two"], ["foo", "one"], ["foo", "two"]],
    columns=["first", "second"],
    )

pd.MultiIndex.from_frame(df)

arrays = [
    np.array(["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"]),
    np.array(["one", "two", "one", "two", "one", "two", "one", "two"]),
    ]

s = pd.Series(np.random.randn(8), index=arrays)


# Create 3 datasets

Nrows = 8
df1 = pd.DataFrame(np.random.randn(Nrows, 2),columns=['x','y'])
#df1['key'] = [1]*Nrows
df2 = pd.DataFrame(np.random.randn(Nrows+1, 2),columns=['x','y'])
#df2['key'] = [2]*Nrows

DF = pd.concat([df1,df2],keys=[0,1])

DF['x'][0]


DF.to_pickle('test.pkl')


unpickled_df = pd.read_pickle("./test.pkl")  


#%%
np.random.seed(1618033)

#Set 3 axis labels/dims
years = np.arange(2000,2010) #Years
samples = np.arange(0,20) #Samples
patients = np.array(["patient_%d" % i for i in range(0,3)]) #Patients

#Create random 3D array to simulate data from dims above
A_3D = np.random.random((years.size, samples.size, len(patients))) #(10, 20, 3)

# Create the MultiIndex from years, samples and patients.
midx = pd.MultiIndex.from_product([years, samples, patients])

# Create sample data for each patient, and add the MultiIndex.
patient_data = pd.DataFrame(np.random.randn(len(midx), 3), index = midx)