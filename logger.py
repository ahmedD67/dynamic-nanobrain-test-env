#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 13:46:38 2021

@author: dwinge
"""

import numpy as np
import pandas as pd

class Logger :
    
    def __init__(self,layers,channels) :
        self.list_data = []
        # Need to get the node names
        self.column_labels = self.column_names(layers,channels)
        
    def column_names(self, layers, channels) :
        names=['Time']
        for idx in layers.keys():
            node_list = layers[idx].get_names(idx)          
            if layers[idx].layer_type == 'input' :
                for key in channels :
                    names.append('I'+str(channels[key])+'-Iout-'+key)
                
            elif layers[idx].layer_type == 'hidden' :
                # Voltages
                for node in node_list :
                    names.append(node+'-Vinh')
                    names.append(node+'-Vexc')
                    names.append(node+'-Vgate')
                # Input currents
                for node in node_list :
                    names.append(node+'-Iinh')
                    names.append(node+'-Iexc')
                # Output currents
                for node in node_list :
                    names.append(node+'-Iout')
                    names.append(node+'-ISD')  
             
            elif layers[idx].layer_type == 'output' :
                # Currents
                for key in channels :
                    names.append('O'+str(channels[key])+'-Iout-'+key)
                    
            else :
                print('Unexpected layer_type in logger.column_names')
                raise RuntimeError
        return names
        
    def add_tstep(self,t,layers) :
        # Extract the data from each node in layers
        row = [t]
        for idx in layers.keys():
            # Node names
            #name_list = layers[idx].get_names(idx)       
            if layers[idx].layer_type == 'input' :
                curr=layers[idx].I.flatten(order='F').tolist()
                row += curr 
            
            elif layers[idx].layer_type == 'hidden' :
                # Voltages
                volt=layers[idx].V.flatten(order='F').tolist()
                row +=volt
                # Input currents
                curr=layers[idx].B[:2].flatten(order='F').tolist()
                row += curr 
                # Output currents
                curr=layers[idx].I.flatten(order='F').tolist()
                row += curr 
                curr=layers[idx].ISD.flatten(order='F').tolist()
                row += curr 
             
            elif layers[idx].layer_type == 'output' :
                # Voltages
                curr=layers[idx].B.diagonal().flatten(order='F').tolist()
                row += curr 
            else :
                print('Unexpected layer_type in logger.add_tstep')
                raise RuntimeError
                
        self.list_data.append(row)
        
    def get_timelog(self) :
        # Convert to pandas data frame
        return pd.DataFrame(self.list_data, columns=self.column_labels)
