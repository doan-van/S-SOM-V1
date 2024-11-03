#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Self-organizing map
# Author: Doan Quang-Van
#======================
import glob, os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from sompy import *


# This script is to run SOM (with both S-SIM and ED) for classifying 
# weather patterns for 4 seasons for the Japan reagion.

# SOM configuration is 
# 1-D map with size from 4 to 20
# Different similarity index
var = 'PMSL' # mean sea level pressure
ifile = 'PMSL_all.nc' # input file in netcdf format



# reading and processing input data
d1 = xr.open_dataset(ifile)[var] 
d1.lon2d.values[d1.lon2d.values < 0] = d1.lon2d.values[d1.lon2d.values < 0] + 360
d1 = d1.isel(lat=slice(0,65),lon=slice(19,91)) 
lat2d, lon2d = d1['lat2d'].values,d1['lon2d'].values 
# grouping by seasons
d2 = {g[0]:g[1] for g in d1.groupby('time.season')}
# run SOM for each seasons
for key in ['MAM','SON', 'DJF', 'JJA'][2:3] :
    odir0 = 'output/'+key+'/'        
    d3 = d2[key].loc['2010':'2015']
    odir = 'output/'+key +'/'
    if not os.path.exists(odir): os.makedirs(odir)     
    d3.to_netcdf('output/'+key+'/orgdata.nc') # write original data for latter checking if necessary 
    
    iput2d = d3.values / 100.
    iput1d = iput2d.reshape(iput2d.shape[0],-1)
    
    for size in range(4,21)[:1]:
        
        for sim in ['ssim', 'ed','cor'][:]: # two similarity indices

            n, iterate = size, 5000 # size of 1-D SOM and number of interation
         
            # run SOM
            somout = som(iput1d, n, iterate = iterate, sim=sim) 

            # write out data 
            odir = odir0 +'/n'+'%.2d' % n +'/'
            if not os.path.exists(odir): os.makedirs(odir)   
            do = xr.Dataset() 
            y = somout['som'].reshape(n,iput2d.shape[-2],iput2d.shape[-1])
            do[var] = (['n','lat', 'lon'],  y)
            do['bmu_proj'] = (('input'), np.array(somout['bmu_proj_fin']))
            do['smu_proj'] = (('input'), np.array(somout['smu_proj_fin']))
            do.coords['lat2d'] =  (['lat','lon'], lat2d)
            do.coords['lon2d'] =  (['lat', 'lon'], lon2d)
            do.attrs['topo_error'] = somout['topo_error']
            do['simbtw'] = ( ('n1','n2') , somout['sim_btw'] )
            
            ofile = odir + sim+'.nc' 
            print(ofile)
            do.to_netcdf(ofile)
            do.close()
            
            # end write out data
            


        
    























