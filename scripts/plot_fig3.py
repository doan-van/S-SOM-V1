#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Self-organizing map
# Doan Quang-Van
import glob, os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from numpy import random as rand
from sompy import *
import string


#==============================================================================
def calculate_similarity_index():
    '''
    Calculate data used for ploting figure 3
    Read input data 
    Calculate similarity between data (for each year as data is big)
    The combine for seasons
    '''
    var = 'PMSL'
    ifile = 'PMSL_all.nc'
    d1 = xr.open_dataset(ifile)[var]
    d1.lon2d.values[d1.lon2d.values < 0] = d1.lon2d.values[d1.lon2d.values < 0] + 360
    d1 = d1.isel(lat=slice(0,65),lon=slice(19,91)) 
    lat2d, lon2d = d1['lat2d'].values,d1['lon2d'].values 
    d2 = {g[0]:g[1] for g in d1.groupby('time.season')}
    d2['all'] = d1
    
    yy = np.arange(1979,2020)
    for year in yy: 
        print(year)
        dd = d1.loc[str(year)] / 100.
        da = d1 / 100.
        xx, yy = dd[:], da[:]
        edv = np.zeros([len(xx), len(yy)])
        simv = edv.copy()
        for ix, x in enumerate(xx):
            print(x.time.values)
            for iy, y in enumerate(yy): 
                simv[ix,iy] = strsim(x.values, y.values)
                edv[ix,iy]  = -np.linalg.norm(x.values - y.values)
    
        do = xr.Dataset()
        do['ed'] = (('from', 'to'), edv)
        do['sim'] = (('from', 'to'), simv)
        do.coords['from'] = (('from'),xx.time)
        do.coords['to'] = (('to'),yy.time)
        
        odir = 'output/selfsim/'
        if not os.path.exists(odir): os.makedirs(odir)          
        do.to_netcdf(odir+str(year)+'.nc')
    
    idir = 'output/selfsim/'
    yy = np.arange(1979,2020)
    ds = []
    for year in yy:
        ds.append(xr.open_dataset(idir+str(year)+'.nc'))
    ds = xr.concat(ds,dim='from')
                
    d2 = {g[0]: [ g2[1] for g2 in g[1].groupby('to.season') if g2[0] == g[0]][0] for g in ds.groupby('from.season')}
    for k in d2.keys(): d2[k].to_netcdf(idir+k+'.nc')
#==============================================================================
    



def normalize1(x): return ( x - x.min() ) / ( x.max() - x.min())
def normalize2(x): return ( x ) / ( x.max() - x.min())



# calculate similarity index: (it takes time)
calculate_similarity_index()
#======


#======
# Plot figure 3
idir = 'output/selfsim/'
from scipy.stats import kurtosis, skew
season = ['DJF', 'MAM', 'JJA', 'SON']
simi = ['sim','ed','cor']

fig = plt.figure(figsize=(12,3.5))
     
for ik, k in enumerate(season[:]):
    ds = xr.open_dataset(idir+k+'.nc')

    ax = plt.axes([.05+ik*.25,.15,.19,.7])   
    label = {'sim':'S-SIM', 'ed':'ED'}
    color = {'sim':'r', 'ed':'g'}
    for sim in simi[-1:]:
        x = ds[sim].values #.isel(n1=bmuind[u1],n2=bmuind[u2])
        #x1 = x.values.mean() * x.size / do[sim].size
        #plt.hist(x.values.flatten(),bins=10,alpha=.5)
        xn = normalize1(x)
        x1 = xn[np.triu_indices(xn.shape[0],k=1)]
        print(k, sim)
        ax.hist(x1,bins = np.arange(0.,1.,.01),
                label=label[sim],alpha=.4,color=color[sim], 
                histtype='stepfilled',lw=0.5,ec="k")
 
        plt.legend(loc=2, frameon=False)
        ax.set_ylim(0,3.5e5)
        ax.set_ylabel('Count')
        ax.set_xlabel('Normalized similarity')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['bottom'].set_position(('axes', -0.0))
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].set_position(('axes', -0.0))
        ax.set_xlim([0, 1])
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)
        ax.ticklabel_format(style='plain')
        #ax.ticklabel_format(useOffset=False)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-4,4))
        text =  string.ascii_lowercase[ik] + ') '+ k
        ax.text(0.,1.1,text,fontsize=18, fontweight = 'bold', transform = ax.transAxes)
        plt.grid(True)
        fig.tight_layout()
        
plt.savefig('output/fig_hist.png', dpi=150)





























    
    


















