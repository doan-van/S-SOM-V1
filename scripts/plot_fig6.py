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
plt.style.use('ggplot')


var = 'PMSL'
ss, somsize, simi = ['DJF', 'MAM', 'JJA','SON'], range(4,21), ['ssim', 'cor']

import string
# Plot Silhoulete combine
fig = plt.figure(figsize=(10,6))
xx, yy = [.1,.6,.1,.6],[.6,.6,.1,.1]
for ik, k in enumerate(ss[:]):
    
    qper = {sim: [] for sim in simi}
    idir0 = 'output/'+k+'/'  
    for size in somsize[:]:   
        idir = idir0 +'/n'+'%.2d' % size +'/' 
        for sim in simi[:]:
            df = pd.read_csv(idir+sim+'_sihoute.csv',index_col=0)
            #print(df.mean().values)
            avg_score = df.mean().values[0]
            qper[sim].append( avg_score )
    
    df = pd.DataFrame(qper,index=somsize)
    xlabel, ylabel, ylim = 'SOM size (number of clusters)', 'Silhouette score', [.2,.8]
    
    
    
    ax = plt.axes([xx[ik],yy[ik],.38,.35])
    
    df.plot(ax=ax,kind='bar', color = ['r','g'],alpha=.5, edgecolor='k', width=0.5)
    plt.legend(['S-SOM','ED-SOM'],frameon=False, ncol=2)
    ax.set_ylim( ylim )
    ax.set_ylabel( ylabel )
    ax.set_xlabel( xlabel )
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.text(0.,1.02,string.ascii_lowercase[ik]+') '+k,fontsize=18, fontweight = 'bold', transform = ax.transAxes)
    
ofile = 'output/fig_silhouette.png'
plt.savefig(ofile, dpi=150) 
    
    
    









sys.exit()
plt.style.use('ggplot')
# Plot Topo error combine
fig = plt.figure(figsize=(10,6))
xx, yy = [.1,.6,.1,.6],[.6,.6,.1,.1]
for ik, k in enumerate(ss[:]):
    d1 = xr.open_dataset('output/selfsim/'+k+'.nc').rename({'sim':'ssim', 'from':'fr'})
    for sim in simi: d1[sim].values[:] = normalize1(d1[sim].values)
    toper = {sim: [] for sim in simi}
    for size in somsize[:]:
        idir0 = 'output/'+k+'/'        
        idir = idir0 +'/n'+'%.2d' % size +'/' 
        odir = idir + '/fig/'
        d2 = {sim: xr.open_dataset(idir + sim+'.nc') for sim in simi[:]}
        for sim in simi:
            d = (d2[sim].bmu_proj - d2[sim].smu_proj).values
            topo_e = np.where(np.abs(d)==1,0,1).mean()
            toper[sim].append(d2[sim].topo_error)
            #toper[sim].append( topo_e )
        
    df = pd.DataFrame(toper,index=somsize) #.plot(kind='bar')
    xlabel, ylabel = 'SOM size', 'Topographic error'
    m1, m2 = df.min().min(), df.max().max()
    ylim = [m1-(m2-m1)*.1,m2+(m2-m1)*.1]
    #ylim = [0.,.6]
    ylim = [1.,2.2]
    
    
    ax = plt.axes([xx[ik],yy[ik],.38,.35])
    
    df.plot(ax=ax,kind='bar', color = ['r','g'],alpha=.5, edgecolor='k', width=0.5)
    plt.legend(['S-SOM','ED-SOM'],frameon=False)
    ax.set_ylim( ylim )
    ax.set_ylabel( ylabel )
    ax.set_xlabel( xlabel )
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.text(0.,1.02,string.ascii_lowercase[ik]+') '+k,fontsize=18, fontweight = 'bold', transform = ax.transAxes)
    
ofile = 'output/fig_topopreserv.png'
plt.savefig(ofile, dpi=150)   





    
    
    
    
    
    
    
    
    
       