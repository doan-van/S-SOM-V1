#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Self-organizing map
# Doan Quang-Van
import glob, os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
plt.style.use('ggplot')


    

# Plot figure 5
var = 'PMSL'
ss, somsize, simi = ['MAM','SON', 'DJF', 'JJA'], range(4,21), ['ssim', 'cor']
import string
for k in ['MAM','SON', 'DJF', 'JJA'][2:3]:    
    qper = {sim: [] for sim in simi}
    fig = plt.figure(figsize=(3.5*4.1,3.5*2.2))
    ixx = np.linspace(.05,1,5)
    for ix, size in enumerate([4,8,12,16][:]): # range(4,21)[:1]:   
        idir0 = 'output/'+k+'/'        
        idir = idir0 +'/n'+'%.2d' % size +'/' 
        iyy = [.56,.1]
        somlab = ['S-SOM','ED-SOM']
        for iy, sim in enumerate(simi[:]):
            df = pd.read_csv(idir+sim+'_sihoute.csv',index_col=0)
            print(df.mean().values)
            
            ax = plt.axes([ixx[ix],iyy[iy],.2,.35])
            y_lower, y_upper = 0, 0
            for ic in range(size):
                cluster_silhouette_vals = df.loc[ic].values[:,0]
                cluster_silhouette_vals.sort()
                y_upper += len(cluster_silhouette_vals)
                ax.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1, alpha=.8)

                if size < 10:
                    ax.text(-0.05, y_lower + 0.5 * len(cluster_silhouette_vals), str(ic+1),ha='center',va='center')
                else:
                    if np.mod(ic,2)==1:
                        print('hello')
                        ax.text(-0.05, y_lower + 0.5 * len(cluster_silhouette_vals), str(ic+1),ha='center',va='center')
                        
                y_lower += len(cluster_silhouette_vals)
        
        
            # Get the average silhouette score and plot it
            avg_score = df.mean().values[0]
            ax.axvline(avg_score, linestyle='--', linewidth=2, color='r')
            ax.set_yticks([])
            ax.set_xlim([-0.1, 1])
            ax.set_xlabel('Silhouette coefficient values')
            ax.set_ylabel('Cluster labels')
            ax.spines['bottom'].set_position(('axes', -0.0))
            #ax.yaxis.set_ticks_position('left')
            ax.spines['left'].set_position(('axes', -0.0))
            #ax.spines['right'].set_color('none')
            #ax.spines['top'].set_color('none')
            #ax.set_title('Silhouette plot for the various clusters', y=1.02);
            plt.setp(ax.spines.values(), linewidth=0.5)
            ax.xaxis.set_tick_params(width=0.5)
            # Scatter plot of data colored with labels
            plt.grid(None)
            qper[sim].append( avg_score )
            
            ax.text(0.,1.02,string.ascii_lowercase[iy*4 + ix]+') ',fontsize=18, fontweight = 'bold', transform = ax.transAxes)
            if ix == 0: plt.text(0.015,0.8 - iy*0.5,somlab[iy],fontsize=18, fontweight = 'bold', transform = fig.transFigure, ha='center', va='center', rotation=90)

    ofile = idir0+'/fig_silhouette_com_rev.png'
    plt.savefig(ofile, dpi=150)










    
    
    
    
    
    
    
    
    
       