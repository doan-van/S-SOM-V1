#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import glob, os, sys
import pandas as pd
import cartopy.crs as ccrs
import xarray as xr
import matplotlib.pyplot as plt
from wrf import to_np, smooth2d, get_basemap
from matplotlib.colors import BoundaryNorm
plt.style.use('ggplot')






idir0 = 'output/'
var = 'PMSL'
simi = ['cor','ssim', 'ed']
somsize = range(4,21)


        
import string
somlab = ['S-SOM','ED-SOM']
for key in ['MAM','SON', 'DJF', 'JJA'][:]:    
   for n in somsize[:1]:
       nxy_str = 'n'+'%.2d' % n
       idir0 = 'output/'+key+'/'   
       idir = idir0 +'/'+nxy_str+'/'
       odir = idir + '/fig/'
       simi = ['ssim', 'ed']
       
       dss = {sim: xr.open_dataset(idir + sim+'.nc') for sim in simi[:]}
       #lat, lon = ds.lat2d.values, ds.lon2d.values
   
       maxv = int(max([dss[sim][var].max() for sim in simi])/4)*4 + 5
       minv = int(max([dss[sim][var].min() for sim in simi])/4)*4 - 4
       levels = np.arange(minv, maxv,4)
       levels = np.arange(minv, maxv,2)
       #for sim in simi[:]: plot_hist(dss[sim].bmu_proj.values,idir+'fig_hist_'+sim+'.png')
       #sys.exit()   
       fig = plt.figure(figsize=(3.5*5.1,3.5*2.2))
       iyy = [.55,.05]
       iabc = 0
       
       for iy, sim in enumerate(simi[:1]):
           
           ds = dss[sim][var]
           lat, lon = ds.lat2d.values, ds.lon2d.values
           sh = ds.shape
           dat = ds.values.reshape(n,sh[-2], sh[-1])
           
           ixx = np.linspace(.05,1,6)
           ax = plt.axes([ixx[4]+.04,iyy[iy]+.05,.12,.28])
           
           val = dss[sim].bmu_proj.values
           h = pd.Series(val).value_counts().sort_index()
           h.index = h.index + 1
           x, y = np.array(h.index), h
           ax.bar(x,y, clip_on=False,width=.5)
           #heights, positions, patches = axes.hist(X, color='white')
           # You can specify a rotation for the tick labels in degrees or with keywords.
           #plt.xticks(x[::ii], x[::ii], rotation='vertical')
           #ax.spines['right'].set_color('none')
           #ax.spines['top'].set_color('none')
           ax.xaxis.set_ticks_position('bottom')
           # was: axes.spines['bottom'].set_position(('data',1.1*X.min()))
           ax.spines['bottom'].set_position(('axes', -0.0))
           ax.yaxis.set_ticks_position('left')
           ax.spines['left'].set_position(('axes', -0.0))
           ax.set_xlim([.5, max(x)+.5])
           ax.xaxis.grid(False)
           ax.yaxis.grid(False)
           fig.tight_layout()
           plt.xlabel('SOM node')
           plt.ylabel('Count')
           plt.setp(ax.spines.values(), linewidth=0.5)
           ax.xaxis.set_tick_params(width=0.5)
           
           plt.grid(True)
           ax.text(0.,1.02,string.ascii_lowercase[iy*5 + 4]+') ',fontsize=18, fontweight = 'bold', transform = ax.transAxes)
           plt.text(0.015,0.8 - iy*0.5,somlab[iy],fontsize=18, fontweight = 'bold', transform = fig.transFigure, ha='center', va='center', rotation=90)

           
           
           
           for ix, inode in enumerate(range(n)[:]):
               print(sim, inode)
               d = dat[inode]
               ax = plt.axes([ixx[ix],iyy[iy],.18,.4])
               lat_p, lon_p = 35., 140.
               m = Basemap(width=3.e6,height=2.9e6,
                   rsphere=(6378137.00,6356752.3142),\
                   resolution='l',area_thresh=1000.,projection='lcc',\
                   lat_1=lat_p,lat_2=lat_p,lat_0=lat_p,lon_0=lon_p )     
               x, y = m(lon, lat)
               z = to_np(smooth2d(d,3))
               cmap= plt.get_cmap("jet")
               m.contourf(x, y, z, levels = levels, cmap = cmap, alpha=1.,extend = 'both',)
               cs = m.contour(x, y, z, levels = levels[::2], linewidths=0.5, colors='k',extend = 'both',) #, cmap= 'k') # plt.get_cmap("jet"))
               plt.clabel(cs, inline=True, fmt='%1.0f', fontsize=12, colors='k')
    
               m.drawcoastlines(linewidth=0.5, color='k',zorder=200)
               m.drawparallels(np.arange(-80,81,10),labels=[1,0,0,0],rotation=90,zorder=200)
               m.drawmeridians(np.arange(0,360,10),labels=[0,0,0,1],zorder=200,linewidth=1.)
               #m.drawmapboundary(fill_color='aqua')   
               ax.text(0.,1.02,string.ascii_lowercase[iy*5 + ix]+') N-'+str(inode+1),fontsize=18, fontweight = 'bold', transform = ax.transAxes)


           
       ofile = idir0+'/fig_som_com.png'
       plt.savefig(ofile, dpi=150)           
       
       
        
        
        
  



