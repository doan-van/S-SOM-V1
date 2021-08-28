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


def cal_som_sim(idir):
    var = 'PMSL'
    do = xr.Dataset() 
    for sim in ['ssim', 'ed'][:]:
        ifile = idir + sim+'.nc' #odir+'/'+key+'_'+fname[sim]+'_'+var+'_x'+str(nx)+'-y'+str(ny)+'.nc'
        ds = xr.open_dataset(ifile)    
        dat = ds[var].values
        print(idir, sim)
        if sim == 'ssim':
            values = [ strsim( d1, d2 ) for d1 in dat for d2 in dat ]
        if sim == 'ed': 
            values = [ - np.linalg.norm(d1 - d2) for d1 in dat for d2 in dat ]
        do[sim] = ( ('n','n') ,np.array(values).reshape(len(dat),len(dat))) #[np.triu_indices(len(dat),k=1)]
    do.to_netcdf(idir+'som_similarity.nc')
    
    

def normalize1(x): return ( x - x.min() ) / ( x.max() - x.min())
def normalize3(x,xmax,xmin): return ( x - xmin ) / ( xmax - xmin )    
def stdlize(x,xmean,xstd): return ( x - xmean ) / xstd  


def uni_dis(x,qq):
    if x < qq.min(): return 0
    else: return np.argwhere((x > qq[:-1]) & (x <= qq[1:]))[0][0] + 1 




# calculate Silhouete values
var = 'PMSL'
ss, somsize, simi = ['MAM','SON', 'DJF', 'JJA'], range(4,21), ['ssim', 'ed','cor']
for k in ss[:]:    
    d1 = xr.open_dataset('output/'+k+'/orgdata.nc')[var] / 100.

    
    #ds = xr.open_dataset('output/selfsim/'+k+'.nc').rename({'sim':'ssim', 'from':'fr'})
    #for sim in simi: ds[sim].values[:] = 1 - normalize1(ds[sim].values)
    
    for size in somsize[:]:
        print(k,size)
        idir0 = 'output/'+k+'/'        
        idir = idir0 +'/n'+'%.2d' % size +'/' 
        for sim in simi[-2:]:
            d2 = xr.open_dataset(idir + sim+'.nc')
            
            ds = xr.open_dataset('output/'+k+'/orgdata_'+sim+'_ss.nc')
            ds["sim"].values[:] = 1 - normalize1(ds["sim"].values)
            
            ds1 = ds["sim"].values
            
            soh = []
            for i in range(d1.shape[0])[:]:
                d3 = d1[i]     
                ithis = d2.bmu_proj.values[i]
                sindex = ds1[i]
                
                ab = np.array([sindex[d2.bmu_proj.values == ig].mean() for ig in range(size)])
                abis = np.argsort(ab)[::-1]
                ai = ab[ithis]
                if abis[0] == ithis: bi = ab[abis[1]]    
                else: bi = ab[abis[0]]
                
                soi = (bi - ai) / max(bi, ai)
                soh.append([ithis,soi])
                
            df = pd.DataFrame(soh)
            df.set_index(0,inplace=True)
            df.to_csv(idir+sim+'_sihoute.csv')
            print(df.mean(),df.max(),df.min())
    
 
    
    
    
    
    
       