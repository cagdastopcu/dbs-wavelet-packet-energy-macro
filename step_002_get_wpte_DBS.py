# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 02:47:57 2022
copied 15/03/2022 to debug and refactoring

@author: cagdas
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:39:53 2022

@author: cagdas
"""
import matplotlib.pyplot as plt
import numpy as np 
import pylab as py 

import pandas as pd
from kcsd.KCSD import oKCSD3D
from mayavi import mlab
import nibabel as nib

from scipy.signal import filtfilt, butter, iirnotch, spectrogram, hilbert

from scipy.fft import fftshift

from scipy import stats

import scipy.io as sio

import matplotlib.pyplot as plt

import pywt

from dtw import *

# %matplotlib qt
# import pywt
# from password_MK import password
# import os
# from pymef.mef_session import MefSession

#%% load converted file 4 kHz 32 bit

sub_name_list = ['210319','210413','210505','210527',
            '210708','210805','210909']

sub_name = sub_name_list[6]

data = np.load('C:/WNencki/processing/dbs_macro/dbs_macro_filtered/DBS_sub_'+sub_name+'_macro_4khz_9sec_'+'ENCODE'+'_filt.npy')

# data_load.close()

#%% functions

def wpte_recon(data):
    """decompose one channel signal with wavelet packet transform and reconstruct for each freaquency band maxlevel = 2^^12 freq bands FS/2/2**maxlevel"""
    wp = pywt.WaveletPacket(data=data, wavelet='db5',maxlevel = 11, mode='symmetric')
    #changed to 12-11
    nodes_all = [n.path for n in wp.get_leaf_nodes(True)]
    #print(nodes_all)
    trim_point = 872
    # trim_point = 4584

    recons_wp = np.zeros((516,len(data)+trim_point)) #1000 hz
    # recons_wp = np.zeros((512,20488)) 
    for indx, node in enumerate(nodes_all[:516]): 
        new_wp_1 = pywt.WaveletPacket(data=None, wavelet='db5',maxlevel = 11, mode='symmetric') 
        new_wp_1[node] = wp[node].data 
        recons_wp[indx] = new_wp_1.reconstruct(update=False)
    return np.float32(recons_wp[:,int(trim_point/2):-int(trim_point/2)])#change it


def wpte_window_fast(data,win_size=4000,win_inc=160):
    K, L, M = np.shape(data)
    
    num_win = int(np.floor((M - win_size)/win_inc)+1)
    wpte = np.empty((K,L,516,num_win),dtype='float32')
    wpte_temp = np.empty((516,M),dtype='float32')
    for chan in range(K):
        for epoch in range(L):
            wpte_temp[:,:] = wpte_recon(data[chan,epoch,:])
            wpte_temp_power = np.float32(np.power(wpte_temp,2))
            indexer = np.arange(int(win_size)).reshape(1, -1) + int(win_inc) * np.arange(num_win).reshape(-1, 1)
            recons_wins = wpte_temp_power[:,indexer]
            # print(recons_wins)
            wpte_mov = np.mean(recons_wins,axis=2)
            # print(wpte_mov)    
            wpte[chan,epoch,:,:]=wpte_mov
            
    return wpte


#%% wpte test

import time
start_time = time.time()

# sampd = data[0:3,0:3,:]

sampd = data[0:2,0:2,:]

sampd = np.float32(sampd)

wpte_test = wpte_window_fast(sampd)

print("--- %s seconds ---" % (time.time() - start_time))

plt.plot(wpte_test[0,0,10,:])

#%% test

# import time
# start_time = time.time()

# sampd = data[0:1,0:1,:]

# wpte = wpte_window(sampd)

# print("--- %s seconds ---" % (time.time() - start_time))

#%% zscore corrected

# plt.plot(stats.zscore(wpte[2,1,44,:], axis=0, ddof=1))

#%% calculate wpte

import time
start_time = time.time()

wpte = wpte_window_fast(data)

print("--- %s seconds ---" % (time.time() - start_time))
#%% save it

wpte = np.float32(wpte)


np.save('C:/WNencki/processing/dbs_macro/dbs_macro_wpte/DBS_sub_'+sub_name+'_macro_4khz_9sec_'+'ENCODE'+'_filt_wpte',wpte)