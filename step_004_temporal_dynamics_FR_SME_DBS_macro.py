# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 17:30:33 2022

@author: cagdas
"""
import matplotlib.pyplot as plt
import numpy as np 
import pylab as py 
import matplotlib.colors as mcolors

import pandas as pd
# from kcsd.KCSD import oKCSD3D
from mayavi import mlab
import nibabel as nib

from scipy.signal import filtfilt, butter, iirnotch, spectrogram, hilbert, resample

from scipy.fft import fftshift

from scipy import stats

import scipy.io as sio

import matplotlib.pyplot as plt

import pywt

from dtw import *

#%% data

subNames = ['210319','210413','210505','210527',
            '210708','210805','210909']

wpte_theta = np.zeros((7,120,6,201))
wpte_HG = np.zeros((7,120,6,201))
# wpte_HF = np.zeros((7,120,6,107))
# wpte_VHF = np.zeros((7,120,6,107))
recall_tt = np.empty((7,120))
# recall_idx = np.empty((7,120))
# forget_idx = np.empty((7,120))

for idx in range(7):

    sub_name = subNames[idx]

    wpte_32 = np.load('C:/WNencki/processing/dbs_macro/dbs_macro_wpte/DBS_sub_'+sub_name+'_macro_4khz_9sec_'+'ENCODE'+'_filt_wpte.npy')


#% zscore

    # wpte_zsc = stats.zscore(wpte_32[:,:,:,:], axis=3, ddof=1)

    wpte_theta[idx,:,:,:] = np.mean(wpte_32[:,:,1:10,:],axis=2);
    
    wpte_HG[idx,:,:,:] = np.mean(wpte_32[:,:,62:124,:],axis=2);
    
    data = sio.loadmat('C:/WNencki/processing/dbs_memory/DBS_sub_'+sub_name+'_recall_tt.mat')

    recall_tt[idx,:] = np.squeeze(data['recall_tt'])

    # recall_idx[idx,:] = np.argwhere(recall_tt > 0)
    # forget_idx[idx,:] = np.argwhere(recall_tt < 0)
    
    



#%% theta

data_recall = np.empty((7,6,201))
data_forget = np.empty((7,6,201))

wpte_zsc_theta = stats.zscore(wpte_theta[:,:,:,:], axis=3, ddof=1)

data = wpte_zsc_theta;

t_macro = np.linspace(-3.5, 5.5, num=201)

for idx in range(7):
    
    recall_idx = np.argwhere(recall_tt[idx,:] > 0)
    data_recall[idx,:,:] = np.squeeze(np.mean(data[idx,recall_idx,:,:],axis=0))
    data_forget[idx,:,:] = np.squeeze(np.mean(data[idx,np.argwhere(recall_tt[idx,:] < 0),:,:],axis=0))
    



# chan_no = 0



# data_mean = np.squeeze(np.mean(data,axis=1))

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 5}

plt.rc('font', **font)

fig, axs = plt.subplots(6,7, figsize=(12, 10), dpi=300)
# plt.figure(figsize=(4, 3), dpi=600)

idx_chan_reverse = [5, 4, 3, 2, 1, 0]

for idx_chan in range(6):
    for idx_pat in range(7):
            
            axs[idx_chan_reverse[idx_chan],idx_pat].plot(t_macro[66:136],data_recall[idx_pat,idx_chan,66:136]-data_forget[idx_pat,idx_chan,66:136],color='g')
            # axs[idx_chan_reverse[idx_chan],idx_pat].plot(t_macro,data_forget[idx_pat,idx_chan,:],color='b')
            
            axs[idx_chan_reverse[idx_chan],idx_pat].axvline(x=0,color='k',linewidth=1)
            axs[idx_chan_reverse[idx_chan],idx_pat].axvline(x=1.5,color='k',linewidth=1)
            # axs[idx_pat,idx_chan].set_ylim()
            
            axs[idx_chan_reverse[idx_chan],idx_pat].set_xlim(t_macro[66], t_macro[135])
            axs[idx_chan_reverse[idx_chan],idx_pat].set_ylim(-0.7, 0.7)
            axs[idx_chan_reverse[idx_chan],idx_pat].set_yticks([])
            if idx_pat == 0:
                axs[idx_chan_reverse[idx_chan],idx_pat].set_yticks((-0.5,0,0.5))

plt.show()

plt.savefig('C:/WNencki/processing/dbs_macro/dbs_macro_results/temporal_dynamics_FR_SME/DBS_sub_macro_temporal_dynamics_FR_SME_theta.png', bbox_inches = 'tight', pad_inches = 0.05,dpi=600)


#%% HG

data_recall = np.empty((7,6,201))
data_forget = np.empty((7,6,201))

wpte_zsc_HG = stats.zscore(wpte_HG[:,:,:,:], axis=3, ddof=1)

data = wpte_zsc_HG;

t_macro = np.linspace(-3.5, 5.5, num=201)

for idx in range(7):
    
    recall_idx = np.argwhere(recall_tt[idx,:] > 0)
    data_recall[idx,:,:] = np.squeeze(np.mean(data[idx,recall_idx,:,:],axis=0))
    data_forget[idx,:,:] = np.squeeze(np.mean(data[idx,np.argwhere(recall_tt[idx,:] < 0),:,:],axis=0))
    



# chan_no = 0



# data_mean = np.squeeze(np.mean(data,axis=1))

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 5}

plt.rc('font', **font)

fig, axs = plt.subplots(6,7, figsize=(12, 10), dpi=300)
# plt.figure(figsize=(4, 3), dpi=600)

idx_chan_reverse = [5, 4, 3, 2, 1, 0]

for idx_chan in range(6):
    for idx_pat in range(7):
            
            axs[idx_chan_reverse[idx_chan],idx_pat].plot(t_macro[66:136],data_recall[idx_pat,idx_chan,66:136]-data_forget[idx_pat,idx_chan,66:136],color='g')
            # axs[idx_chan_reverse[idx_chan],idx_pat].plot(t_macro,data_forget[idx_pat,idx_chan,:],color='b')
            
            axs[idx_chan_reverse[idx_chan],idx_pat].axvline(x=0,color='k',linewidth=1)
            axs[idx_chan_reverse[idx_chan],idx_pat].axvline(x=1.5,color='k',linewidth=1)
            # axs[idx_pat,idx_chan].set_ylim()
            
            axs[idx_chan_reverse[idx_chan],idx_pat].set_xlim(t_macro[66], t_macro[135])
            axs[idx_chan_reverse[idx_chan],idx_pat].set_ylim(-1.05, 1.05)
            axs[idx_chan_reverse[idx_chan],idx_pat].set_yticks([])
            if idx_pat == 0:
                axs[idx_chan_reverse[idx_chan],idx_pat].set_yticks((-1,0,1))

plt.show()


plt.savefig('C:/WNencki/processing/dbs_macro/dbs_macro_results/temporal_dynamics_FR_SME/DBS_sub_macro_temporal_dynamics_FR_SME_HG.png', bbox_inches = 'tight', pad_inches = 0.05,dpi=600)