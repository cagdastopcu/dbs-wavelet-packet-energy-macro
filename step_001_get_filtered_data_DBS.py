# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 12:29:46 2022
copied 15/03/2022 to debug and refactoring

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

#%% load converted file 8 kHz
# data=pots_d, ch_pos=ele_pos, ch_names=channel_names_clean, ch_names_mean=channel_names_df

sub_name_list = ['210319','210413','210505','210527',
            '210708','210805','210909']

sub_name = sub_name_list[6]

data = sio.loadmat('C:/WNencki/processing/dbs_memory/DBS_sub_'+sub_name+'_macro_4khz.mat')

data_macro = np.squeeze(data['data_macro'])

# delete_data = data_macro[:,:,::5];

# data_macro = np.setdiff1d(data_macro,delete_data)
data = data_macro[:,:,:-1]


#%% functions



#%%

Fs = 4000
NFFT = Fs

# aa = np.array([])
# for idx, x in np.ndenumerate(data[94,:,:]):
#     for y in np.ndenumerate(x):
#     # aa[idx,idy] = np.mean(x)
#         print(y)

notch_freq = 50.0  # Frequency to be removed from signal (Hz)
quality_factor = 30.0  # Quality factor
b_notch, a_notch = iirnotch(notch_freq, quality_factor, Fs)

data_filt = filtfilt(b_notch, a_notch, data)


notch_freq = 100.0
b_notch, a_notch = iirnotch(notch_freq, quality_factor, Fs)

data_filt = filtfilt(b_notch, a_notch, data_filt)

notch_freq = 150.0
b_notch, a_notch = iirnotch(notch_freq, quality_factor, Fs)

data_filt = filtfilt(b_notch, a_notch, data_filt)

notch_freq = 200.0
b_notch, a_notch = iirnotch(notch_freq, quality_factor, Fs)

data_filt = filtfilt(b_notch, a_notch, data_filt)


notch_freq = 250.0
b_notch, a_notch = iirnotch(notch_freq, quality_factor, Fs)

data_filt = filtfilt(b_notch, a_notch, data_filt)

notch_freq = 350.0
b_notch, a_notch = iirnotch(notch_freq, quality_factor, Fs)

data_filt = filtfilt(b_notch, a_notch, data_filt)

b,a = butter(4, [1/(Fs/2), 502/(Fs/2)], btype ='bandpass')

# data_filt_words = np.zeros([180,40000])


data_filt_wbp = filtfilt(b,a, data_filt)

#np.save('sub-005_4kHz_5sec_'+'ENCODE'+'_filt',data_filt_wbp)

#%%

data_filt_wbp = np.float32(data_filt_wbp)

np.save('C:/WNencki/processing/dbs_macro/dbs_macro_filtered/DBS_sub_'+sub_name+'_macro_4khz_9sec_'+'ENCODE'+'_filt',data_filt_wbp)

# np.save('DBS_sub_210319_macro_4khz_5sec_'+'ENCODE'+'_32b_filt',data_filt_wbp)


#%% test filter plots

# plt.plot(data[1,1,10:2000])

# plt.plot(data_filt_wbp[1,1,10:2000])

# plt.show()