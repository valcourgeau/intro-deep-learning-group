# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 09:52:39 2017

@author: PierFrancesco
"""
import numpy as np
from matplotlib import pyplot as plt

''' Import Data '''
Prediction = np.load('Prediction.npy')
ret_w = np.load('ret_w.npy')
index_ret_w = np.load('index_ret_w.npy')
dates = np.load('dates_w.npy')
dates = dates[751+55:]

ret_w = ret_w[751+55:,:] #Returns corresponding to validation set
index_ret_w = index_ret_w[751+55:]

''' Consider only prediction above treshold '''
port_long = Prediction[:,:,0]>0.55
port_short = Prediction[:,:,1]>0.80 #short difficult to forecast...

''' build an equally weighted portfolio for prediction above median '''
port_long = np.swapaxes(port_long,0,1)
N_long_pos = np.sum(port_long,1)
N_long_pos[N_long_pos==0] = 1
port_long = port_long/N_long_pos[:,np.newaxis]

''' build an equally weighted portfolio for prediction below median '''
port_short = np.swapaxes(port_short,0,1)
N_short_pos = np.sum(port_short,1)
N_short_pos[N_short_pos==0] = 1
port_short = port_short/N_short_pos[:,np.newaxis]

long_strategy = np.sum(ret_w*(port_long),1)
short_strategy = np.sum(-ret_w*port_short,1)


''' Plot Results '''
f, (ax1) = plt.subplots(1, 1, sharey=False)
ax1.plot(np.cumsum(long_strategy[50:])+np.cumsum(short_strategy[50:]), label='Strategy')
ax1.plot(np.cumsum(index_ret_w[50:]), label='S&P 500 Index')
ax1.grid(linestyle='--')
ax1.set_ylabel('Return', color='b')
ax1.set_xlabel('Time', color='b')
ax1.legend()