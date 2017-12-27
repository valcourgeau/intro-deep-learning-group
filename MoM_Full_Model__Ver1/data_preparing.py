# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 17:17:31 2017

@author: PierFrancesco
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime

path = 'C:\\Users/PierFrancesco/Documents/Python/UCL/Intro to DeepLearning/Projectwork'
os.chdir(path)

''' Import and save dates '''
dates = np.loadtxt("index_price.csv", delimiter=",", usecols=(0), dtype=object)
dates = dates[1:]
dates_w = dates[0:-1:5]
np.save('dates_w',dates_w)


''' Import Components and compute returns '''
data = np.genfromtxt('s_and_p.csv', delimiter=',')
daily_pr = data[1:,1:]
N_stock = daily_pr.shape[1]

ret_d = np.diff(np.log(daily_pr),axis=0)
ret_d[np.isnan(ret_d)] = 0 #Nan values set to 0. it will correspond to carry over when cumulating
ret_w = np.sum(np.reshape(ret_d, (int(ret_d.shape[0]/5), 5, N_stock)), axis=1)

''' Import Index and compute returns '''
index = np.genfromtxt('index_price.csv', delimiter=',')
index_pr = index[0:,1:]
index_ret_d = np.diff(np.log(index_pr),axis=0)
index_ret_d[np.isnan(index_ret_d)] = 0 #Nan values set to 0. it will correspond to carry over when cumulating
index_ret_w = np.sum(np.reshape(index_ret_d, (int(index_ret_d.shape[0]/5), 5)), axis=1)


''' Create 3D Tensor of sliding windows '''
ret_w_sliding = np.zeros([ret_w.shape[0]-51,52,ret_w.shape[1]])
ret_d_sliding = np.zeros([ret_d.shape[0]-19,20,ret_d.shape[1]])

for i in range(0, len(ret_w)-51):
    ret_w_sliding[i,:,:] = ret_w[i:i+52,:]
     
for i in range(0, len(ret_d)-19):
    ret_d_sliding[i,:,:] = ret_d[i:i+20,:]

ret_w_sliding = ret_w_sliding[0:-4,:,:]
ret_d_sliding = ret_d_sliding[((52*5)):,:,:]
ret_d_sliding = ret_d_sliding[np.arange(0,ret_d_sliding.shape[0],5),:,:]


''' Compute cumulative returns (over rows) '''
cumRet_d = np.cumsum(ret_d_sliding,1)
cumRet_w = np.cumsum(ret_w_sliding,1)

''' Compute cross-section mean and standard deviation '''
mean_d = np.mean(cumRet_d,2)
mean_w = np.mean(cumRet_w,2)
std_d = np.std(cumRet_d,2)
std_w = np.std(cumRet_w,2)

''' Compute cross-sectional Z-scores (normalization) '''
d_Zscore = (cumRet_d - np.tile(mean_d[:,:,np.newaxis],(1,1,cumRet_d.shape[2]))) / np.tile(std_d[:,:,np.newaxis],(1,1,cumRet_d.shape[2]))
w_Zscore = (cumRet_w - np.tile(mean_w[:,:,np.newaxis],(1,1,cumRet_w.shape[2]))) / np.tile(std_w[:,:,np.newaxis],(1,1,cumRet_w.shape[2]))


''' Compute output labels '''
Y_w = np.ones(ret_w[55:,:].shape) 
Y_w = Y_w*ret_w[55:,:] # 1° year is eliminated
Y_median = np.median(Y_w,1) # cmpute cross-sectional median
Y_w[Y_w>=np.tile(Y_median[:,np.newaxis],N_stock)] = 1 # 1 <- returns above the median
Y_w[Y_w<np.tile(Y_median[:,np.newaxis],N_stock)] = 0 # 0 <- returns above the median
 
''' Rename and define training and test set '''
X_w = w_Zscore
X_d = d_Zscore

X_train0 = X_w[0:800,:,:]
X_valid0 = X_w[801:,:,:]

y_train0 = Y_w[0:800,:]
y_valid0 = Y_w[801:,:]

''' Save variables to be used to run the architecture '''
np.save('X_train0',X_train0)
np.save('X_valid0',X_valid0)
np.save('y_train0',y_train0)
np.save('y_valid0',y_valid0)

np.save('ret_w',ret_w)
np.save('index_ret_w',index_ret_w)