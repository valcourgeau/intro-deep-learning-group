# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 17:17:31 2017

@author: PierFrancesco
"""

import os
import numpy as np
from matplotlib import pyplot as plt

path = 'C:\\Users/PierFrancesco/Documents/Python/UCL/Intro to DeepLearning/Projectwork'
os.chdir(path)
data = np.genfromtxt('s_and_p.csv', delimiter=',')
daily_pr = data[3:,1:]

ret_d = np.diff(np.log(daily_pr))
ret_d[np.isnan(ret_d)] = 0 #Nan values set to 0. it will correspond to carry over when cumulating
ret_w = np.sum(np.reshape(ret_d, (int(ret_d.shape[0]/5),5,ret_d.shape[1])), axis=1)

ret_w_sliding = np.zeros([ret_w.shape[0]-51,52,ret_w.shape[1]])
ret_d_sliding = np.zeros([ret_d.shape[0]-19,20,ret_d.shape[1]])

for i in range(0, len(ret_w)-51):
    ret_w_sliding[i,:,:] = ret_w[i:i+52,:]
     
for i in range(0, len(ret_d)-19):
    ret_d_sliding[i,:,:] = ret_d[i:i+20,:]

ret_w_sliding = ret_w_sliding[0:-4,:,:]
ret_d_sliding = ret_d_sliding[((52*5)):,:,:]
ret_d_sliding = ret_d_sliding[np.arange(0,ret_d_sliding.shape[0],5),:,:]

cumRet_d = np.cumsum(ret_d_sliding,1)
cumRet_w = np.cumsum(ret_w_sliding,1)

mean_d = np.mean(cumRet_d,2)
mean_w = np.mean(cumRet_w,2)
std_d = np.std(cumRet_d,2)
std_w = np.std(cumRet_w,2)

d_Zscore = (cumRet_d - np.tile(mean_d[:,:,np.newaxis],(1,1,cumRet_d.shape[2]))) / np.tile(std_d[:,:,np.newaxis],(1,1,cumRet_d.shape[2]))
w_Zscore = (cumRet_w - np.tile(mean_w[:,:,np.newaxis],(1,1,cumRet_w.shape[2]))) / np.tile(std_w[:,:,np.newaxis],(1,1,cumRet_w.shape[2]))


Y_w = ret_w[55:,:]
Y_median = np.median(Y_w,1)
Y_w[Y_w>=np.tile(Y_median[:,np.newaxis],347)] = 1
Y_w[Y_w<np.tile(Y_median[:,np.newaxis],347)] = 0
 
X_w = w_Zscore
X_d = d_Zscore

X_train = X_w[0:400,:,:]
X_valid = X_w[401:,:,:]

y_train0 = Y_w[0:400,:]
y_valid0 = Y_w[401:,:]

