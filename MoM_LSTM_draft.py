# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 19:02:03 2017

@author: PierFrancesco
"""
import os
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
 
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten
from keras.layers.core import Permute, Reshape
from sklearn.metrics import f1_score
    
 
dropout_rate = 0.1
num_classes = 2
_, win_len, dim = X_train.shape
         
batch_size = 128
num_hidden_lstm = 64
epochs = 3

Prediction = np.zeros([dim,len(X_valid),2])
F1 = np.zeros([dim,1])
for i in range(0,dim): 
   
    #Building LSTM architecture 
    print(i)
    model = Sequential()  
    
    y_train = keras.utils.to_categorical(y_train0[:,i], num_classes)
    y_valid = keras.utils.to_categorical(y_valid0[:,i], num_classes)
    
    model.add(LSTM(num_hidden_lstm, input_shape=(win_len,dim), return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(num_hidden_lstm, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    
#    model.summary()
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])
     
    H = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True, validation_data=(X_valid, y_valid))
    
    Prediction[i,:,:] = model.predict(X_valid)
    a = model.predict(X_valid)
    y_pred = np.argmax(a, axis=1)
    y_true = np.argmax(y_valid, axis=1)
    class_wise_f1 = np.round(f1_score(y_true, y_pred, average=None)*100)*0.01
    print(np.mean(class_wise_f1))
    F1[i,0] = np.mean(class_wise_f1)
    
    
