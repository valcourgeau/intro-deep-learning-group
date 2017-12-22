# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 19:02:03 2017

@author: PierFrancesco
"""

import numpy as np
import scipy.io
from matplotlib import pyplot as plt
 
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, BatchNormalization
from keras.layers.core import Permute, Reshape

np.random.seed(5)
''' Load data set '''

X_train0 = np.load('X_train0.npy')
X_valid0 = np.load('X_valid0.npy')
y_train0 = np.load('y_train0.npy')
y_valid0 = np.load('y_valid0.npy')


''' Inputs Definition '''
dropout_rate = 0.5
num_classes = 2
_, win_len, dim = X_train0.shape

num_feat_map = 16
batch_size = 128
num_hidden_lstm = 32
epochs = 5

 

def _data_reshaping(X_tr, X_va):
    _, win_len, dim = X_tr.shape         
    # make it into (frame_number, dimension, window_size, channel=1) for convNet
    X_tr = np.swapaxes(X_tr,1,2)
    X_va = np.swapaxes(X_va,1,2)
 
    X_tr = np.reshape(X_tr, (-1, dim, win_len, 1))
    X_va = np.reshape(X_va, (-1, dim, win_len, 1))
    return X_tr, X_va

X_train, X_valid = _data_reshaping(X_train0, X_valid0)
       

Prediction = np.zeros([dim,len(X_valid),2])
F1 = np.zeros([dim,1])


''' prediction for each stock '''

for i in range(dim): 
      
    y_train = keras.utils.to_categorical(y_train0[:,i], num_classes)
    y_valid = keras.utils.to_categorical(y_valid0[:,i], num_classes)

    print('\n\nstock number:',i,'\n')
    
    
    '''Building architecture '''
    model = Sequential()  
    
    model.add(Conv2D(num_feat_map, (3, 3), padding='same',activation='elu', input_shape=(dim, win_len, 1)))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(dropout_rate))
    #model.add(BatchNormalization())
    #model.add(Conv2D(16, (3, 3), padding='same',activation='elu'))
    #model.add(MaxPooling2D(pool_size=(1, 2)))
    #model.add(Dropout(dropout_rate))
    
    model.add(Permute((2, 1, 3))) # for swap-dimension
    model.add(Reshape((-1,num_feat_map*dim)))
    #model.add(LSTM(32, return_sequences=True, stateful=False))
    #model.add(Dropout(dropout_rate))    
    model.add(LSTM(32, return_sequences=False, stateful=False))
    model.add(Dropout(dropout_rate))
    #model.add(BatchNormalization())
    
    model.add(Dense(2, activation='softmax'))
    
#    model.summary()
    model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
    h = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True, validation_data=(X_valid, y_valid))
    Prediction[i,:,:] = model.predict(X_valid)
    
    del model
    del h
    
np.save('Prediction',Prediction)

    
'''
f, (ax1, ax2) = plt.subplots(2, 1, sharey=False)
ax1.plot(h.history['acc'],'b')
ax1.grid(linestyle='--')
ax1.set_ylabel('Train Accuracy', color='b')
ax1b = ax1.twinx()
ax1b.plot(h.history['loss'],'g')
ax1b.set_ylabel('Train Loss', color='g')

ax2.plot(h.history['val_acc'],'r')
ax2.grid(linestyle='--')
ax2.set_ylabel('Test Accuracy', color='r')
ax2.set_xlabel('Epoch')
ax2b = ax2.twinx()
ax2b.plot(h.history['val_loss'],'g')
ax2b.set_ylabel('Val Loss', color='g')
'''

'''
a = model.predict(X_valid)
y_pred = np.argmax(a, axis=1)
y_true = np.argmax(y_valid, axis=1)
class_wise_f1 = np.round(f1_score(y_true, y_pred, average=None)*100)*0.01
print(np.mean(class_wise_f1))
F1[i,0] = np.mean(class_wise_f1)
'''
