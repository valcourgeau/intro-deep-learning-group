import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
import keras

# with tf.Session() as sess:
#         latent_val = hidden4.eval(feed_dict={X: x_train})

# convert an array of values into a dataset matrix
def create_dataset(trainX, trainY, look_back=1):
    dataX, dataY = [], []
    for i in range(len(trainX)-look_back-1):
        a = trainX[i:(i+look_back),:]
        dataX.append(a)
        dataY.append(trainY[i + look_back])
    return np.array(dataX), np.array(dataY)

latent_val = pd.read_csv('sae_output.csv')
latent_val = np.array(latent_val)
y_lazy = pd.read_csv('data/sp500_index_data.csv')['Close Price']
y_train = y_lazy.shift(-1)
y_train = np.array(y_train)

#this means we have an na for the last value training is latent_val, target y_train

look_back =4
trainX = latent_val[:-1]
trainY = np.array(y_train[:-1])
trainX, trainY = create_dataset(trainX, trainY, look_back)

# create and fit the LSTM network
opti_adam = keras.optimizers.Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model = Sequential()

model.add(LSTM(5, input_shape=(look_back, 10)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=opti_adam)
model.fit(trainX, trainY, epochs=1000, batch_size=60, verbose=2)