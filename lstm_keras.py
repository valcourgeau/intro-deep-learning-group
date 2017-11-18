import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
import keras

# convert an array of values into a dataset matrix
def create_dataset(trainX, trainY, look_back=1):
    dataX, dataY = [], []
    for i in range(len(trainX)-look_back-1):
        a = trainX[i:(i+look_back),:]
        dataX.append(a)
        dataY.append(trainY[i + look_back])
    return np.array(dataX), np.array(dataY)


# with tf.Session() as sess:
#         latent_val = hidden4.eval(feed_dict={X: x_train})

latent_val = pd.read_csv('sae_output_5000.csv')
latent_val = np.array(latent_val)
y_lazy = pd.read_csv('data/sp500_index_data.csv')['Close Price']
y_train = y_lazy.shift(-1)
y_train = np.array(y_train)

#this means we have an na for the last value training is latent_val, target y_train

look_back =4
trainX = latent_val[:-1]
trainX = trainX[:round(0.9*len(trainX))]
testX = trainX[round(0.9*len(trainX)):]

trainY = np.array(y_train[:-1])
trainY = trainY[:round(0.9*len(trainY))]
testY = trainY[round(0.9*len(trainY)):]

trainX, trainY = create_dataset(trainX, trainY, look_back)
testX, testY = create_dataset(testX, testY, look_back) 

# create and fit the LSTM network
LEARNING_RATE = 0.01
BATCH_SIZE = 100
EPOCHS = 5000
opti_adam = keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model = Sequential()

model.add(LSTM(5, input_shape=(look_back, 10)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=opti_adam)
model.fit(trainX, trainY, epochs=EPOCHS, batch_size=BATCH_SIZE,
          verbose=2, validation_data=(testX, testY))