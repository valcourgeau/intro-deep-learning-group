import tensorflow as tf
import numpy as np
import numpy.random as rnd
import os
import sys
import time

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()
from functools import partial

def train_autoencoder(X_train, n_neurons, n_epochs, batch_size,
                      learning_rate = 0.01, l2_reg = 0.0005,
                      activation=tf.nn.elu, seed=42):
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(seed)

        n_inputs = X_train.shape[1]

        X = tf.placeholder(tf.float32, shape=[None, n_inputs])
        
        # Change the initiatlisation for sigmoids?
        #kernel_init = tf.contrib.layers.variance_scaling_initializer()
        kernel_init = tf.contrib.layers.xavier_initializer(uniform=False,
                                                           seed=seed)
        my_dense_layer = partial(
            tf.layers.dense,
            activation=activation,
            kernel_initializer=kernel_init,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))

        hidden = my_dense_layer(X, n_neurons, name="hidden")
        outputs = my_dense_layer(hidden, n_inputs, activation=None, name="outputs")

        reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([reconstruction_loss] + reg_losses)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        init.run()
        for epoch in range(n_epochs):
            n_batches = len(X_train) // batch_size
            for iteration in range(n_batches):
                #print("\r{}%".format(100 * iteration // n_batches), end="")
                sys.stdout.flush()
                indices = rnd.permutation(len(X_train))[:batch_size]
                X_batch = X_train[indices]
                sess.run(training_op, feed_dict={X: X_batch})
            loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})
            if epoch % 50 == 0:
                print("{}".format(epoch), "Train MSE:", loss_train)
        params = dict([(var.name, var.eval()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
        hidden_val = hidden.eval(feed_dict={X: X_train})
        return hidden_val, params["hidden/kernel:0"], params["hidden/bias:0"], params["outputs/kernel:0"], params["outputs/bias:0"]

# Retrieving the data created with WT
x_train = np.genfromtxt('data/data_wt/sp500_wt.csv', delimiter=',', dtype=None, names=True)
x_train = [list(item) for item in x_train]
x_train = np.array(x_train)[:,1:] # remove Ntime
print(x_train.shape)


def normalise_dataset(dataset):
    n_cols = dataset.shape[1]
    norm_dataset = np.zeros_like(dataset)
    for i in range(n_cols): 
        mean = np.sum(dataset[:,i])/len(dataset[:,i])
        std_dev = np.dot(dataset[:,i]-mean, dataset[:,i]-mean)/(len(dataset[:,i]) - 1)
        std_dev = np.sqrt(std_dev)
        norm_dataset[:,i] = (dataset[:,i] - mean) / std_dev

    return norm_dataset

# Normalised dataset
x_train = normalise_dataset(x_train)

# Following the paper's instructions
BATCH_SIZE = 60
N_EPOCHS = 500
ACTIVATION = tf.nn.sigmoid
LEARNING_RATE = 0.05



start_time = time.clock()

# Should we not incerrase the learning_rate for the first one?
# of number of neurons, etc?
hidden_output1, W1, b1, W8, b8 = train_autoencoder(x_train, 
                                                   n_neurons=50, 
                                                   n_epochs=N_EPOCHS, 
                                                   batch_size=BATCH_SIZE, 
                                                   learning_rate=LEARNING_RATE,
                                                   activation=ACTIVATION)
print(hidden_output1[1,:])
print("{} seconds".format(time.clock() - start_time))
print("-------------------------------")
hidden_output2, W2, b2, W7, b7 = train_autoencoder(hidden_output1, 
                                                   n_neurons=40, 
                                                   n_epochs=N_EPOCHS, 
                                                   batch_size=BATCH_SIZE,
                                                   learning_rate=LEARNING_RATE,
                                                   activation=ACTIVATION)
print(hidden_output2[1,:])
print("{} seconds".format(time.clock() - start_time))
print("-------------------------------")
hidden_output3, W3, b3, W6, b6 = train_autoencoder(hidden_output2, 
                                                   n_neurons=20, 
                                                   n_epochs=N_EPOCHS, 
                                                   batch_size=BATCH_SIZE,
                                                   learning_rate=LEARNING_RATE,
                                                   activation=ACTIVATION)
print(hidden_output3[1,:])
print("{} seconds".format(time.clock() - start_time))
print("-------------------------------")
output4, W4, b4, W5, b5 = train_autoencoder(hidden_output3, 
                                      n_neurons=10, 
                                      n_epochs=N_EPOCHS, 
                                      batch_size=BATCH_SIZE,
                                      learning_rate=LEARNING_RATE,
                                      activation=ACTIVATION)

print("{:2f} seconds".format(time.clock() - start_time))

def save_as_csv(data_array, file_name="data"):
    """
        Create .csv file with given name from given NumPy data array
        in the current directory.
        
        Args:
            data_array: NumPy structures data array
            file_name: name of .csv file to be created
            
        Returns:
            Nothing but creates a file in the current directory.
    """
    output_name = file_name + ".csv"
    np.savetxt(output_name, data_array, delimiter=",", fmt='%-7.4f', newline='\n')

# Normalise output of SAEs
output4 = normalise_dataset(output4)

# Save output in .csv file
save_as_csv(output4, "sae_output")