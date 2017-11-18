import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lstm_model import generate_data, lstm_model, split_data, prepare_data
from tensorflow.contrib.learn.python.learn import learn_runner

LOG_DIR = './ops_logs'
TIMESTEPS = 5
RNN_LAYERS = [{'steps': TIMESTEPS,
               'keep_prob': 0.5,
               'activation': tf.nn.relu,
               'dropout': 0.5}]
DENSE_LAYERS = [10, 10]
TRAINING_STEPS = 100000
BATCH_SIZE = 100
PRINT_STEPS = TRAINING_STEPS / 100

params = {
    'batch_size': BATCH_SIZE,
    'optimizer': 'Adagrad',
    'n_classes': 0,
    'verbose': 1,
    'learning_rate': 0.03,
    'steps': TRAINING_STEPS
}
# regressor = tf.estimator.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS), 
#                                    params=params)



# Data preparation
x_dataset = np.genfromtxt('sae_output.csv', delimiter=',', dtype=None, names=True)
x_dataset = [list(item) for item in x_dataset]
x_dataset = np.array(x_dataset)

y_dataset = np.genfromtxt('data/sp500_index_data.csv', delimiter=',', dtype=None, names=True)
#print(y_dataset["Close_Price"])
#y_dataset = [list(item) for item in y_dataset]
y_dataset = np.array(y_dataset["Close_Price"])

X = prepare_data(x_dataset, time_steps=5, val_size=0.1, test_size=0.1)
y = prepare_data(y_dataset, time_steps=5, val_size=0.1, test_size=0.1, labels=True)


print(len(X['train']))
print(len(y['train'][1:]))

def experiment_fn(output_dir):
    # run experiment
    return tf.contrib.learn.Experiment(
        tf.estimator.Estimator(model_fn=simple_rnn,
                               model_dir=output_dir),
        features=X['train'],
        labels=y['train'][1:],
        eval_metrics={
            'rmse': tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_accuracy
            )
        }
    )

LSTM_SIZE = 5  # number of hidden layers in each of the LSTM cells

# create the inference model
def simple_rnn(features, labels, mode):
    N_OUTPUTS = 1
    N_INPUTS = len(features) - LSTM_SIZE

    # 1. configure the RNN
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE, forget_bias=1.0)
    outputs, _ = tf.contrib.rnn.static_rnn(lstm_cell, features, dtype=tf.float32)

    # slice to keep only the last cell of the RNN
    outputs = outputs[-1]
    #print 'last outputs={}'.format(outputs)

    # output is result of linear activation of last layer of RNN
    weight = tf.Variable(tf.random_normal([LSTM_SIZE, N_OUTPUTS]))
    bias = tf.Variable(tf.random_normal([N_OUTPUTS]))
    predictions = tf.matmul(outputs, weight) + bias
        
    # 2. Define the loss function for training/evaluation
    loss = tf.losses.mean_squared_error(labels, predictions)
    eval_metric_ops = {
        "rmse": tf.metrics.root_mean_squared_error(labels, predictions)
    }

    # 3. Define the training operation/optimizer
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.01,
        optimizer="Adam")

    # 4. Create predictions
    predictions_dict = {"predicted": predictions}

    # 5. return ModelFnOps
    return tf.contrib.learn.ModelFnOps(
        mode=mode,
        predictions=predictions_dict,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)





learn_runner.run(experiment_fn, 'data/')





# # Validation monitor
# validation_monitor = learn.monitors.ValidationMonitor(X['val'], y['val'],
#                                                       every_n_steps=PRINT_STEPS,
#                                                       early_stopping_rounds=1000)

# regressor.fit(X['train'], y['train'], monitors=[validation_monitor], logdir=LOG_DIR)

# # Training and prediction accuracy
# predicted = regressor.predict(X['test'])
# mse = mean_squared_error(y['test'], predicted)
# print ("Error: %f" % mse)            

# plot_predicted, = plt.plot(predicted, label='predicted')
# plot_test, = plt.plot(y['test'], label='test')
# plt.legend(handles=[plot_predicted, plot_test])                          