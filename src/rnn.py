# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from script import compile_and_fit
from script import WindowGenerator
from IPython.display import clear_output

def n_hidden_nodes(n_in, n_out, n_train=0, alpha=0):
    '''
    Good estimate for the number of hidden nodes
    
    Input:
        n_in: (int) Number of input neurons
        n_out: (int) Number of output neurons
        n_train: (int) Number of samples in training data
        alpha: Scaling factor between 2 and 10
    Output:
        n_h: Estimate for number of hidden nodes
    '''
    # Default estimate
    if (alpha == 0 or n_train == 0):
        return (2/3) * (n_in + n_out)
    
    return n_train / (alpha * (n_in + n_out))

def best_alpha(window):
    'Determine the best alpha scaling factor'
    alphas = list(range(2, 11))
    num_train = window.train_df.shape[0] # Number of training data samples
    num_features = window.train_df.shape[1] # Number of features
    # Performances
    performances = {}
    for alpha in alphas:
        n_units = n_hidden_nodes(num_features, num_features, num_train, alpha)
        model = FeedBack(units=n_units, out_steps=24)

        compile_and_fit(model, window)
        clear_output()
        performances[alpha] = model.evaluate(window.val)
    return min(performances, key=performances.get)


class FeedBack(tf.keras.Model):
    def __init__(self, units, num_features, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.num_features = num_features
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        # Converts the LSTM layer's outputs to model predictions
        self.dense = tf.keras.layers.Dense(num_features)

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the LSTM state.
        prediction, state = self.warmup(inputs)

        # Insert the first prediction.
        predictions.append(prediction)

        # Run the rest of the prediction steps.
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state,
                                    training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions
