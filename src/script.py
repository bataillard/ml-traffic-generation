# -*- coding: utf-8 -*-
'''Pipeline and helper functions'''

# --- IMPORTS ---
import os
import datetime
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

# --- CONSTANTS AND LIBRARY SETUP ---
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# --- DATA LOADING ---
def load_dataframe(path, cols_map):
    ''' Load the data into a pandas dataframe '''
    return NotImplementedError

# --- DATA PREPARATION ---
def split_norm_data(data, return_mean_std=False, splitter=None, mean_std_scalars=False):
    ''' Returns a 70/20/10 split for normalized training, validating, & testing data.
        If `return_mean_std` is True, also returns the mean and std of the training set'''
    n = len(data.index)
    # Split
    if splitter:
        train, val, test = splitter(data)
    else:
        train, val, test = data[0:int(n*0.7)], data[int(n*0.7):int(n*0.9)], data[int(n*0.9):]
       
    # Normalize
    if mean_std_scalars:
        mean, std = train.values.mean(), train.values.std()
    else: 
        mean, std = train.mean(), train.std()
    
    train = (train - mean) / std
    val = (val - mean) / std
    test = (test - mean) / std
    
    if return_mean_std:
        return train, val, test, mean, std
    else:
        return train, val, test
    
def year_splitter(data, val_years=2, test_years=2):
    """Takes a time series and returns a split into training, validation, and test dataset.
    The split respects the ordering of the data, so the training set happens before the validation set,
    which is before the test set.
    The test and validation will be of length `test_year` and `val_years` respectively (in years).
    The training set is any remaining data"""
    # Consider year as discrete number of weeks
    weeks_in_year = 52
    days_in_year = 7 * weeks_in_year
    year_delta = pd.Timedelta(f"{days_in_year} days")
    
    test_split = data.index.max() - test_years * year_delta
    val_split = test_split - val_years * year_delta
    
    return data[:val_split], data[val_split:test_split], data[test_split:]

# Add sin/cos periodicity columns to features dataframe
def add_time_period_cols(data, time_length, time_string):
    """Takes a time series, a period in seconds, and returns a column name.
    Returns a copy of the dataset with the sin and cos of the timeseries index along the
    given period"""
    data = data.copy()
    
    # Transform datetime index to seconds
    timestamp_s = data.index.map(pd.Timestamp.timestamp)
    
    data[time_string + '_sin'] = np.sin(timestamp_s * (2 * np.pi / time_length))
    data[time_string + '_cos'] = np.cos(timestamp_s * (2 * np.pi / time_length))
    
    return data



# --- WindowGenerator CLASS ---
class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df, mean, std,
                 label_columns=None, zero_column='n_vehicles'):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.mean = mean[zero_column]
        self.std = std[zero_column]

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}
        self.zero_column_index = train_df.columns.get_loc(zero_column)

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
    
    
    def set_example(self, inputs, labels):
        self._example = inputs, labels
        
    def test_function(self):
        print('Test')
    
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        
        return inputs, labels
    
    
    def plot(self, model=None, plot_col='n_vehicles', max_subplots=3):
        inputs, labels = self.example
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        
        fig, axs = plt.subplots(max_n, 2, figsize=(12, 8), sharey='row', sharex='col')

        for n in range(max_n):
            ax_in = axs[n, 0]
            
            ax_in.set_ylabel(f'{plot_col} [normed]')
            ax_in.plot(self.input_indices, inputs[n, :, plot_col_index], 
                       label='Inputs', marker='.', zorder=-10)
            
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index
                
            if label_col_index is None:
                continue
            
            ax_out = axs[n, 1]
            
            ax_out.plot(self.label_indices, labels[n, :, label_col_index],
                        label='Labels', color='#2ca02c', marker='.', ms=10)
            if model is not None:
                predictions = model(inputs)
                ax_out.scatter(self.label_indices, predictions[n, :, label_col_index], 
                               marker='X', label='Predictions', color='#ff7f0e')
            
            if n == 0:
                ax_in.legend()
                ax_out.legend()
                
            ax_in.set_xlabel('Time [h]')
            ax_out.set_xlabel('Time [h]')
            
        return fig
    
       
    def make_dataset(self, data, training=True):
        
        def filter_zero_weeks(window):
            start = window[0:self.input_width, self.zero_column_index] * self.std + self.mean
            end = window[self.label_start:self.label_start + self.label_width,
                        self.zero_column_index] * self.std + self.mean
                        
            # Both start or end must be non zero
            return tf.reduce_any(tf.abs(start) >= 1e-4) and tf.reduce_any(tf.abs(end) >= 1e-4)

        batch_size = 32 if training else 52
        sequence_stride = 1 if training else self.input_width
        shuffle = training
        
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data, targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=sequence_stride, shuffle=shuffle,
            batch_size=batch_size,)
        
        if training:
            # Remove periods where sensors output zero => consider as invalid data
            ds = ds.unbatch()
            ds = ds.filter(filter_zero_weeks)
            ds = ds.batch(batch_size)
        
        ds = ds.map(self.split_window)
        
        return ds
    
    @property
    def train(self):
        return self.make_dataset(self.train_df)
    
    @property
    def val(self):
        return self.make_dataset(self.val_df)
    
    @property
    def test(self):
        return self.make_dataset(self.test_df)
    
    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

# --- WINDOWS ---
def single_step_window(train, val, test, label_cols=['n_vehicles']):
    return WindowGenerator(
        train_df=train, val_df=val, test_df=test,
        input_width=1, label_width=1, shift=1,
        label_columns=label_cols)

def wide_window(train, val, test, label_cols=['n_vehicles']):
    return WindowGenerator(input_width=24, label_width=24, shift=1, label_columns=label_cols,
                           train_df=train, val_df=val, test_df=test)

def conv_window(train, val, test, input_w=1, label_w=1, label_cols=['n_vehicles']):
    return WindowGenerator(input_width=input_w, label_width=label_w, shift=1, label_columns=label_cols,
                            train_df=train, val_df=val, test_df=test)

# Better function easier to understand in code (just use constructor otherwise lol)
def make_window(train, val, test, mean, std,
                input_w=1, label_w=1, shift=1, label_cols=['n_vehicles']):
    return WindowGenerator(input_width=input_w, label_width=label_w, shift=shift, label_columns=label_cols, 
                          train_df=train, val_df=val, test_df=test, mean=mean, std=std)

# --- MODELS ---
def compile_and_fit(model, data, patience=2, data_is_window=True,
                    metric=tf.metrics.MeanAbsoluteError()):
    MAX_EPOCHS = 20
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[metric])
    
    if data_is_window:
        train, val = data.train, data.val
    else:
        train, val = data

    history = model.fit(train, epochs=MAX_EPOCHS,
                        validation_data=val,
                        callbacks=[early_stopping])
    return history

class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]
    
def linear_model(output_layers=1):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(units=output_layers)
    ])

def dense_model(output_layers=1):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=output_layers)
    ])

def multi_step_dense_model():
    return tf.keras.Sequential([
        # Shape: (time, features) => (time*features)
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
        # Add back the time dimension.
        # Shape: (outputs) => (1, outputs)
        tf.keras.layers.Reshape([1, -1]),
    ])
    
def conv_model(conv_width, output_layers=1):
    return tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32,
                               kernel_size=(conv_width,),
                               activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=output_layers),
    ])

def lstm_model(num_features):
    return tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=num_features)
    ])