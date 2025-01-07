# For ORPL bubble fill algorithm, source is available at:
# Source: https://github.com/mr-sheg/orpl
# Guillaume Sheehy, Fabien Picot, Frédérick Dallaire, Katherine Ember, Tien Nguyen, Kevin Petrecca, Dominique Trudel, and Frédéric Leblond "Open-sourced Raman spectroscopy data processing package implementing a baseline removal algorithm validated from multiple datasets acquired in human tissue and biofluids," Journal of Biomedical Optics 28(2), 025002 (21 February 2023). https://doi.org/10.1117/1.JBO.28.2.025002

# Script containing methods usually used in my scripts, used to decrease cell size
import os, gc
import glob
import re
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

np.set_printoptions(suppress=True, precision = 3)
import tensorflow as tf
from tensorflow.python.client import device_lib
print("Available computational components")
print(device_lib.list_local_devices())

# GPU support comes from tensorflow, so use builtin version of keras in tensorflow
# Change these depending on tf version
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Flatten, Dropout, Add, LSTM, GRU, Concatenate
from tensorflow.keras.layers import Conv1D, Input, Reshape, SpatialDropout1D, MaxPooling1D, LocallyConnected1D, ReLU
from tensorflow.keras.layers import Conv2D, SpatialDropout2D, MaxPooling2D, LocallyConnected2D, Lambda, GaussianNoise, Dot
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.activations import sigmoid, tanh, softmax, relu
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2, L1L2
import tensorflow.keras.initializers
import sklearn.metrics as metrics
#from BaselineRemovalCopy import *


from keras import backend as K

def reset_seed(SEED = 0):
    
    """Reset the seed for every random library in use (System, numpy and tensorflow)"""
    
    os.environ['PYTHONHASHSEED']=str(SEED)
    np.random.seed(SEED)
    
    # Check tf version. some versions may have a different seed method!
    tf.random.set_seed(SEED)

def root_mean_squared_error(y_true, y_pred):
    
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

# coefficient of determination (R^2) for regression  (only for Keras tensors)
def r_square(y_true, y_pred):

    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# Edited R^2 function with minimum of 0
def r_square_loss(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (SS_res/(SS_tot + K.epsilon()) )

# Loss function of RMSE + R^2
def joined_loss(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred) + r_square_loss(y_true, y_pred)

def normalize(x):
    # Normalize list of spectra, of shape: (# samples, spectrum_length)
    min_ = np.expand_dims(np.min(x, axis=1), -1)
    max_ = np.expand_dims(np.max(x, axis=1), -1)
    
    return (x-min_)/(max_-min_)


def normalize_1D(x):
    # Normalize one spectrum
    # use nan to num to avoid division by 0 if spectrum is flat at 0
    return np.nan_to_num((x - np.min(x))/(np.max(x) - np.min(x)))

class CustomPad(tf.keras.layers.Layer):
    def __init__(self, size, **kwargs):
        super(CustomPad, self).__init__(**kwargs)
        self.mid = int(size / 2)
        self.remainder = size - self.mid * 2 # If size is uneven

    def call(self, inputs, training=False):
        shape = tf.shape(inputs)
        seq_length = shape[1]
        
        left_column = tf.gather(inputs, [0], axis=1)
        right_column = tf.gather(inputs, [seq_length - 1], axis=1)
        
        padding_left = tf.tile(left_column, [1, self.mid, 1])
        padding_right = tf.tile(right_column, [1, self.mid -1 + self.remainder, 1])
        
        out = tf.concat([padding_left, inputs, padding_right], axis=1)
        
        return out
    def get_config(self):
        config = super().get_config()
        config.update({
            'size': self.mid,
        })
        return config



# Our ml models
def make_standard(lr = 0.00001):
    
    reset_seed(SEED = 0)  
    scaler = 20
    dim_red_size = 16**2
    l1_param = 1e-6
    l2_param = 1e-6
    
    inp = Input(shape = (None,1))
    
    kernel_size = 128
    padded = CustomPad(kernel_size)(inp)
    t = Conv1D(filters = 1 * scaler, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)
    
    kernel_size = 64
    padded = CustomPad(kernel_size)(t)
    inp_key = Conv1D(filters = dim_red_size, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)

    inp_key = GlobalMaxPooling1D()(inp_key)
    inp_key = tf.keras.backend.expand_dims(inp_key, -1)
    
    t_dot = Dot(axes=(2, 2))([inp, inp_key])
    t = tf.keras.backend.expand_dims(t_dot, -1)
    
    kernel_size = 32
    t = Conv2D(filters = 1 * scaler,
               kernel_size = (kernel_size, int(dim_red_size/10)),
               strides = (1, int(np.sqrt(dim_red_size))),
               padding = "same",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(t)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)
    
    kernel_size = 16
    t = Conv2D(filters = 1 * scaler, 
              kernel_size = (kernel_size, int(dim_red_size/10)),
              strides = (1, int(np.sqrt(dim_red_size))),
              padding = "same",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(t)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)
    
    t = Reshape((-1, scaler))(t)
    
    kernel_size = 128
    padded = CustomPad(kernel_size)(t)
    t = Conv1D(filters = 1 * scaler, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)
    
    kernel_size = 64
    padded = CustomPad(kernel_size)(t)
    t = Conv1D(filters = 1 * scaler, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)

    # Baseline
    kernel_size = 64
    padded = CustomPad(kernel_size)(t)
    bl = Conv1D(filters = 1, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=1e-5, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)
    #bl = BatchNormalization()(bl)
    bl = tf.keras.activations.relu(bl)

    pool_size = 33
    padded = CustomPad(pool_size)(bl)
    bl = AveragePooling1D(pool_size, 1, padding = "valid")(padded)
    bl = tf.keras.activations.relu(bl)

    
    # Cosmic rays
    cr_size = 3
    cr_padded = CustomPad(cr_size)(t)
    cosmic_rays = Conv1D(filters = 1, kernel_size = cr_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=1e-5, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(cr_padded)
    #cosmic_rays = BatchNormalization()(cosmic_rays)
    cosmic_rays = tf.keras.activations.relu(cosmic_rays)
    
    # Peaks, make the kernel size small to deal with potentially sharp peaks
    kernel_size = 8 # Can also be e.g. 5, an arbitrary choice for us so long as it is small
    padded = CustomPad(kernel_size)(t)
    peaks = Conv1D(filters = 1, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=1e-5, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)
    #peaks = BatchNormalization()(peaks) # Remove this line IFF: test with smaller kernel size doesn't work
    peaks = tf.keras.activations.relu(peaks)

    pool_size = 4
    padded = CustomPad(pool_size)(peaks)
    peaks = AveragePooling1D(pool_size, 1, padding = "valid")(padded)
    peaks = tf.keras.activations.relu(peaks)

    
    # Adopt Ions strategy: The noise is the signal left when subtracting baseline, comsic rays and peaks from the input
    noise = Add()([inp, -bl, -cosmic_rays, -peaks])

    
    # Flatten each part to make them comparable to the labels
    bl = Flatten()(bl)
    cosmic_rays = Flatten()(cosmic_rays)
    noise = Flatten()(noise)
    peaks = Flatten()(peaks)
    
    # Store output in a list which we return
    output = [bl, cosmic_rays, noise, peaks]
    
    model = Model(inp, output)
    
    model.compile(
        optimizer= Adam(learning_rate=lr),
        loss= root_mean_squared_error)
    """
    model = Model(inp, t)
    model.compile()
    """
    return model

def make_ensemble(lr=0.00001):
    
    reset_seed(SEED = 0)  
    scaler = 10
    dim_red_size = 12**2
    l1_param = 1e-6
    l2_param = 1e-6
    
    inp = Input(shape = (None,1))
    
    kernel_size = 128
    padded = CustomPad(kernel_size)(inp)
    t = Conv1D(filters = 1 * scaler, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)
    
    kernel_size = 64
    padded = CustomPad(kernel_size)(t)
    inp_key = Conv1D(filters = dim_red_size, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)

    inp_key = GlobalMaxPooling1D()(inp_key)
    inp_key = tf.keras.backend.expand_dims(inp_key, -1)
    
    t_dot = Dot(axes=(2, 2))([inp, inp_key])
    t = tf.keras.backend.expand_dims(t_dot, -1)
    
    kernel_size = 32
    t = Conv2D(filters = 1 * scaler,
               kernel_size = (kernel_size, int(dim_red_size/10)),
               strides = (1, int(np.sqrt(dim_red_size))),
               padding = "same",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(t)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)
    
    kernel_size = 16
    t_res = Conv2D(filters = 1 * scaler, 
              kernel_size = (kernel_size, int(dim_red_size/10)),
              strides = (1, int(np.sqrt(dim_red_size))),
              padding = "same",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(t)
    t = BatchNormalization()(t_res)
    t = LeakyReLU()(t)
    
    t = Reshape((-1, scaler))(t)
    
    kernel_size = 128
    padded = CustomPad(kernel_size)(t)
    t = Conv1D(filters = 1 * scaler, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)
    
    kernel_size = 64
    padded = CustomPad(kernel_size)(t)
    t = Conv1D(filters = 1 * scaler, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)

    # Baseline
    kernel_size = 64
    padded = CustomPad(kernel_size)(t)
    b = Conv1D(filters = 1, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=1e-5, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)
    
    pool_size = 33
    padded = CustomPad(pool_size)(b)
    b = AveragePooling1D(pool_size, 1, padding = "valid")(padded)
    b = tf.keras.activations.relu(b)
    
    # Cosmic rays
    cr_size = 3
    cr_padded = CustomPad(cr_size)(t)
    cr = Conv1D(filters = 1, kernel_size = cr_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=1e-5, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(cr_padded)

    cr = tf.keras.activations.relu(cr)

    ### Second part: Extract peaks and noise from the input
    reduced_spectrum = Add()([inp, -b, -cr])
    kernel_size = 128
    padded = CustomPad(kernel_size)(reduced_spectrum)
    t = Conv1D(filters = 1 * scaler, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)
    
    kernel_size = 64
    padded = CustomPad(kernel_size)(t)
    inp_key = Conv1D(filters = dim_red_size, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)

    inp_key = GlobalMaxPooling1D()(inp_key)
    inp_key = tf.keras.backend.expand_dims(inp_key, -1)
    
    t_dot = Dot(axes=(2, 2))([reduced_spectrum, inp_key])
    t = tf.keras.backend.expand_dims(t_dot, -1)
    
    kernel_size = 32
    t = Conv2D(filters = 1 * scaler,
               kernel_size = (kernel_size, int(dim_red_size/10)),
               strides = (1, int(np.sqrt(dim_red_size))),
               padding = "same",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(t)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)
    
    kernel_size = 16
    t = Conv2D(filters = 1 * scaler, 
              kernel_size = (kernel_size, int(dim_red_size/10)),
              strides = (1, int(np.sqrt(dim_red_size))),
              padding = "same",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(t)
    
    t = Add()([t, t_res])
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)
    
    t = Reshape((-1, scaler))(t)
    
    kernel_size = 128
    padded = CustomPad(kernel_size)(t)
    t = Conv1D(filters = 1 * scaler, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)
    
    kernel_size = 64
    padded = CustomPad(kernel_size)(t)
    t = Conv1D(filters = 1 * scaler, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)
    
    
    # Peaks, make the kernel size small to deal with potentially sharp peaks
    kernel_size = 8 
    padded = CustomPad(kernel_size)(t)
    p = Conv1D(filters = 1, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=1e-5, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)

    pool_size = 4
    padded = CustomPad(pool_size)(p)
    p = AveragePooling1D(pool_size, 1, padding = "valid")(padded)
    p = tf.keras.activations.relu(p)

    n = Conv1D(filters = 1, kernel_size = 1, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=1e-5, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(t)

    n = tf.keras.activations.tanh(n)


    # Flatten each part to make them comparable to the labels
    b = Flatten()(b)
    cr = Flatten()(cr)
    n = Flatten()(n)
    p = Flatten()(p)
    
    # Store output in a list which we return
    output = [b, cr, n, p]
    
    model = Model(inp, output)
    
    model.compile(
        optimizer= Adam(learning_rate=lr),
        loss= joined_loss,
        metrics = [
            root_mean_squared_error,
            r_square,
            ])

    return model


