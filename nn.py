from random import random
from unittest import skip
import tensorflow as tf
from tensorflow import keras
from keras import layers, initializers, applications
from keras.layers import Add, Conv2D, Activation, BatchNormalization, Dropout, MaxPooling2D, Conv2DTranspose, concatenate, Input
from keras.initializers import random_uniform, glorot_uniform
import numpy as np
import matplotlib.pyplot as plt
import os
### ResNets

keras,applications.re
## We use identity residual block when the input and the output dimensions are the same

def identity_block(X, f, filters, training=True, initializer=random_uniform):
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters = F1, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training = training) # Default axis
    X = Activation('relu')(X)
    
    ## Second component of main path
    X = Conv2D(filters = F2, kernel_size = f,strides = (1, 1),padding='same',kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training=training)
    X = Activation('relu')(X)

    ## Third component of main path
    X = Conv2D(filters = F3, kernel_size = 1, strides = (1, 1), padding='valid', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training=training)
    
    ## Final step: Add shortcut value to main path, and pass it through a RELU activation
    # keras.layers.Add()[...] takes as input a list of tensors, all of the same shape, 
    # and returns a single tensor (also of the same shape).
    X = Add()([X_shortcut,X])
    X = Activation('relu')(X)

    return X





## We use convolutional blocks when the input and the output dimensions are different

def convolutional_block(X, f, filters, s = 2, training=True, initializer=glorot_uniform):
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X
    
    # First component of main path glorot_uniform(seed=0)
    X = Conv2D(filters = F1, kernel_size = 1, strides = (s, s), padding='valid', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training=training)
    X = Activation('relu')(X)
    
    ## Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = f,strides = (1, 1),padding='same',kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training=training)
    X = Activation('relu')(X)

    ## Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = 1, strides = (1, 1), padding='valid', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training=training)
    
    ##### SHORTCUT PATH #####
    X_shortcut = Conv2D(filters = F3, kernel_size = 1, strides = (s, s), padding='valid', kernel_initializer = initializer(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3)(X_shortcut, training=training)

    # Final step: Add shortcut value to main path (Use this order [X, X_shortcut]), and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X


def waveNet():
    model = keras.sequential()
    model.add(layers.InputLayer(input_shape=[None, 1]))

    for dilation_rate in (1, 2, 4, 8, 16, 32):
        model.add(
            layers.Conv1D(filters=32,
            kernel_size=2,
            strides=1,
            dilation_rate=dilation_rate,
            padding='casual',
            activation='relu')
        )

        model.add(layers.Conv1D(filters=1, kernel_size=1))

        return model