#Importing the libraries

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import pandas as pd
import numpy as np
import time
import string
import re
import matplotlib.pyplot as plt

import utils


#Creating a neural network with one-hidden layer for the data prepared using TF-IDF Vectorization, for our regression problem

def dnn_reg(X_train, y_train, X_test, y_test):
    model = keras.Sequential([
          layers.Dense(128, activation='relu'),
          layers.Dense(1, activation='relu')
      ])

#Using our loss function as MAE and our optimizer as Adam
    model.compile(loss='mean_absolute_error',
                    optimizer=tf.keras.optimizers.Adam(0.001))

#Early stopping of our model when the val_loss starts increasing
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    
    history = model.fit(X_train,y_train, validation_split=0.33, epochs=100, callbacks=[callback])
    return history, model.predict(X_test)


#Creating a neural network with one hidden layer for the data prepared using Word2Vec, for our regression problem   

def dnn_word2vec(X_train, y_train, X_test, y_test):
    model = keras.Sequential([
          layers.Dense(128, activation='relu'),
          layers.Dense(1, activation='relu')
      ])

#Using our loss function as MAE and our optimizer as Adam
    model.compile(loss='mean_absolute_error',                       
                    optimizer=tf.keras.optimizers.Adam(0.01))

#Early stopping of our model when the val_loss starts increasing
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    
    
    history = model.fit(X_train,y_train, validation_split=0.33,epochs=100, callbacks=[callback])
    return history, model.predict(X_test)
    
#Plotting the training loss vs the validation loss    
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()