#!/usr/bin/env python3
import sys
import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Lambda, Concatenate
from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD,Nadam, Adamax
import keras.backend as K
from keras.utils import plot_model


class Discriminator(object):
    def __init__(self, width = 28, height= 28, channels = 1, starting_filters=64):
        self.W = width
        self.H = height
        self.C = channels
        self.CAPACITY = width*height*channels
        self.SHAPE = (width,height,channels)
        self.FS = starting_filters #FilterStart
        
        self.Discriminator = self.model()
        self.OPTIMIZER = Adam(lr=2e-4, beta_1=0.5,decay=1e-5)
        self.Discriminator.compile(loss='mse', optimizer=self.OPTIMIZER, metrics=['accuracy'] )

        self.save_model()
        self.summary()

    def model(self):


        input_A = Input(shape=self.SHAPE)
        input_B = Input(shape=self.SHAPE)
        input_layer = Concatenate(axis=-1)([input_A, input_B])

        up_layer_1 = Convolution2D(self.FS, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(input_layer)

        up_layer_2 = Convolution2D(self.FS*2, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(up_layer_1)
        leaky_layer_2 =  BatchNormalization(momentum=0.8)(up_layer_2)

        up_layer_3 = Convolution2D(self.FS*4, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(leaky_layer_2)
        leaky_layer_3 =  BatchNormalization(momentum=0.8)(up_layer_3)

        up_layer_4 = Convolution2D(self.FS*8, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(leaky_layer_3)
        leaky_layer_4 = BatchNormalization(momentum=0.8)(up_layer_4)

        output_layer = Convolution2D(1, kernel_size=4, strides=1, padding='same')(leaky_layer_4)
        
        return Model([input_A, input_B],output_layer)

    def summary(self):
        return self.Discriminator.summary()

    def save_model(self):
        plot_model(self.Discriminator, to_file='/out/Discriminator_Model.png')

