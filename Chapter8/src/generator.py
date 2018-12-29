#!/usr/bin/env python3
import sys
import numpy as np
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv3D, Deconv3D
from keras.layers import Input, BatchNormalization, Dense, Reshape
from keras.layers.core import Activation
from keras.optimizers import Adam, SGD
from keras.utils import plot_model

class Generator(object):
    def __init__(self, latent_size=100):

        self.INPUT_SHAPE = (1, 1, 1, latent_size)
        # self.OPTIMIZER = Adam(lr=0.0001,beta_1=0.5)
        self.OPTIMIZER = SGD(lr=0.001, nesterov=True)


        self.Generator = self.model()
        self.Generator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER)
        # self.save_model()
        self.summary()

    def block(self,first_layer,filter_size=512,stride_size=(2,2,2),kernel_size=(4,4,4),padding='same'):

        x = Deconv3D(filters=filter_size, kernel_size=kernel_size,
                    strides=stride_size, kernel_initializer='glorot_normal',
                    bias_initializer='zeros', padding=padding)(first_layer)
        x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)

        return x


    def model(self):
        input_layer = Input(shape=self.INPUT_SHAPE)

        x = self.block(input_layer,filter_size=256,stride_size=(1,1,1),kernel_size=(4,4,4),padding='valid')
        x = self.block(x,filter_size=128,stride_size=(2,2,2),kernel_size=(4,4,4))

        x = Deconv3D(filters=3, kernel_size=(4,4,4),
                    strides=(2,2,2), kernel_initializer='glorot_normal',
                    bias_initializer='zeros', padding='same')(x)
        x = BatchNormalization()(x)
        output_layer = Activation(activation='sigmoid')(x)


        model = Model(inputs=input_layer, outputs=output_layer)
        return model

    def summary(self):
        return self.Generator.summary()

    def save_model(self):
        plot_model(self.Generator.model, to_file='/data/Generator_Model.png')

