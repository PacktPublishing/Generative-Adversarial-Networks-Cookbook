#!/usr/bin/env python3
import sys
import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization
from keras.layers.convolutional import Conv3D, Deconv3D
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import plot_model


class Discriminator(object):
    def __init__(self, side=16):
        self.INPUT_SHAPE = (side,side,side,3)
        self.OPTIMIZER = Adam(lr=0.000001, beta_1=0.5)


        self.Discriminator = self.model()
        self.Discriminator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER, metrics=['accuracy'] )
        # self.save_model()
        self.summary()

    def block(self,first_layer,filter_size=512,kernel_size=(3,3,3)):

        x = Conv3D(filters=filter_size, kernel_size=kernel_size, kernel_initializer='glorot_normal',
                    bias_initializer='zeros', padding='same')(first_layer)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

        return x

    def model(self):
        input_layer = Input(shape=self.INPUT_SHAPE)
        x = self.block(input_layer,filter_size=8)
        x = self.block(x,filter_size=16,)
        x = self.block(x,filter_size=32)
        x = self.block(x,filter_size=64)


        x = Conv3D(filters=1, kernel_size=(3,3,3),
                    strides=(1,1,1), kernel_initializer='glorot_normal',
                    bias_initializer='zeros', padding='valid')(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        output_layer = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=input_layer, outputs=output_layer)

        return model

    def summary(self):
        return self.Discriminator.summary()

    def save_model(self):
        plot_model(self.Discriminator.model, to_file='/data/Discriminator_Model.png')

