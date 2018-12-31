#!/usr/bin/env python3
import sys
import numpy as np
from keras.layers import Dense, Reshape, Input, BatchNormalization, Concatenate, Activation
from keras.layers.core import Activation
from keras.layers.convolutional import UpSampling2D, Convolution2D, MaxPooling2D,Deconvolution2D, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, Nadam,Adamax
from keras import initializers
from keras import layers
from keras.utils import plot_model
import tensorflow as tf
from loss import  self_regularization_loss

class Generator(object):
    def __init__(self, width = 35, height= 55, channels = 1,name='generator'):
        
        self.W = width
        self.H = height
        self.C = channels
        self.SHAPE = (height,width,channels)
        self.NAME = name

        self.Generator = self.model()
        self.OPTIMIZER = SGD(lr=0.001)
        self.Generator.compile(loss=self_regularization_loss, optimizer=self.OPTIMIZER)

        self.save_model_graph()
        self.summary()

    def model(self):
        # Input
        input_layer = Input(shape=self.SHAPE)
        x = Convolution2D(64, 3,3, border_mode='same',activation='relu')(input_layer)

        # ResNet Block 1
        res_x_input_1 = Conv2D(64, (3,3), border_mode='same',activation='relu')(x)
        x = Convolution2D(64, 3,3, border_mode='same',activation='relu')(res_x_input_1)
        x = layers.Add()([res_x_input_1,x])
        x = Activation('relu')(x)

        # ResNet Block 2
        res_x_input_2 = Conv2D(64, (3,3), border_mode='same',activation='relu')(x)
        x = Convolution2D(64, 3,3, border_mode='same',activation='relu')(res_x_input_2)
        x = layers.Add()([res_x_input_2,x])
        x = Activation('relu')(x)

        # ResNet Block 3
        res_x_input_3 = Conv2D(64, (3,3), border_mode='same',activation='relu')(x)
        x = Convolution2D(64, 3,3, border_mode='same',activation='relu')(res_x_input_3)
        x = layers.Add()([res_x_input_3,x])
        x = Activation('relu')(x)

        # ResNet Block 4
        res_x_input_4 = Conv2D(64, (3,3), border_mode='same',activation='relu')(x)
        x = Convolution2D(64, 3,3, border_mode='same',activation='relu')(res_x_input_4)
        x = layers.Add()([res_x_input_4,x])
        x = Activation('relu')(x)

        output_layer = Convolution2D(self.C,1,1, border_mode='same',activation='tanh')(x)
        
        return Model(input_layer,output_layer)

    def summary(self):
        return self.Generator.summary()

    def save_model_graph(self):
        plot_model(self.Generator, to_file='/out/Generator_Model.png')
    
    def save_model(self,epoch,batch):
        self.Generator.save('/out/'+self.NAME+'_Epoch_'+epoch+'_Batch_'+batch+'model.h5')
