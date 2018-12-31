#!/usr/bin/env python3
import sys
import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Lambda, Concatenate, MaxPooling2D
from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD,Nadam, Adamax
from keras.utils import plot_model
import tensorflow as tf
from loss import local_adversarial_loss



class Discriminator(object):
    def __init__(self, width = 35, height= 55, channels = 1,name='discriminator'):
        self.W = width
        self.H = height
        self.C = channels
        self.SHAPE = (height,width,channels)
        self.NAME = name
        
        self.Discriminator = self.model()
        self.OPTIMIZER = SGD(lr=0.001)
        self.Discriminator.compile(loss=local_adversarial_loss, optimizer=self.OPTIMIZER)

        self.save_model_graph()
        self.summary()

    def model(self):
        input_layer = Input(shape=self.SHAPE)
        x = Convolution2D(96,3,3, subsample=(2,2), border_mode='same',activation='relu')(input_layer)
        x = Convolution2D(64,3,3, subsample=(2,2), border_mode='same',activation='relu')(x)
        x = MaxPooling2D(pool_size=(3,3),border_mode='same')(x)
        x = Convolution2D(32,3,3, subsample=(1,1), border_mode='same',activation='relu')(x)
        x = Convolution2D(32,1,1, subsample=(1,1), border_mode='same',activation='relu')(x)
        x = Convolution2D(2,1,1, subsample=(1,1), border_mode='same',activation='relu')(x)
        output_layer = Reshape((-1,2))(x)
        return Model(input_layer,output_layer) 

    def summary(self):
        return self.Discriminator.summary()

    def save_model_graph(self):
        plot_model(self.Discriminator, to_file='/data/Discriminator_Model.png')

    def save_model(self,epoch,batch):
        self.Discriminator.save('/out/'+self.NAME+'_Epoch_'+epoch+'_Batch_'+batch+'model.h5')
