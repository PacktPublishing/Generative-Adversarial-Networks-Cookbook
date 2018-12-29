#!/usr/bin/env python3
import sys
import numpy as np
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import plot_model

class GAN(object):
    def __init__(self,discriminator,generator):
        self.OPTIMIZER = Adam(lr=0.008, beta_1=0.5)
        
        self.Generator = generator
        self.Discriminator = discriminator
        self.Discriminator.trainable = True
        
        self.gan_model = self.model()
        self.gan_model.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER)
        # self.save_model()
        self.summary()

    def model(self):
        model = Sequential()
        model.add(self.Generator)
        model.add(self.Discriminator)
        return model

    def summary(self):
        return self.gan_model.summary()

    def save_model(self):
        plot_model(self.gan_model.model, to_file='/data/GAN_Model.png')
