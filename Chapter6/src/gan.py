#!/usr/bin/env python3
import sys
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input
from keras.optimizers import Adam, SGD
from keras.utils import plot_model

class GAN(object):
    def __init__(self, model_inputs=[],model_outputs=[]):
        self.OPTIMIZER = SGD(lr=2e-4,nesterov=True)

        self.inputs = model_inputs
        self.outputs = model_outputs
        self.gan_model = Model(inputs = self.inputs, outputs = self.outputs)
        self.OPTIMIZER = Adam(lr=2e-4, beta_1=0.5)
        self.gan_model.compile(loss=['mse', 'mae'],
                            loss_weights=[  1, 100],
                            optimizer=self.OPTIMIZER)
        self.save_model()
        self.summary()

    def model(self):
        model = Model()
        return model

    def summary(self):
        return self.gan_model.summary()

    def save_model(self):
        plot_model(self.gan_model, to_file='/out/GAN_Model.png')
