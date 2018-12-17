#!/usr/bin/env python3
import sys
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input
from keras.optimizers import Adam, SGD
from keras.utils import plot_model
from loss import self_regularization_loss, local_adversarial_loss
import tensorflow as tf


class GAN(object):
    def __init__(self, model_inputs=[],model_outputs=[], name='gan'):
        self.OPTIMIZER = SGD(lr=2e-4,nesterov=True)
        self.NAME=name

        self.inputs = model_inputs
        self.outputs = model_outputs
        self.gan_model = Model(inputs = self.inputs, outputs = self.outputs)
        self.OPTIMIZER = SGD(lr=0.001)
        self.gan_model.compile(loss=[self_regularization_loss, self_regularization_loss],
                            optimizer=self.OPTIMIZER)
        self.save_model_graph()
        self.summary()

    def summary(self):
        return self.gan_model.summary()

    def save_model_graph(self):
        plot_model(self.gan_model, to_file='/out/GAN_Model.png')

    def save_model(self,epoch,batch):
        self.gan_model.save('/out/'+self.NAME+'_Epoch_'+epoch+'_Batch_'+batch+'model.h5')

    def self_regularization_loss(self,y_true,y_pred):
        return tf.multiply(0.0002,tf.reduce_sum(tf.abs(y_pred-y_true)))

    def local_adversarial_loss(self,y_true,y_pred):
        truth = tf.reshape(y_true,(-1,2))
        predicted = tf.reshape(y_pred,(-1,2))

        computed_loss = tf.nn.softmax_cross_entropy_with_logits(labels=truth,logits=predicted)
        output = tf.reduce_mean(computed_loss)
        return output