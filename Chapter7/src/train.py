#!/usr/bin/env python3
from gan import GAN
from generator import Generator
from discriminator import Discriminator
from keras.datasets import mnist
from keras.layers import Input
from random import randint
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
from copy import deepcopy

class Trainer:
    def __init__(self, height=55,width=35, channels=1,epochs =100, batch=16, checkpoint=50,sim_path='',real_path='',data_limit=0.001,generator_steps=2,discriminator_steps=1):
        self.W = width
        self.H = height
        self.C = channels
        self.EPOCHS = epochs
        self.BATCH = batch
        self.CHECKPOINT = checkpoint
        self.DATA_LIMIT=data_limit
        self.GEN_STEPS = generator_steps
        self.DISC_STEPS = discriminator_steps

        self.X_real = self.load_h5py(real_path)
        self.X_sim = self.load_h5py(sim_path)

        self.refiner = Generator(height=self.H, width=self.W, channels=self.C)
        self.discriminator = Discriminator(height=self.H, width=self.W, channels=self.C)
        self.discriminator.trainable = False

        self.synthetic_image = Input(shape=(self.H, self.W, self.C))
        self.real_or_fake = Input(shape=(self.H, self.W, self.C))


        self.refined_image = self.refiner.Generator(self.synthetic_image)
        self.discriminator_output = self.discriminator.Discriminator(self.real_or_fake)
        self.combined = self.discriminator.Discriminator(self.refined_image)

        model_inputs  = [self.synthetic_image]
        model_outputs = [self.refined_image, self.combined]
        self.gan = GAN(model_inputs=model_inputs,model_outputs=model_outputs)

     def load_h5py(self,data_path):
        with h5py.File(data_path,'r') as t_file:
            print('Images found:',len(t_file['image']))
            image_stack = np.stack([np.expand_dims(a,-1) for a in t_file['image'].values()],0)
        return image_stack

    def train(self):
        for e in range(self.EPOCHS):

            b = 0
            X_real_temp = deepcopy(self.X_real)
            X_sim_temp = deepcopy(self.X_sim)
            combined_loss = np.zeros(shape=len(self.gan.gan_model.metrics_names))
            discriminator_loss_real = np.zeros(shape=len(self.discriminator.Discriminator.metrics_names))
            discriminator_loss_sim = np.zeros(shape=len(self.discriminator.Discriminator.metrics_names))

            while min(len(X_real_temp),len(X_sim_temp))>self.BATCH:
                # Keep track of Batches
                b=b+1

                count_real_images = int(self.BATCH)
                starting_indexs = randint(0, (min(len(X_real_temp),len(X_sim_temp))-count_real_images))
              
                real_images_raw = X_real_temp[ starting_indexs : (starting_indexs + count_real_images) ]
                real_images = real_images_raw.reshape( count_real_images, self.H, self.W, self.C )

                y_real = np.array([[[1.0, 0.0]] * self.discriminator.Discriminator.output_shape[1]] * self.BATCH)
                
                sim_images_raw = X_sim_temp[ starting_indexs : (starting_indexs + count_real_images) ]
                sim_images = sim_images_raw.reshape( count_real_images, self.H, self.W, self.C )

                y_sim = np.array([[[0.0, 1.0]] * self.discriminator.Discriminator.output_shape[1]] * self.BATCH)

                for _ in range(self.GEN_STEPS):
                    combined_loss = np.add(self.gan.gan_model.train_on_batch(sim_images,[sim_images, y_real]), combined_loss)
        
                for _ in range(self.DISC_STEPS):
                    improved_image_batch = self.refiner.Generator.predict_on_batch(sim_images)
                    discriminator_loss_real = np.add(self.discriminator.Discriminator.train_on_batch(real_images, y_real), discriminator_loss_real)
                    discriminator_loss_sim = np.add(self.discriminator.Discriminator.train_on_batch(improved_image_batch, y_sim),discriminator_loss_sim)

            print ('Epoch: '+str(int(e))+', [Real Discriminator :: Loss: '+str(discriminator_loss_real)+'], [ GAN :: Loss: '+str(combined_loss)+']')
                        
        return
