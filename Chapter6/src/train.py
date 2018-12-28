#!/usr/bin/env python3
from gan import GAN
from generator import Generator
from discriminator import Discriminator
from keras.layers import Input
from keras.datasets import mnist
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import os
from PIL import Image
import random
import numpy as np

class Trainer:
    def __init__(self, height = 256, width = 256, channels=3, epochs = 50000, batch = 1, checkpoint = 50, train_data_path = '',test_data_path=''):
        self.EPOCHS = epochs
        self.BATCH = batch
        self.H = height
        self.W = width
        self.C = channels
        self.CHECKPOINT = checkpoint

        self.X_train_B, self.X_train_A = self.load_data(train_data_path)
        self.X_test_B, self.X_test_A  = self.load_data(test_data_path)


        self.generator = Generator(height=self.H, width=self.W, channels=self.C)

        self.orig_A = Input(shape=(self.W, self.H, self.C))
        self.orig_B = Input(shape=(self.W, self.H, self.C))

        self.fake_A = self.generator.Generator(self.orig_B)

        self.discriminator = Discriminator(height=self.H, width=self.W, channels=self.C)
        self.discriminator.trainable = False
        self.valid = self.discriminator.Discriminator([self.fake_A,self.orig_B])

        model_inputs  = [self.orig_A,self.orig_B]
        model_outputs = [self.valid, self.fake_A]
        self.gan = GAN(model_inputs=model_inputs,model_outputs=model_outputs)
        
        

    def load_data(self,data_path):
        listOFFiles = self.grabListOfFiles(data_path,extension="jpg")
        imgs_temp = np.array(self.grabArrayOfImages(listOFFiles))
        imgs_A = []
        imgs_B = []
        for img in imgs_temp:
            imgs_A.append(img[:,:self.H])
            imgs_B.append(img[:,self.H:])

        imgs_A_out = self.norm_and_expand(np.array(imgs_A))
        imgs_B_out = self.norm_and_expand(np.array(imgs_B))

        return imgs_A_out, imgs_B_out

    def norm_and_expand(self,arr):
        arr = (arr.astype(np.float32) - 127.5)/127.5
        normed = np.expand_dims(arr, axis=3)
        return normed

    def grabListOfFiles(self,startingDirectory,extension=".webp"):
        listOfFiles = []
        for file in os.listdir(startingDirectory):
            if file.endswith(extension):
                listOfFiles.append(os.path.join(startingDirectory, file))
        return listOfFiles

    def grabArrayOfImages(self,listOfFiles,gray=False):
        imageArr = []
        for f in listOfFiles:
            if gray:
                im = Image.open(f).convert("L")
            else:
                im = Image.open(f).convert("RGB")
            imData = np.asarray(im)
            imageArr.append(imData)
        return imageArr


    def train(self):
        for e in range(self.EPOCHS):
            b = 0
            X_train_A_temp = deepcopy(self.X_train_A)
            X_train_B_temp = deepcopy(self.X_train_B)

            number_of_batches = len(self.X_train_A)
        
            for b in range(number_of_batches):
                # Train Discriminator
                # Grab Real Images for this training batch
                starting_ind = randint(0, (len(X_train_A_temp)-1))
                real_images_raw_A = X_train_A_temp[ starting_ind : (starting_ind + 1) ]
                real_images_raw_B = X_train_B_temp[ starting_ind : (starting_ind + 1) ]

                # Delete the images used until we have none left
                X_train_A_temp = np.delete(X_train_A_temp,range(starting_ind,(starting_ind + 1)),0)
                X_train_B_temp = np.delete(X_train_B_temp,range(starting_ind,(starting_ind + 1)),0)

                batch_A = real_images_raw_A.reshape( 1, self.W, self.H, self.C )
                batch_B = real_images_raw_B.reshape( 1, self.W, self.H, self.C )

                # PatchGAN
                y_valid = np.ones((1,)+(int(self.W / 2**4), int(self.W / 2**4), 1))
                y_fake = np.zeros((1,)+(int(self.W / 2**4), int(self.W / 2**4), 1))

                fake_A = self.generator.Generator.predict(batch_B)

                # Now, train the discriminator with this batch of reals
                discriminator_loss_real = self.discriminator.Discriminator.train_on_batch([batch_A,batch_B],y_valid)[0]
                discriminator_loss_fake = self.discriminator.Discriminator.train_on_batch([fake_A,batch_B],y_fake)[0]
                full_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)

                generator_loss = self.gan.gan_model.train_on_batch([batch_A, batch_B],[y_valid,batch_A])    

                print ('Batch: '+str(int(b))+', [Full Discriminator :: Loss: '+str(full_loss)+'], [ Generator :: Loss: '+str(generator_loss)+']')
                if b % self.CHECKPOINT == 0 :
                    label = str(e)+'_'+str(b)
                    self.plot_checkpoint(label)

            print ('Epoch: '+str(int(e))+', [Full Discriminator :: Loss:'+str(full_loss)+'], [ Generator :: Loss: '+str(generator_loss)+']')
                        
        return

    def plot_checkpoint(self,b):
        orig_filename = "/out/batch_check_"+str(b)+"_original.png"

        r, c = 3, 3
        random_inds = random.sample(range(len(self.X_test_A)),3)
        imgs_A = self.X_test_A[random_inds].reshape(3, self.W, self.H, self.C )
        imgs_B = self.X_test_B[random_inds].reshape( 3, self.W, self.H, self.C )
        fake_A = self.generator.Generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Style', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("/out/batch_check_"+str(b)+".png")
        plt.close('all')

        return