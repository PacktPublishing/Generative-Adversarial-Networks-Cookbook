#!/usr/bin/env python3
from gan import GAN
from generator import Generator
from discriminator import Discriminator
from keras.datasets import mnist
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import h5py    
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from voxelgrid import VoxelGrid


class Trainer:
    def __init__(self, side=16, latent_size=32, epochs =100, batch=32, checkpoint=50, data_dir = ''):
        self.SIDE=side
        self.EPOCHS = epochs
        self.BATCH = batch
        self.CHECKPOINT = checkpoint

        self.load_3D_MNIST(data_dir)
        self.load_2D_encoded_MNIST()
        self.LATENT_SPACE_SIZE = latent_size
        self.LABELS = [1]

        self.generator = Generator(latent_size=self.LATENT_SPACE_SIZE)
        self.discriminator = Discriminator(side=self.SIDE)
        self.gan = GAN(generator=self.generator.Generator, discriminator=self.discriminator.Discriminator)
        
    # Translate data to color
    def array_to_color(self,array, cmap="Oranges"):
        s_m = plt.cm.ScalarMappable(cmap=cmap)
        return s_m.to_rgba(array)[:,:-1]

    def translate(self,x):
        xx = np.ndarray((x.shape[0], 4096, 3))
        for i in range(x.shape[0]):
            xx[i] = self.array_to_color(x[i])
        del x
        return xx
    def load_3D_MNIST(self,input_dir):
        raw = h5py.File(input_dir, 'r')


        self.X_train_3D = np.array(raw['X_train'])
        self.X_train_3D = ( np.float32(self.X_train_3D) - 127.5) / 127.5
        self.X_train_3D = self.translate(self.X_train_3D).reshape(-1, 16, 16, 16, 3)
        self.X_test_3D = np.array(raw['X_test'])
        self.X_test_3D = ( np.float32(self.X_test_3D) - 127.5) / 127.5
        self.X_test_3D = self.translate(self.X_test_3D).reshape(-1, 16, 16, 16, 3)
        self.Y_train_3D = np.array(raw['y_train'])
        self.Y_test_3D = np.array(raw['y_test'])

        return 

    def load_2D_encoded_MNIST(self):
        (_, self.Y_train_2D), (_, self.Y_test_2D) = mnist.load_data()
        self.X_train_2D_encoded = np.load('x_train_encoded.npy')
        self.X_test_2D_encoded = np.load('x_test_encoded.npy')
        return

    def train(self):
        
        count_generated_images = int(self.BATCH/2)
        count_real_images = int(self.BATCH/2)
        for e in range(self.EPOCHS):
            for label in self.LABELS:

                # Grab the Real 3D Samples
                all_3D_samples = self.X_train_3D[np.where(self.Y_train_3D==label)]
                starting_index = randint(0, (len(all_3D_samples)-count_real_images))
                real_3D_samples = all_3D_samples[ starting_index : int((starting_index + count_real_images)) ]
                y_real_labels =  np.ones([count_generated_images,1])

                # Grab Generated Images for this training batch
                all_encoded_samples = self.X_train_2D_encoded[np.where(self.Y_train_2D==label)]
                starting_index = randint(0, (len(all_encoded_samples)-count_generated_images))
                batch_encoded_samples = all_encoded_samples[ starting_index : int((starting_index + count_generated_images)) ]
                batch_encoded_samples = batch_encoded_samples.reshape( count_generated_images, 1, 1, 1,self.LATENT_SPACE_SIZE)

                x_generated_3D_samples = self.generator.Generator.predict(batch_encoded_samples)
                y_generated_labels = np.zeros([count_generated_images,1])

                # Combine to train on the discriminator
                x_batch = np.concatenate( [real_3D_samples, x_generated_3D_samples] )
                y_batch = np.concatenate( [y_real_labels, y_generated_labels] )

                # Now, train the discriminator with this batch
                self.discriminator.Discriminator.trainable = False
                discriminator_loss = self.discriminator.Discriminator.train_on_batch(x_batch,y_batch)[0]
                self.discriminator.Discriminator.trainable = True

                # Generate Noise
                starting_index = randint(0, (len(all_encoded_samples)-self.BATCH))
                x_batch_encoded_samples = all_encoded_samples[ starting_index : int((starting_index + self.BATCH)) ]
                x_batch_encoded_samples = x_batch_encoded_samples.reshape( int(self.BATCH), 1, 1, 1,self.LATENT_SPACE_SIZE)
                y_generated_labels = np.ones([self.BATCH,1])
                generator_loss = self.gan.gan_model.train_on_batch(x_batch_encoded_samples,y_generated_labels)
                print ('Epoch: '+str(int(e))+' Label: '+str(int(label))+', [Discriminator :: Loss: '+str(discriminator_loss)+'], [ Generator :: Loss: '+str(generator_loss)+']')
                if e % self.CHECKPOINT == 0 and e != 0 :
                    self.plot_checkpoint(e,label)
            
        return

    def plot_checkpoint(self,e,label):
        filename = "/out/epoch_"+str(e)+"_label_"+str(label)+".png"

        all_encoded_samples = self.X_test_2D_encoded[np.where(self.Y_test_2D==label)]
        index = randint(0, (len(all_encoded_samples)-1))
        batch_encoded_samples = all_encoded_samples[ index ]
        batch_encoded_samples = batch_encoded_samples.reshape( 1, 1, 1, 1,self.LATENT_SPACE_SIZE)

        images = self.generator.Generator.predict(batch_encoded_samples)
        xs = []
        ys = []
        zs = []
        cs = []
        for i in range(16):
            for j in range(16):
                for k in range(16):
                    color = images[0][i][j][k]
                    if np.mean(color)<0.75 and np.mean(color)>0.25:
                        xs.append(i)
                        ys.append(j)
                        zs.append(k)
                        cs.append(color)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(xs,ys,zs,alpha=0.1,c=cs)
        plt.savefig(filename)

        return