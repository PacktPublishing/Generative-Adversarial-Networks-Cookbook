#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# Read from this directory
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

# Look at the shape of the images training data:
print("Shape of the Image Training Data is "+str(mnist.train.images.shape))
# Shape of the Image Training Data is (55000, 784)


# Look at the shape of the labels training data:
print("Shape of the Label Training Data is "+str(mnist.train.labels.shape))
# One-Hot False : Shape of the Label Training Data is (55000,)
# One-Hot True  : Shape of the Label Training Data is (55000,10)

# Take a Random Example from the DatasetS:
index = np.random.choice(mnist.train.images.shape[0], 1)
random_image = mnist.train.images[index]
random_label = mnist.train.labels[index]
random_image = random_image.reshape([28, 28]);

# Plot the Image
plt.gray()
plt.imshow(random_image)
plt.show()
