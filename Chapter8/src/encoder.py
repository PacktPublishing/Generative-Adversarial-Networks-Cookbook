from keras.datasets import mnist
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

# Download the data and format for learning
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# How much encoding do we want for our setup?
encoding_dimension = 256  

# Keras has an input shape of 784
input_layer = Input(shape=(784,))
encoded_layer = Dense(encoding_dimension, activation='relu')(input_layer)
decoded = Dense(784, activation='sigmoid')(encoded_layer)

# Build the Model
ac = Model(input_layer, decoded)

# Create an encoder model that we will save later
encoder = Model(input_layer, encoded_layer)

# Train the autoencoder model, ac
ac.compile(optimizer='adadelta', loss='binary_crossentropy')
ac.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# Save the Predicted Data x_train
x_train_encoded = encoder.predict(x_train)
np.save('/src/x_train_encoded.npy',x_train_encoded)
# Save the Predicted Data x_test
x_test_encoded = encoder.predict(x_test)
np.save('/src/x_test_encoded.npy',x_test_encoded)
# Save the Encoder model
encoder.save('/src/encoder_model.h5')