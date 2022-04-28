# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 01:28:25 2022

@author: abdal
"""

import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
# seaborn is of use for visualizing.
import seaborn as sns



# load train, test, and submission sample dataset.
import tensorflow.python as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import Dropout
# Part 1 - Data Preprocessing

# Generating images for the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.3,
                                   zoom_range = 0.3,
                                   rotation_range=45,
                                   width_shift_range=.15,
                                   height_shift_range=.15,
                                   horizontal_flip = True)

# Generating images for the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)

# Creating the Training set
training_set = train_datagen.flow_from_directory('Patches',
                                                 target_size = (64, 64),
                                                 batch_size = 64,
                                                 class_mode = 'binary')

# Creating the Test set
test_set = test_datagen.flow_from_directory('TestSet',
                                            target_size = (64, 64),
                                            batch_size = 64,
                                            class_mode = 'binary')



# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding="same", activation="relu", input_shape=[64, 64, 3]))
cnn.add(Dropout(.2))
# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=32, strides=2, padding='valid'))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
cnn.add(Dropout(.2))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=256, activation='elu'))
cnn.add(Dropout(.3))


# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])

# Training the CNN on the Training set and evaluating it on the Test set
history=cnn.fit_generator(training_set,verbose =2, workers=3,
                  epochs = 20,
                  validation_data = test_set,
                  )

acc = history.history['acc']
val_acc = history.history['val_acc']

# Retrieve a list of list results on training and validation data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')

plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')
y_pred=cnn.predict(test_set)

cnn.save('diabetic_foot.h5')