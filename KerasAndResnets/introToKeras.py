# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 04:56:55 2020

@author: Sasha
"""

##
import numpy as np
from tensorflow import keras
import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical

##
train_images = mnist.train_images()
test_images  = mnist.test_images()
train_labels = mnist.train_labels()
test_labels = mnist.test_labels()

print("=======================================================")
print ("train images dimension: " + str(train_images.shape))
print ("train_labels images dimension: " + str(train_labels.shape))
print ("test images dimension: " + str(test_images.shape))
print ("test images dimension: " + str(test_labels.shape))

#get the shapes and also the normalisation aka prepering the data

#Normalize images 

train_images = (train_images/255) - 0.5
test_images  = (test_images/255) - 0.5

#reshape
train_images = np.expand_dims(train_images, axis = 3)
test_images = np.expand_dims(test_images, axis = 3)

print("=======================================================")
print ("train images reshaped dimension: " + str(train_images.shape))
print ("train_labels reshaped images dimension: " + str(train_labels.shape))

#working with the model; using 3 layers for the CNN

#initializing params
num_filters = 8 
filter_size = 3
pool_size = 2

#sequential model
model = Sequential([   
    Conv2D(num_filters, (filter_size, filter_size), strides = (1,1), input_shape = (28, 28, 1)), 
    MaxPooling2D(pool_size = pool_size),
    Flatten(),
    Dense(10, activation = 'softmax')    
    ])

#compiling the model; compute the loss and also determine the optimizer

model.compile(
    'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
        
    )

#train the model

model.fit(
    train_images,
    to_categorical(train_labels),
    epochs = 5,
    validation_data=(test_images, to_categorical(test_labels))
    )
#the % accuracy is 97.3%
#cost is 0.0855

 #predict an image

model.predict(test_images[:3])









 
 