#This script is to build a network, train the network, and save the model using Keras

#import necessary libraries
import numpy as np
import os
import csv
import cv2
import sklearn

samples = []
with open('C:/Users/Vatche/Documents/Udacity/recorded driving data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

#split data into training and validation sets (80/20)
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#generator definition:
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name_center = batch_sample[0].split('/')[-1]
                name_left = batch_sample[1].split('/')[-1]
                name_right = batch_sample[2].split('/')[-1]
                center_image = cv2.imread(name_center)
                left_image = cv2.imread(name_left)
                right_image = cv2.imread(name_right)
                #create angles and adjustments
                correction = .3 # this is a parameter to tune
                center_angle = float(batch_sample[3])
                left_angle = center_angle + correction
                right_angle = center_angle - correction
                images.extend([center_image, left_image, right_image])
                angles.extend([center_angle, left_angle, right_angle])
            # trim image to only see section with road
            X_train = np.asarray(images) 
            y_train = np.asarray(angles)
            #print(X_train.shape)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#create neural network model
from keras.models import Sequential, Model
from keras.layers import Cropping2D, Lambda, Convolution2D
from keras.activations import relu, softmax
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.pooling import MaxPooling2D

model = Sequential()

#Preprocessing:

#crop model images
model.add(Cropping2D(cropping=((50,10), (0,0)), input_shape=(160,320,3)))

# Normalize image data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/255 - 0.5))

#convolution layer 1 with max pooling and activation:
model.add(Convolution2D(24,5,5, border_mode='valid'))
model.add(MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid'))
#add a dropout layer
model.add(Dropout(0.5))
model.add(Activation('relu'))

#convolution layer 2 with max pooling and activation:
model.add(Convolution2D(36,5,5, border_mode='valid'))
model.add(MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid'))
model.add(Activation('relu'))

#convolution layer 3 with max pooling and activation:
model.add(Convolution2D(48,5,5, border_mode='valid'))
model.add(MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid'))
model.add(Activation('relu'))

#convolution layer 4 with max pooling and activation:
model.add(Convolution2D(64,3,3, border_mode='valid'))
model.add(MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid'))
model.add(Activation('relu'))

#Extra activation layer to handle nonlinearities
model.add(Activation('relu'))

#flatten image
model.add(Flatten())

#fully connected layer 1 aka Dense with activation
model.add(Dense(100))
model.add(Activation('relu'))

#fully connected layer 2 with activation
model.add(Dense(50))
model.add(Activation('relu'))

#fully connected layer 3 with activation
model.add(Dense(10))
model.add(Activation('relu'))

#output layer
model.add(Dense(1))		#1 output because it is steering angle, no softmax because no classification

#model compilation
model.compile(loss='mse', optimizer='adam')

#if finetuning and I want to start at saved weights, uncomment this and comment the model construction
'''from keras.models import load_model
model = load_model('model.h5')'''

#run training
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=10)

#visualize the model
'''from keras.utils.visualize_util import plot
plot(model, to_file='model.png')'''

#save model for using in simulator with drive.py
model.save('model.h5')