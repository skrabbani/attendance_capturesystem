# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 12:44:56 2020

@author: hafim
"""


from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'E:/Face recognisation project/trainset'
valid_path = 'E:/Face recognisation project/testset'

#print[train_labels]

# add preprocessing layer to the front of VGG
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# not training the existing weights
for layer in vgg.layers:
  layer.trainable = False
  
##train_labels = to_categorical(train_labels)
  
  # getting number of classes
folders = glob('E:/Face recognisation project/trainset')
  

# flattening the layers
x = Flatten()(vgg.output)

prediction = Dense(output_dim = 4,  activation='softmax')(x)

# creating a model object
model = Model(inputs=vgg.input, outputs=prediction)

# to view the structure of the model if needed
model.summary()

# Compiling the model
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

# importing the ImageDataGenerator to view the classes and the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('E:/Face recognisation project/trainset',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('E:/Face recognisation project/trainset',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')
 
# fitting the model
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

# plotting the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plotting the accuracies
plt.plot(r.history['accuracy'], label='train accuracy')
plt.plot(r.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()
plt.savefig('AccVal_accuracy')


# saving the final model
import tensorflow as tf

from keras.models import load_model

model.save('E:/face recognisation project/facefeatures_new_model.h5')
