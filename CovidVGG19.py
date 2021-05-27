#%tensorflow_version 2.0.0
import tensorflow as tf
from tensorflow.keras import Sequential,Model
from tensorflow.keras.layers import Dense,Dropout,Flatten,BatchNormalization
from tensorflow.keras.applications.vgg19 import VGG19,preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(directory='train',
                                              target_size = (224, 224),
                                              class_mode = 'categorical',
                                              batch_size=32)

validation_generator = datagen.flow_from_directory(directory='test',
                                              target_size = (224,224),
                                              class_mode = 'categorical',
                                              batch_size=32)

vgg_arch=VGG19(input_shape=(224,224,3),weights="imagenet",include_top=False)

for layers in vgg_arch.layers:
  layers.trainable=False
  
model=Sequential()
model.add(vgg_arch)
model.add(Flatten())
model.add(Dense(128,activation='relu',))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(3,activation="softmax"))

model.summary()

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])

history=model.fit_generator(generator=train_generator, steps_per_epoch=int(5144/32), epochs = 25, 
                              validation_data=validation_generator, validation_steps=int(1288/32), verbose = 1)

