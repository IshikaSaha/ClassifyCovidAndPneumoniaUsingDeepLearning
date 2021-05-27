import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from IPython.display import Image
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
#import tf.keras.optimizers.RMSprop()


mobile = keras.applications.mobilenet_v2.MobileNetV2()


base_model=MobileNetV2(weights='imagenet',include_top=False, input_shape=(224,224,3)) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(3,activation='softmax')(x) #final layer with softmax activation

model=Model(inputs=base_model.input,outputs=preds)
#specify the inputs
#specify the outputs
#now a model has been created based on our architecture


for i,layer in enumerate(model.layers):
  print(i,layer.name)
  
  
for layer in model.layers:
    layer.trainable=False
# or if we want to set the first 20 layers of the network to be non-trainable
for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True
    
    
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

datagen = ImageDataGenerator(rescale=1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
train_generator = datagen.flow_from_directory(directory='Data/train',
                                              target_size = (224, 224),
                                              class_mode = 'categorical',
                                              batch_size=32)

validation_generator = datagen.flow_from_directory(directory='Data/test',
                                              target_size = (224,224),
                                              class_mode = 'categorical',
                                              batch_size=32)

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy


history=model.fit_generator(generator=train_generator, steps_per_epoch=int(5144/32), epochs = 25, 
                              validation_data=validation_generator, validation_steps=int(1288/32), verbose = 1)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(["Train_acc","Validation_acc"])
plt.show()


