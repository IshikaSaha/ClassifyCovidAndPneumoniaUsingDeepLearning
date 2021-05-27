# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
        
        
        
DATA_PATH = "C:\\Users\\KIIT\\Desktop\\MachineLearning\\Paper2\\covid\\Data\\Data"

if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)
    print("Dataset folder created")
    
TRAIN_PATH = "C:\\Users\\KIIT\\Desktop\\MachineLearning\\Paper2\\covid\\Data\\Data\\train"
VAL_PATH = "C:\\Users\\KIIT\\Desktop\\MachineLearning\\Paper2\\covid\\Data\\Data\\test"

import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.applications import ResNet50

TRAIN_COVID_PATH = "C:\\Users\\KIIT\\Desktop\\MachineLearning\\Paper2\\covid\\Data\\Data\\train\\COVID19"
TRAIN_NORMAL_PATH = "C:\\Users\\KIIT\\Desktop\\MachineLearning\\Paper2\\covid\\Data\\Data\\train\\NORMAL"
TRAIN_PNE_PATH = "C:\\Users\\KIIT\\Desktop\\MachineLearning\\Paper2\\covid\\Data\\Data\\train\\PNEUMONIA"


VAL_NORMAL_PATH = "C:\\Users\\KIIT\\Desktop\\MachineLearning\\Paper2\\covid\\Data\\Data\\test\\NORMAL"
VAL_PNEU_PATH = "C:\\Users\\KIIT\\Desktop\\MachineLearning\\Paper2\\covid\\Data\\Data\\test\\PNEUMONIA"
VAL_COVID_PATH = "C:\\Users\\KIIT\\Desktop\\MachineLearning\\Paper2\\covid\\Data\\Data\\test\\COVID19"


train_datagen = image.ImageDataGenerator(
    rescale = 1./255,
   # shear_range = 0.2,
   # zoom_range = 0.2,
    #horizontal_flip = True,
)
test_datagen = image.ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'categorical')



train_generator.class_indices



validation_generator = test_datagen.flow_from_directory(
    VAL_PATH,
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'categorical')


annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=100)
mc = ModelCheckpoint("own.h5", monitor='val_loss',save_best_only=True, mode='min',verbose=1)



input_t = Input(shape=(224, 224, 3))
IMAGE_SIZE = [224, 224]

from keras.layers import Input, Lambda, Dense, Flatten, Dropout, BatchNormalization
from keras.models import Sequential
model = ResNet50(classes=3,
      #           activation='softmax',
                 weights='imagenet',
                 include_top=False,
                 input_shape=IMAGE_SIZE + [3]
                 )
# don't train existing weights
for layer in model.layers:
  layer.trainable = False
  
model=Sequential()
model.add(Dense(3,activation="softmax"))
# create a model object
model = Model(inputs=model.input, outputs=prediction)

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


hist = model.fit_generator(
    train_generator,
    epochs=25,
  #  callbacks=[annealer,mc,es],
    steps_per_epoch=int(5144/32),
    validation_data=validation_generator,
    validation_steps = int(1288/32)
)



preds = model.evaluate(validation_generator)
print ("Validation Loss = " + str(preds[0]))
print ("Validation Accuracy = " + str(preds[1]))


plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(["Train_acc","Validation_acc"])
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(["Train_loss","Validation Loss"])
plt.show()


