import numpy as np
import pandas as pd
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import os
classes = 3
# CNN structure
flower_recognizer_model = Sequential()
flower_recognizer_model.add(Conv2D(64,(3,3), input_shape = (100,100,3),strides=2,activation = 'relu'))
flower_recognizer_model.add(Conv2D(64,(3,3),strides=2,activation = 'relu'))
flower_recognizer_model.add(Conv2D(64,(3,3),activation = 'relu'))
flower_recognizer_model.add(Conv2D(64,(3,3),activation = 'relu'))
flower_recognizer_model.add(Conv2D(64,(3,3),activation = 'relu'))
flower_recognizer_model.add(Flatten())
flower_recognizer_model.add(Dense(128,activation='relu'))
# Output layer
flower_recognizer_model.add(Dense(classes,activation='softmax'))

flower_recognizer_model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

flower_recognizer_model.summary()