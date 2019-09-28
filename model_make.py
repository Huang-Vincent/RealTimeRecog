import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.datasets import mnist
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()#loading the data

#reshaping the dataset
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

#Used for convention
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#normalize the dataset
X_train = X_train/255.0
X_test = X_test/255.0

print(X_train.shape)

#convert labels to vectors
categoryNum = 10
y_train = keras.utils.to_categorical(y_train, categoryNum)
y_test = keras.utils.to_categorical(y_test, categoryNum)

#Creating the model
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))
model.add(Conv2D(64, (3,3), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(categoryNum, activation='softmax'))

#Compiling the model
model.compile(loss = "categorical_crossentropy",
             optimizer = "adam",
             metrics = ['accuracy'])

model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_test, y_test))
