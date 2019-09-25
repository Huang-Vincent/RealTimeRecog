{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(60000, 28, 28, 1)\n"
     ]
    }
   ],
   "source": [
    "import keras\n",
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D\n",
    "from keras.datasets import mnist\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "(X_train, y_train), (X_test, y_test) = mnist.load_data()#loading the data\n",
    "\n",
    "#reshaping the dataset\n",
    "X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)\n",
    "X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)\n",
    "input_shape = (28, 28, 1)\n",
    "\n",
    "#Used for convention\n",
    "X_train = X_train.astype('float32')\n",
    "X_test = X_test.astype('float32')\n",
    "\n",
    "#normalize the dataset\n",
    "X_train = X_train/255.0\n",
    "X_test = X_test/255.0\n",
    "\n",
    "print(X_train.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "#convert labels to vectors\n",
    "categoryNum = 10\n",
    "y_train = keras.utils.to_categorical(y_train, categoryNum)\n",
    "y_test = keras.utils.to_categorical(y_test, categoryNum)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From c:\\users\\vinhu\\appdata\\local\\programs\\python\\python37\\lib\\site-packages\\tensorflow\\python\\keras\\layers\\core.py:143: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.\n"
     ]
    }
   ],
   "source": [
    "model = Sequential()\n",
    "\n",
    "model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))\n",
    "model.add(Conv2D(64, (3,3), activation = 'relu'))\n",
    "\n",
    "model.add(MaxPooling2D(pool_size=(2,2)))\n",
    "model.add(Dropout(0.2))\n",
    "\n",
    "model.add(Flatten())\n",
    "\n",
    "model.add(Dense(128, activation='relu'))\n",
    "model.add(Dense(categoryNum, activation='softmax'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(loss = \"categorical_crossentropy\",\n",
    "             optimizer = \"adam\",\n",
    "             metrics = ['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train on 60000 samples, validate on 10000 samples\n",
      "WARNING:tensorflow:From c:\\users\\vinhu\\appdata\\local\\programs\\python\\python37\\lib\\site-packages\\tensorflow\\python\\ops\\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Use tf.cast instead.\n",
      "Epoch 1/10\n",
      "60000/60000 [==============================] - 120s 2ms/sample - loss: 0.1598 - acc: 0.9510 - val_loss: 0.0475 - val_acc: 0.9843\n",
      "Epoch 2/10\n",
      "60000/60000 [==============================] - 122s 2ms/sample - loss: 0.0465 - acc: 0.9855 - val_loss: 0.0400 - val_acc: 0.9871\n",
      "Epoch 3/10\n",
      "60000/60000 [==============================] - 125s 2ms/sample - loss: 0.0299 - acc: 0.9904 - val_loss: 0.0381 - val_acc: 0.9869\n",
      "Epoch 4/10\n",
      "60000/60000 [==============================] - 121s 2ms/sample - loss: 0.0213 - acc: 0.9931 - val_loss: 0.0390 - val_acc: 0.9882\n",
      "Epoch 5/10\n",
      "60000/60000 [==============================] - 104s 2ms/sample - loss: 0.0165 - acc: 0.9948 - val_loss: 0.0321 - val_acc: 0.9903\n",
      "Epoch 6/10\n",
      "60000/60000 [==============================] - 105s 2ms/sample - loss: 0.0122 - acc: 0.9958 - val_loss: 0.0321 - val_acc: 0.9893\n",
      "Epoch 7/10\n",
      "60000/60000 [==============================] - 117s 2ms/sample - loss: 0.0092 - acc: 0.9971 - val_loss: 0.0328 - val_acc: 0.9901\n",
      "Epoch 8/10\n",
      "60000/60000 [==============================] - 124s 2ms/sample - loss: 0.0087 - acc: 0.9971 - val_loss: 0.0363 - val_acc: 0.9894\n",
      "Epoch 9/10\n",
      "60000/60000 [==============================] - 120s 2ms/sample - loss: 0.0091 - acc: 0.9969 - val_loss: 0.0391 - val_acc: 0.9901\n",
      "Epoch 10/10\n",
      "60000/60000 [==============================] - 123s 2ms/sample - loss: 0.0060 - acc: 0.9980 - val_loss: 0.0424 - val_acc: 0.9890\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<tensorflow.python.keras.callbacks.History at 0x1dad808a588>"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_test, y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test loss: 0.04237589347365138\n",
      "Test accuracy: 0.989\n"
     ]
    }
   ],
   "source": [
    "score = model.evaluate(X_test, y_test, verbose=0)\n",
    "print('Test loss:', score[0])\n",
    "print('Test accuracy:', score[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save('cnn.model')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
