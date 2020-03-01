import tensorflow as tf
from tensorflow import keras
from keras import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D
import numpy as np

import datetime


# log_dir = "C:\\logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

x_train = np.load('x_train_v1.npy')
y_train = np.load('y_train_v1.npy')
#x_train= np.array(x_train)

x_train = x_train / 255.0

model = Sequential()
#layer 1 input
model.add(Conv2D(64,(5,5), input_shape= (x_train.shape[1:])))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
# layer 2
model.add(Conv2D(16, (3,3)))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
# layer 3
model.add(Flatten())
model.add(Dense(128))

model.add(Activation("relu"))
model.add(Dropout(0.3))
# layer 4 output
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer ='adam', metrics = ['accuracy'] )

model.fit(x_train, y_train, batch_size=32,epochs=5, validation_split=0.3)

loss_and_metrics = model.evaluate(x_train, y_train, batch_size=128)
print( loss_and_metrics)


