import tensorflow as tf
from tensorflow import keras
from keras import metrics

from matplotlib import pyplot as plt         
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# normailizing data
train_images = tf.keras.utils.normalize(train_images, axis = 1)
test_images = tf.keras.utils.normalize(test_images, axis = 1)

# creating model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
activation = tf.nn.relu
activation_soft = tf.nn.softmax
model.add(tf.keras.layers.Dense(128,activation ))
model.add(tf.keras.layers.Dense(10,activation_soft ))

# defining parameter for traning data
model.compile(optimizer ='adam', loss='sparse_categorical_crossentropy',metrics = ['accuracy'] )

# training the model
model.fit(train_images, train_labels, epochs = 3)

# Calculating the validation loss 
val_loss, val_accuracy = model.evaluate(test_images, test_labels) 
print("val_loss: ", val_loss, "Val accurarcy", val_accuracy)




color_map = plt.cm.binary
plt.imshow(train_images[198],color_map)
plt.show()
print (train_images[198])
#print("len of test_images: ",len(test_images))
#print("len of test_lables: ",len(test_labels))


# saving the model
model.save('test.model')
# loading model
new_model = tf.keras.models.load_model('test.model')

# making prediction
predictions  = new_model.predict(test_images)
print(" predictions: ", predictions)

print(np.argmax(predictions[198]))

plt.imshow(test_images[198])
plt.show()
