import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_lables), (test_images, test_lables) = fashion_mnist.load_data()

class_name = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'coat', 'Sandal', 'Shirt', 'Bag', 'Sneaker', 'Boot']

print (train_images.shape)
print(len(train_lables))

print(train_lables)
print(test_images.shape)

print(len(test_lables))
for i in range(25):
    # plt.figure()
    plt.imshow(train_images[i])
    plt.colorbar()
    #plt.grid(True)
#plt.show()
train_images = train_images / 255
test_images = test_images / 255

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_name[train_lables[i]])
#plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu), # 128 hidden layer / neuron
    keras.layers.Dense(10, activation=tf.nn.softmax)
    
])

model.compile(optimizer ='adam', loss='sparse_categorical_crossentropy',metrics = ['accuracy'] )

model.fit(train_images, train_lables, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_lables)

print("Test acc is: ", test_acc)
print("Test loss is : ", test_loss)

predictions = model.predict(test_images)
print(predictions[0])
arg_max = np.argmax(predictions[0])
print((arg_max))  # this is means that predictions is a boot or  a class of number 9

print(test_lables[0])

def plot_img(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_name[predicted_label],
    100 * np.max(predictions_array),
    class_name[true_label]
    ), color=color)

def plot_val_arr(i, predictions_arr, true_lable):
    predictions_arr, true_lable = predictions_arr[i], true_lable[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    this_plot = plt.bar(range(10), predictions_arr, color="#777777")
    plt.ylim([0, 1])
    predictions_label = np.argmax(predictions_arr)
    this_plot[predictions_label].set_color('yellow')
    this_plot[true_lable].set_color('blue')


#plt.figure(figsize=(2*2*3,2*4))

for i in range(15):
    plt.subplot(5,6,2*i+1)
    plot_img(i, predictions, test_lables, test_images)
    plt.subplot(5,6,2*i+2)
    plot_val_arr(i, predictions, test_lables)
plt.show()
# Predicting single image
img = test_images[0]
print(img.shape)
img = (np.expand_dims(img, 0))
print(img.shape)
predictions_signle = model.predict(img)
print(predictions_signle)
plot_val_arr(0, predictions_signle, test_lables)
_ = plt.xticks(range(10), class_name, rotation=45)

plt.show()

single_arg_max = np.argmax(predictions_signle[0])
print(single_arg_max)