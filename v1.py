import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
import random
from numpy import save, load
import pickle
datadir = "image"
Categories = ["IMG_0521-3", "IMG_0525-9","IMG_0530-5","IMG_0536-7","IMG_0538-40","IMG_0544-6","IMG_0547-9"]

img_size = 70
for cg in Categories:
    path = os.path.join(datadir, cg)  # Path for dog or cat
    for img in os.listdir(path):
        img_array = cv.imread(os.path.join(path, img), cv.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap="gray")
        plt.show()

        new_array = cv.resize(img_array,(img_size,img_size))
        plt.imshow(new_array, cmap="gray")
        plt.show()
        print(img_array.shape)
        break
    break

  
training_data = []
def create_training_data():
    for cg in Categories:
        path = os.path.join(datadir, cg)  # Path for dog or cat
        class_name = Categories.index(cg)
        for img in os.listdir(path):
            try:
                img_array = cv.imread(os.path.join(path, img), cv.IMREAD_GRAYSCALE)
                new_array = cv.resize(img_array,(img_size,img_size))
                training_data.append([new_array, class_name])
            except Exception as ex:
                pass
create_training_data()
print("the len of training data is: ", len(training_data))
random.shuffle(training_data)
for sample in training_data:
    print(sample[1])
x_train = []
y_train = []

for features, label in training_data:
    x_train.append(features)
    y_train.append(label)
x_train = np.array(x_train).reshape(-1, img_size, img_size, 1)


save('x_train_v1.npy', x_train)

save('y_train_v1.npy',y_train)

x_train =load('x_train_v1.npy')
# pickle_out = open('x_train.pickle', "wb")
# pickle.dump(x_train, pickle_out)
# pickle_out.close()

# pickle_out = open('y_train.pickle', "wb")
# pickle.dump(y_train, pickle_out)
# pickle_out.close()
# pickle_in = open("x_train.pickle", "rb")
# x_train = pickle.load(pickle_in)

print(x_train[1])
