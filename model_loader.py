import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

rc = {"axes.spines.left" : False,
      "axes.spines.right" : False,
      "axes.spines.bottom" : False,
      "axes.spines.top" : False,
      "xtick.bottom" : False,
      "xtick.labelbottom" : False,
      "ytick.labelleft" : False,
      "ytick.left" : False}
plt.rcParams.update(rc)


import pickle

file = open('test_data.pkl', 'rb')
test_image, test_label = pickle.load(file)
file.close()


# Recreate the exact same model, including its weights and the optimizer
model_dir = './models/20102019-0940/' + 'l2883-a88-20102019-0940.h5'
model = keras.models.load_model(model_dir, custom_objects={'leaky_relu': tf.nn.leaky_relu})

# Show the model architecture
model.summary()


image_size_y = 50
image_size_x = 80

test_image = test_image.reshape(len(test_image), image_size_x, image_size_y, 3)
test_image = test_image / 255.0

test_loss = model.evaluate(test_image, test_label)
print("Loss: {}, Accuracy: {}".format(test_loss[0], test_loss[1]))

classifications = model.predict(test_image)

import random

curr_start = random.randint(0, 1000)
curr_end = curr_start + 10
for preview in range(2):
    idx = [x for x in range(curr_start, curr_end)]

    fig, axs = plt.subplots(2, 5)

    count = 0
    for i in range(2):
        for j in range(5):
            axs[i, j].imshow(test_image[idx[count]])
            prob = classifications[idx[count]]
            axs[i, j].set_title('{:.2f}'.format(prob.item()))
            count += 1

    plt.show()

    curr_end += 10
    curr_start += 10

f, axarr = plt.subplots(3,4)

FIRST_IMAGE= random.randint(0, 1000)
SECOND_IMAGE= random.randint(0, 1000)
THIRD_IMAGE= random.randint(0, 1000)
CONVOLUTION_NUMBER = 2

from tensorflow.keras import models

layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
for x in range(0,4):
    f1 = activation_model.predict(test_image[FIRST_IMAGE].reshape(1, image_size_x, image_size_y, 3))[x]
    axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='gray_r')
    axarr[0,x].grid(False)
    f2 = activation_model.predict(test_image[SECOND_IMAGE].reshape(1, image_size_x, image_size_y, 3))[x]
    axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='gray_r')
    axarr[1,x].grid(False)
    f3 = activation_model.predict(test_image[THIRD_IMAGE].reshape(1, image_size_x, image_size_y, 3))[x]
    axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='gray_r')
    axarr[2,x].grid(False)

    plt.show()
