#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from tensorflow.keras.utils import plot_model
import os


# In[2]:


image_size_y = 50
image_size_x = 80


# In[3]:


file = open('test_data.pkl', 'rb')
test_image, test_label = pickle.load(file)
file.close()

file = open('train_data.pkl', 'rb')
train_image, train_label = pickle.load(file)
file.close()

train_image = train_image.reshape(len(train_image), image_size_x, image_size_y, 3)
train_image = train_image / 255.0

test_image = test_image.reshape(len(test_image), image_size_x, image_size_y, 3)
test_image = test_image / 255.0


# In[4]:


now = datetime.now()
dt_string = now.strftime("%d%m%Y-%H%M")

name = dt_string

foldername = './models/' + name + '/'
os.mkdir(foldername)


# In[5]:


class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, test_data, max_loss_to_save):
        self.test_data = test_data
        self.min_loss = 99.99
        self.max_acc = -1
        self.counter = 0
        self.max_loss_to_save = max_loss_to_save

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

        if acc > self.max_acc:
            self.max_acc = acc

        if loss < self.min_loss:
            self.min_loss = loss

            if loss < self.max_loss_to_save:
                mnow = datetime.now()
                mdt_string = mnow.strftime("%d%m%Y-%H%M")

                mname = 'l{:4.0f}-a{:2.0f}-'.format(loss*10000, acc*100) + dt_string

                filename = foldername + mname

                print('Saving model as {}...\n\n'.format(filename))
                plot_model(model, to_file=(filename+'.png'), show_shapes=True)
                model.save(filename+'.h5')

        if loss > self.min_loss:
            self.counter += 1

            if self.counter > 15 or loss > 0.65:
                print('Model hasn\'t improved in a while, cancelling training')
                self.model.stop_training = True

    def on_train_end(self, logs={}):
        print('\n\n\nSUMMARRY')
        print('========')
        print('Best loss: {}, Best Acc: {}'.format(self.min_loss, self.max_acc))


# In[6]:


model = tf.keras.models.Sequential(layers=[
    tf.keras.layers.Conv2D(16, (3, 3), activation=tf.nn.relu,
                            input_shape=(image_size_x, image_size_y, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(16, (3, 3), activation=tf.nn.leaky_relu),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# In[7]:


model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['acc'])

# model.compile(optimizer='adam',
#                 loss='binary_crossentropy',
#                 metrics=['acc'])

model.summary()


# In[9]:


history = model.fit(train_image, train_label, epochs=15,
          validation_split=0.25, shuffle=True,
          callbacks=[CustomModelCheckpoint((test_image, test_label), 0.30)],
                   verbose=1)

test_loss = model.evaluate(test_image, test_label)


# In[ ]:


print("Loss: {}, Accuracy: {}".format(test_loss[0], test_loss[1]))

filename = foldername + name

print(filename)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(filename + '_gacc.png', dpi=64)
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(filename + '_gloss.png', dpi=64)
plt.show()
