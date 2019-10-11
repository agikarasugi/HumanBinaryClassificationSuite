#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
from sklearn.model_selection import train_test_split
import numpy as np


# In[2]:


dataset_path = './dataset'

image_size_y = 50
image_size_x = 80

image_data = []
image_class = []


# In[3]:


for i, dir in enumerate(os.listdir(dataset_path)):
    class_dir = (dataset_path + '/' + dir)
    for image in os.listdir(class_dir):
        image_file = class_dir + '/' + image
        img = cv2.imread(image_file)
        res = cv2.resize(img, dsize=(image_size_y, image_size_x), interpolation=cv2.INTER_CUBIC)
        image_data.append(res)
        image_class.append((i - 1) * -1)
        
image_data = np.array(image_data, dtype='float32')
image_class = np.array(image_class, dtype='float64')


# In[4]:


train_image, test_image, train_label, test_label = train_test_split(image_data,
                                                                    image_class,
                                                                    test_size=0.2)


# In[5]:


import pickle

file = open('test_data.pkl', 'wb')
test_set = (test_image, test_label)
pickle.dump(test_set, file)
file.close()

file = open('train_data.pkl', 'wb')
train_set = (train_image, train_label)
pickle.dump(train_set, file)
file.close()

