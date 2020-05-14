# -*- coding: utf-8 -*-
"""
Created on Fri May  2 12:40:35 2020

@author: saikr
"""

# this the code for training the face and non-face dataset I have created manually.

# important things to note:
	
# THE MODEL CREATED HAS BEEN SAVED IN A LOCATION , WHICH WILL AGAIN BE USED DURING TESTING PHASEfrom imutils import paths
import numpy as np
import pandas as pd
import cv2
from keras.preprocessing.image import img_to_array
from random import random
import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np
from tensorflow.keras.utils import to_categorical
from scipy.stats import uniform
from sklearn.model_selection import train_test_split
import keras
import math
from facenetwork import LeNet
import pickle
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

data = []
labels = []
imagePaths = list(paths.list_images("D:\SEM 2-spring 2020\COMPUTER VISION\project-3\FaceNonFace\FaceNonFace\Face"))
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath,-1)
    image = cv2.resize(image, (128, 128))
    image = img_to_array(image)
    data.append(image)
	# extract set of class labels from the image path and update the
	# labels list
   
    l = 1
    labels.append(l)
    
#imagePaths = list(paths.list_images("D:/NonFace"))
imagePaths = list(paths.list_images(r"D:\SEM 2-spring 2020\COMPUTER VISION\project-3\FaceNonFace\FaceNonFace\NonFace"))
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (128, 128))
    image = img_to_array(image)
    data.append(image)
	# extract set of class labels from the image path and update the
	# labels list
    l = 0
    labels.append(l)

data = np.asarray(data)

num_classes = 2

(xTrain, xVal, yTrain, yVal) = train_test_split(data,labels, test_size=0.20)
yTrain = to_categorical(yTrain, num_classes)
yVal = to_categorical(yVal, num_classes)
        
'''print('x_train shape:', xTrain.shape)
   print(xTrain.shape[0], 'train samples')
   print(xTest.shape[0], 'test samples')
   print(xTrain[0].shape, 'image shape')'''
        
        
'''xTrain = xTrain.astype('float32')
        #xVal = xVal.astype('float32')
        xTrain /= 255
        xVal /= 255
        xTrain -= np.mean(xTrain)
        xVal -= np.mean(xVal)
        xTrain /= np.std(xTrain, axis = 0)
        xVal /= np.std(xVal, axis = 0)'''
            
model = LeNet.build(128,128,3, num_classes,2.3e-6,9e-08)
opt = Adam(lr=2.3e-6)
model.compile(loss=categorical_crossentropy,metrics=['accuracy'], optimizer=opt)
EPOCHS = 20
H = model.fit(xTrain, yTrain,
              validation_data=(xVal, yVal),
        	epochs=EPOCHS, verbose=1)




model.save(r"D:\SEM 2-spring 2020\COMPUTER VISION\project-3\FaceNonFace\FaceNonFace")
