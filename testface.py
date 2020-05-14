
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 12:40:35 2020

@author: saikr
"""
# this the code for testing the face and non-face dataset I have created manually.

# important things to note:
	
# THE MODEL THAT WAS SAVED IN A LOCATION DURING TESTING PHASE IS LOADED AND THEN TESTED- USING A SEPARATE TESTING DATASET THAT HAS BEEN ALLOCATED BY ME.

from imutils import paths
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
from keras.preprocessing.image import img_to_array
from imutils import paths
import numpy as np
import pickle
import cv2
import pandas as pd
from sklearn.metrics import classification_report
from keras.models import load_model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

def vectorized_result(j):
    e = np.zeros((2, 1))
    e[j] = 1.0
    return e
data = []
labels = []
imagePaths = list(paths.list_images("D:\SEM 2-spring 2020\COMPUTER VISION\project-3\FaceTest"))
for imagePath in imagePaths:
    image = cv2.imread(imagePath,-1)
    image = cv2.resize(image, (128, 128))
    image = img_to_array(image)
    data.append(image)
   
    l = 1
    labels.append(l)
    
imagePaths = list(paths.list_images("D:\SEM 2-spring 2020\COMPUTER VISION\project-3\NonFaceTest"))
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (128, 128))
    image = img_to_array(image)
    data.append(image)
    l = 0
    labels.append(l)

print("[INFO] loading network...")
model = load_model(r"D:\SEM 2-spring 2020\COMPUTER VISION\project-3")
opt = Adam(lr=2.3e-6)
model.compile(loss=categorical_crossentropy,metrics=['accuracy'], optimizer=opt)

fp = []
result=[0]*len(data)
i=0
for image in data:
    image = np.expand_dims(image, axis=0)
    result_temp=model.predict(image)[0]
   
    predictions=int(np.argmax(result_temp))
    fp.append(predictions)
    result[i]=vectorized_result(predictions).T
    i+=1    
    
count = 0
for i in range (0,len(labels),1):
    
    if fp[i] == labels[i]:
        count +=1


y_true = labels
y_pred = fp  
target_names = ['Face', 'NonFace']
print(classification_report(y_true, y_pred, target_names=target_names))


