'''
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 21:46:09 2020

@author: saikr
"""

import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np
from tensorflow.keras.utils import to_categorical
from network import LeNet

(xTrain,yTrain),(xTest,yTest) = datasets.cifar10.load_data()

# UNCOMMENT THE FOLLOWING ,LINES FOR OVERFITTING THE DATA (WITH 20 SAMPLES):
'''
	
'''xTrain = xTrain[0:20]
yTrain = yTrain[0:20]
xTest = xTest[0:10]
yTest = yTest[0:10]'''

'''


num_classes = 10
yTrain = to_categorical(yTrain, num_classes)
yTest = to_categorical(yTest, num_classes)



xTrain = xTrain.astype('float32')
xTest = xTest.astype('float32')
xTrain /= 255
xTest /= 255
xTrain -= np.mean(xTrain)
xTest -= np.mean(xTest)
xTrain /= np.std(xTrain, axis = 0)
xTest /= np.std(xTest, axis = 0)
    
model = LeNet((xTrain[0].shape), num_classes)
EPOCHS = 10

H = model.fit(xTrain, yTrain,
	validation_data=(xTest, yTest),
	epochs=EPOCHS, verbose=1)

'''

########################################################################################

# CODE FOR FINE-COARSE SEARCH OF OPTIMAL LEARNING RATES AND REGULARIZATIONS
import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
from random import random
import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np
from tensorflow.keras.utils import to_categorical
from network import LeNet
from scipy.stats import uniform
from tensorflow import keras
import math




num_classes = 10
        
(x_train,y_train),(x_test,y_test) = datasets.cifar10.load_data()    

  
# this is the data pre-processing 

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train -= np.mean(x_train)
x_test -= np.mean(x_test)
x_train /= np.std(x_train, axis = 0)
x_test /= np.std(x_test, axis = 0)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
            
model = LeNet((x_train[0].shape), num_classes,1e-3,0.0002)


# uncomment the following for data-augmentation:

'''
aug = ImageDataGenerator(rotation_range=35, width_shift_range=0.1,
	          height_shift_range=0.1,zoom_range=0.3,shear_range=0.1, fill_mode="reflect")
              
        valaug = ImageDataGenerator()
        
        model = LeNet((xTrain[0].shape), num_classes,lr,reg)
        EPOCHS = 20
        H = model.fit_generator(aug.flow(xTrain, yTrain, batch_size = 8),
                	validation_data=valaug.flow(xTest, yTest),
                	epochs=EPOCHS, verbose=1)
	
'''
# COMMENT THE FOLLOWING LINES AND JUST RUN THE ALREADY COMMENTED (xTRAIN,yTRAIN),(xTEST,yTEST) line below for obtainig values without fine and coarse cross-validation.

for count in range(2, 3):
    for iteration in range (0,10):
        iteration = iteration+1
        keras.backend.clear_session()
        #(xTrain,yTrain),(xTest,yTest) = datasets.cifar10.load_data()
        reg = random()*pow(10,np.random.randint(-5, 0))
        lr = random()*pow(10,-count)
        print("learning rate: ",lr, "Regularisation", reg)
        model=LeNet(x_train[0].shape, num_classes, lr, reg)
        #print("[INFO] training model... ")
        model.fit(x=x_train, y=y_train, epochs=10, validation_data=(x_test, y_test), verbose=1)
        
H = model.fit(x_train, y_train,validation_data=(x_test, y_test),epochs=EPOCHS, verbose=2)
        
    
    
    

