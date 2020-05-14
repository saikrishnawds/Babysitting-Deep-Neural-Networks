
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 11:40:35 2020

@author: saikr
"""
# AGAIN, FOR THE FACE NON FACE DATASET, WE HAVE USED THE SAME LENET5 ARCHITECTURE TO BUILD THHIS FACE DETECTION MODEL

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy


from keras import regularizers

class LeNet:
    def build(width, height, depth,num_classes, learn_r, reg):
        model = Sequential()
        inputShape = (height, width, depth)
        
        
        model.add(Conv2D(6, (5, 5), strides= (1,1), padding="same",
            input_shape=inputShape))
        model.add(Activation("relu"))
        
        model.add(AveragePooling2D(pool_size=(2, 2),strides= (2,2)))
        
        model.add(Conv2D(16, (5, 5), strides= (1,1), padding="valid",
            input_shape=inputShape))
        model.add(Activation("relu"))
        
        model.add(AveragePooling2D(pool_size=(2, 2),strides= (2,2)))

        model.add(Flatten())
        model.add(Dense(120, activation='relu',activity_regularizer=regularizers.l2(reg)))
        model.add(Dense(84, activation='relu',activity_regularizer=regularizers.l2(reg)))
        model.add(Dense(num_classes,activation = 'softmax'))
        
    
        
        # return the constructed network architecture
        return model
    
    

