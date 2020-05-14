# -*- coding: utf-8 -*-
"""
Created on Fri May  1 12:40:35 2020

@author: saikr
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D,BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam


opt=Adam(learning_rate=10**-3)

# 0.001269 lr, reg= 0.0005


# un-comment the lines in between if you need to check weight initialization using Xavier and batch normalization
'''
class LeNet(Sequential):
    def __init__(self, input_shape, nb_classes):
        super().__init__()

        self.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape, padding="same"))# tanh changed to relu
        #self.add(Conv2D(6, kernel_initializer='glorot_uniform', kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape, padding="same"))# tanh changed to relu
        self.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid')) # tanh changed to relu
        #self.add(Conv2D(16, kernel_initializer='glorot_uniform', kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid')) # tanh changed to relu
		self.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(Flatten())
        self.add(Dense(120, activation='relu',activity_regularizer=regularizers.l2(2e-3)))
        #self.add(Dense(120, kernel_initializer='glorot_uniform', activation='relu',activity_regularizer=regularizers.l2(2e-3)))
		self.add(Dense(84, activation='relu',activity_regularizer=regularizers.l2(2e-3)))
        #self.add(Dense(84, kernel_initializer='glorot_uniform', activation='relu',activity_regularizer=regularizers.l2(2e-3)))
		self.add(BatchNormalization())
		self.add(Dense(nb_classes, activation='softmax'))
		
        self.compile(optimizer=opt,loss=categorical_crossentropy,metrics=['accuracy'])'''

class LeNet(Sequential):
    def __init__(self, input_shape, nb_classes,learn_r,reg):
        super().__init__()

        self.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape, padding="same"))
        self.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'))
        self.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(Flatten())
        self.add(Dense(120, activation='relu',activity_regularizer=regularizers.l2(reg)))
        self.add(Dense(84, activation='relu',activity_regularizer=regularizers.l2(reg)))
		#self.add(BatchNormalization())
        self.add(Dense(nb_classes, activation='softmax'))
        opt = Adam(lr=learn_r)
        self.compile(loss=categorical_crossentropy,metrics=['accuracy'], optimizer=opt)
	           

        #self.compile(optimizer='adam',loss=,lr=learn_r)