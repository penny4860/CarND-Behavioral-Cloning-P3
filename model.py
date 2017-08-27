# -*- coding: utf-8 -*-

import cv2
import numpy as np
import csv
from sklearn.model_selection import train_test_split
import sklearn
from random import shuffle
import os


def plot_history(history_object):
    import matplotlib.pyplot as plt
    # history_object = model.fit_generator(train_generator, samples_per_epoch =
    #     len(train_samples), validation_data = 
    #     validation_generator,
    #     nb_val_samples = len(validation_samples), 
    #     nb_epoch=5, verbose=1)
    
    ### print the keys contained in the history object
    print(history_object.history.keys())
    
    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


def build_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x - 128.0)/ 128.0, input_shape=(64, 64, 3)))
    model.add(Convolution2D(8, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(8, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
     
    # model.add(BatchNormalization())
    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
     
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
     
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
        
    model.add(Flatten())
    #model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.summary()
    model.compile(loss='mse', optimizer='adam')
    return model

    
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import MaxPooling2D, Activation, Cropping2D

number_of_epochs = 8
number_of_samples_per_epoch = 20032
number_of_validation_samples = 6400
if __name__ == "__main__":

    import json
    from generator.image_augment import CarAugmentor
    from generator.generator import ImgGenerator
    with open('annotation.json', 'r') as fp:
        anns = json.load(fp)

    # Todo : train / valid annotation file을 나누고, img_generator instance 를 2개 생성하자.
    gen = ImgGenerator("C://Users//joonsup//git//data//IMG", anns, CarAugmentor())
    train_gen = gen.next_batch()
    validation_gen = gen.next_batch()
    
    model = build_model()
    
#     history = model.fit_generator(train_gen,
#                                   samples_per_epoch=number_of_samples_per_epoch,
#                                   nb_epoch=number_of_epochs,
#                                   validation_data=validation_gen,
#                                   nb_val_samples=number_of_validation_samples,
#                                   verbose=1)

#     # history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2, verbose=2)
#      
#     history_object = model.fit_generator(train_generator,
#                                          samples_per_epoch=len(train_lines)*6,
#                                          validation_data=validation_generator,
#                                          nb_val_samples=len(validation_lines),
#                                          nb_epoch=20,
#                                          verbose=2)
#      
#     plot_history(history_object)
#     model.save('model.h5')


