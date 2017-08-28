# -*- coding: utf-8 -*-

import pickle
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers import Flatten, Dense, Lambda
from keras.layers import MaxPooling2D, Activation


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

# Todo: args
image_path = "C://Users//joonsup//git//dataset//images"
number_of_epochs = 3
number_of_samples_per_epoch = 300
number_of_validation_samples = 64
if __name__ == "__main__":

    import json
    from generator.image_augment import CarAugmentor
    from generator.image_preprocess import Preprocessor
    from generator.generator import ImgGenerator
    with open('annotation.json', 'r') as fp:
        anns = json.load(fp)

    # Todo : train / valid annotation file을 나누고, img_generator instance 를 2개 생성하자.
    gen = ImgGenerator(image_path, anns, CarAugmentor(), Preprocessor())
    train_gen = gen.next_batch()
    validation_gen = gen.next_batch()
    
    model = build_model()
    
    history_object = model.fit_generator(train_gen,
                                         samples_per_epoch=number_of_samples_per_epoch,
                                         nb_epoch=number_of_epochs,
                                         validation_data=validation_gen,
                                         nb_val_samples=number_of_validation_samples,
                                         verbose=1)

    pickle.dump(history_object.history, open('training_history.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    model.save('model.h5')


