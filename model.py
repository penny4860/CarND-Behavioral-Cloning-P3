# -*- coding: utf-8 -*-

import pickle
import cv2
import json
import tensorflow as tf

from random import shuffle
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers import Flatten, Dense, Lambda
from keras.layers import MaxPooling2D, Activation

from generator.image_augment import CarAugmentor, NothingAugmentor
from generator.image_preprocess import Preprocessor
from generator.generator import DataGenerator

"""Usage
> python model.py --image_path dataset//images --n_epochs 2 --training_ratio 0.8
"""

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('image_path', "C://Users//joonsup//git//dataset//images", 'Directory containing images')
# flags.DEFINE_string('image_path', 'dataset//images', 'Directory containing images')
flags.DEFINE_integer('n_epochs', 8, 'number of epochs')
flags.DEFINE_float('training_ratio', 0.8, 'ratio of training samples')


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


def main(_):
    with open('annotation.json', 'r') as fp:
        anns = json.load(fp)
    shuffle(anns)
    
    n_train_samples = int(len(anns)*FLAGS.training_ratio)
    train_annotations = anns[:n_train_samples]
    valid_annotations = anns[n_train_samples:]

    # validation generator : augment (x)
    # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    test_data_generator = DataGenerator(FLAGS.image_path, valid_annotations, NothingAugmentor(), Preprocessor())
    train_data_generator = DataGenerator(FLAGS.image_path, train_annotations, CarAugmentor(), Preprocessor())

    train_gen = train_data_generator.next_batch()
    validation_gen = test_data_generator.next_batch()
     
    model = build_model()
     
    history_object = model.fit_generator(train_gen,
                                         samples_per_epoch=len(train_annotations),
                                         nb_epoch=FLAGS.n_epochs,
                                         validation_data=validation_gen,
                                         nb_val_samples=len(valid_annotations),
                                         verbose=1)

    pickle.dump(history_object.history, open('training_history.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    model.save('model.h5')


if __name__ == '__main__':
    tf.app.run()
