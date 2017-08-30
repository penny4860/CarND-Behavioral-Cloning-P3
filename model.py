# -*- coding: utf-8 -*-

import pickle
import cv2
import json
import tensorflow as tf

from random import shuffle

from generator.image_augment import CarAugmentor, NothingAugmentor
from generator.image_preprocess import Preprocessor
from generator.generator import DataGenerator
from model_arch import build_model

"""Usage
> python model.py --image_path dataset//images --n_epochs 2 --training_ratio 0.8
"""

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('image_path', "..//dataset//images", 'Directory containing images')
# flags.DEFINE_string('image_path', 'dataset//images', 'Directory containing images')
flags.DEFINE_integer('n_epochs', 8, 'number of epochs')
flags.DEFINE_float('training_ratio', 0.8, 'ratio of training samples')


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
