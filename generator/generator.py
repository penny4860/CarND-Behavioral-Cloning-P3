# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from .image_preprocess import Preprocessor

class ImgGenerator(object):
    """generate batch images from image files"""
    
    def __init__(self, image_directory, annotations, image_augmentor, preprocessor=Preprocessor()):
        """
        # Args
            image_directory : str
                The directory location of image files.

            annotations : list of dictionary
                Annotations including filename and its target label.

            augmentor : _ImageAugmentor inatance
                
        """
        self._image_dir = image_directory
        self._annotations = annotations
        self._augmentor = image_augmentor
        self._preprocessor = preprocessor
    
    def next_batch(self, batch_size=32):
        while True:
            X_batch = []
            y_batch = []
            image_files, targets = self._get_next_files(batch_size)

            for image_file ,target in zip(image_files, targets):
                img_path = os.path.join(self._image_dir, image_file)
                image = plt.imread(img_path)

                image, target = self._augment(image, target)
                image = self._preprocess(image)
                X_batch.append(image)
                y_batch.append(target)
     
            assert len(X_batch) == batch_size, 'len(X_batch) == batch_size should be True'
            yield np.array(X_batch), np.array(y_batch)
 
    def _get_next_files(self, batch_size):
        n_files = len(self._annotations)
        rnd_indices = np.random.randint(0, n_files, batch_size)

        image_files = []
        targets = []     
        for index in rnd_indices:
            image_file = self._annotations[index]['filename']
            target = self._annotations[index]['target']
            image_files.append(image_file)
            targets.append(target)
            
        return image_files, targets

    def _augment(self, image, target):
        image, target = self._augmentor.augment(image, target)
        return image, target
    
    def _preprocess(self, image):
        return self._preprocessor.preprocess(image)

