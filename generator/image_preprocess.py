# -*- coding: utf-8 -*-

import numpy as np
import scipy.misc

class Preprocessor(object):
    
    def __init__(self):
        pass

    def preprocess(self, image, top_crop_percent=0.35, bottom_crop_percent=0.1, resize_dim=(64, 64)):
        image = self._crop(image, top_crop_percent, bottom_crop_percent)
        image = self._resize(image, resize_dim)
        return image

    def _crop(self, image, top_percent, bottom_percent):
        assert 0 <= top_percent < 0.5, 'top_percent should be between 0.0 and 0.5'
        assert 0 <= bottom_percent < 0.5, 'top_percent should be between 0.0 and 0.5'
    
        top = int(np.ceil(image.shape[0] * top_percent))
        bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))
        return image[top:bottom, :]

    def _resize(self, image, new_dim):
        return scipy.misc.imresize(image, new_dim)
