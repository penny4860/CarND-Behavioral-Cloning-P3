# -*- coding: utf-8 -*-
from abc import abstractmethod

import numpy as np
import cv2
from scipy.stats import bernoulli


class _ImageAugmentor(object):
    
    def __init__(self):
        pass

    @abstractmethod
    def augment(self, image, target):
        return image, target


class NothingAugmentor(_ImageAugmentor):
    def augment(self, image, target):
        return image, target

class CarAugmentor(_ImageAugmentor):
    def augment(self,
                image,
                target,
                top_crop_percent=0.35,
                bottom_crop_percent=0.1,
                resize_dim=(64, 64),
                do_shear_prob=0.9):

        head = bernoulli.rvs(do_shear_prob)
        if head == 1:
            image, target = self._random_shear(image, target)

        image, target = self._random_flip(image, target)
        image = self._random_gamma(image)
        return image, target

    def _random_shear(self, image, steering_angle, shear_range=200):
        rows, cols, ch = image.shape
        dx = np.random.randint(-shear_range, shear_range + 1)
        random_point = [cols / 2 + dx, rows / 2]
        pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
        pts2 = np.float32([[0, rows], [cols, rows], random_point])
        dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
        steering_angle += dsteering
        return image, steering_angle

    def _random_flip(self, image, steering_angle, flipping_prob=0.5):
        head = bernoulli.rvs(flipping_prob)
        if head:
            return np.fliplr(image), -1 * steering_angle
        else:
            return image, steering_angle
        
    def _random_gamma(self, image):
        gamma = np.random.uniform(0.4, 1.5)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
