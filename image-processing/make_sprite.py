# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 00:24:45 2020

@author: rkako
"""
import numpy as np


def images_to_sprite(images):
    if len(images.shape) == 3:
        images = np.tile(images[...,np.newaxis], (1,1,1,3))
    images = images.astype(np.float32)
    min = np.min(images.reshape((images.shape[0], -1)), axis=1)
    images = (images.transpose(1,2,3,0) - min).transpose(3,0,1,2)
    max = np.max(images.reshape((images.shape[0], -1)), axis=1)
    images = (images.transpose(1,2,3,0) / max).transpose(3,0,1,2)

    n = int(np.ceil(np.sqrt(images.shape[0])))
    padding = ((0, n ** 2 - images.shape[0]), (0, 0),
            (0, 0)) + ((0, 0),) * (images.ndim - 3)
    images = np.pad(images, padding, mode='constant',
            constant_values=0)
    # Tile the individual thumbnails into an image.
    images = images.reshape((n, n) + images.shape[1:]).transpose((0, 2, 1, 3)
            + tuple(range(4, images.ndim + 1)))
    images = images.reshape((n * images.shape[1], n * images.shape[3]) + images.shape[4:])
    images = (images * 255).astype(np.uint8)
    return images/255.0
