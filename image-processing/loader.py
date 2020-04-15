# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 04:11:52 2020

@author: rkako
"""
#%%
import os
import pandas as pd
import numpy as np
from datetime import datetime
import dateutil.parser
from settings import Params

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

#%% self = Loader
class Loader:
    def __init__(self, working_dir, COLAB=False):
        self.current_dir = os.getcwd()
        self.params = Params(working_dir, COLAB=COLAB)
    
    def _load_csv(self):
        ### Read CSV
        metadata_pub_df = pd.read_csv('metadata_pub.csv')
        metadata_uog_df = pd.read_csv('metadata_uog.csv')
        
        csv = pd.concat([metadata_pub_df, metadata_uog_df])
        self.csv = csv
        return csv

    def _normalize(self, image, maxval=255):
        """Scales images to be roughly [-1024 1024]."""
        image = (2 * (image.astype(np.float32) / maxval) - 1.) * 1024
        #image = image / np.std(image)
        return image
    
    def _rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
  
    def _get_labels(self):
        return self.csv['label'].values
      
    def load_images(self, change_img_shape=False):
        csv = self._load_csv()
        
        # get image names
        image_names = csv['filename'].values
        num_images = len(image_names)
        # get image paths prefix
        image_paths_postfix = csv['path_postfix'].values
        
        # make image pathes
        image_pathes = image_paths_postfix
        
        # convert images to other formats
        images_arr = []
        img_shapes = []
        for img_path in image_pathes:
            img = load_img(img_path, target_size=(self.params.img_targ_H, self.params.img_targ_W))
            img = img_to_array(img)
            img /= 255.0
                
            if change_img_shape:
                # Check that images are 2D arrays
                if len(img.shape) > 2:
                    img = img[:, :, 0]
                if len(img.shape) < 2:
                    print("error, dimension lower than 2 for image")
                # Add color channel
                img = img[:, :, None]        
                
            images_arr.append(img)
            img_shapes.append(img.shape)
            
        images_arr = np.array(images_arr)
        images_mat = np.array([self._rgb2gray(img) for img in images_arr])
        images_vec = images_mat.reshape(num_images, self.params.img_targ_H * self.params.img_targ_W)
        images_list = list(images_arr)
      
        labels = self._get_labels()
        
        return num_images, image_pathes, image_names, images_vec, images_list, images_mat, images_arr, labels
