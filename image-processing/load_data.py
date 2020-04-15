# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 20:03:34 2020

@author: reza
"""
#%%
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras import metrics
from tensorflow.keras.models import load_model


#%%
#### Load the imageset
def load_images(class0_pathes, class1_pathes, IMG_SIZE, logger):
  logger.info("=========== Start loading images =============================")
  print("----- Start loading images")
  
#  class0_pathes = class0_pathes[:2000]
#  class1_pathes = class1_pathes[:2000]
  images_arr = []
  labels_list = []
  
  i = 0
  for img_path in class0_pathes:
    ##### label
    labels_list.append(0)
      
    ##### image
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = img_to_array(img)
    images_arr.append(img)
    
    ##### print
    if i % 5000 == 0:
      logger.info("----- Num of loaded images: {}".format(i))
      print("----- Num of loaded images: {}".format(i))
    
    i += 1
    
  for img_path in class1_pathes:
    ##### label
    labels_list.append(1)
      
    ##### image
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = img_to_array(img)
    images_arr.append(img)
    
    ##### print
    if i % 5000 == 0:
      logger.info("----- Num of loaded images: {}".format(i))
      print("----- Num of loaded images: {}".format(i))
    
    i += 1
      
  ##### convert to array
  images_arr = np.array(images_arr)
  labels = np.array(labels_list)
  
  return images_arr, labels