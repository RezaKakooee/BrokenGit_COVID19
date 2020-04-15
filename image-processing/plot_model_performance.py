# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 18:54:40 2020

@author: Reza
"""
#%%
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from metadata import Classes_Metadata
from setting import Settings

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.utils import to_categorical

#%%
current_dir =  os.getcwd()
imageset_name = Settings().imageset_name
imageset_dir = os.path.join(current_dir, imageset_name)
logfiles_dir = os.path.join(current_dir, 'logfiles')
pretrained_model_name = Settings().pretrained_model_name
logs_dir = os.path.join(logfiles_dir, pretrained_model_name, 'logs')
histories_dir = os.path.join(logfiles_dir, pretrained_model_name, 'histories')
models_dir = os.path.join(logfiles_dir, pretrained_model_name, 'models')
excel_file_path = os.path.join(current_dir, 'data.xlsx')
NUM_TOTAL_EPOCHS = Settings().TOTAL_EPOCHS
initial_epochs = Settings().INITIAL_EPOCHS

#%%
def load_large_imageset(imageset_dir):
    class0_name = Classes_Metadata().class_names[0]
    class1_name = Classes_Metadata().class_names[1]
    class0_dir = os.path.join(imageset_dir, class0_name)
    class1_dir = os.path.join(imageset_dir, class1_name)

    class0_names = os.listdir(class0_dir)
    class1_names = os.listdir(class1_dir)
        
    class0_pathes = [os.path.join(class0_dir, fc_in)  for fc_in in class0_names]
    class1_pathes = [os.path.join(class1_dir, sc_in)  for sc_in in class1_names]
    
    num_class0 = len(class0_pathes)
    num_class1 = len(class1_pathes) 

    num_imgs = num_class0 + num_class1
    print('Number of images: {}'.format(num_imgs))
    print('Class Names: {}, {}'.format(class0_name, class1_name))
    print('Number of unselfies: {} \nNumber of selfies: {}'.format(num_class0, num_class1))
    print('The number of photos in the big class in about {} times more than that of small class'.format(num_class0/num_class1))
    
    IMG_SIZE = Settings().IMG_SIZE
    print('Image size: ', IMG_SIZE)

load_large_imageset(imageset_dir)


#%%
num_samples = 100
def get_smaple_image_pathes(imageset_dir, num_samples):
    class0_name = Classes_Metadata().class_names[0]
    class1_name = Classes_Metadata().class_names[1]
    class0_dir = os.path.join(imageset_dir, class0_name)
    class1_dir = os.path.join(imageset_dir, class1_name)

    class0_names = os.listdir(class0_dir)
    class1_names = os.listdir(class1_dir)
        
    class0_pathes = [os.path.join(class0_dir, fc_in)  for fc_in in class0_names]
    class1_pathes = [os.path.join(class1_dir, sc_in)  for sc_in in class1_names]
    
    num_class0 = len(class0_pathes)
    num_class1 = len(class1_pathes) 

    num_imgs = num_class0 + num_class1
    
    sample_images_class0 = random.sample(class0_pathes, num_samples)
    sample_images_class1 = random.sample(class1_pathes, num_samples)
    
    return sample_images_class0, sample_images_class1
    
sample_images_class0, sample_images_class1 = get_smaple_image_pathes(imageset_dir, num_samples=num_samples)


#%%

def load_sample_images(sample_image_pathes, IMG_SIZE, class_name):
  images_arr = []
  labels_list = []
  if class_name == 'unselfie': 
    label = 0
  else:
    label = 1
        
  for i, img_path in enumerate(sample_image_pathes):
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = img_to_array(img)
    img = img/255.0
    images_arr.append(img)
    
    labels_list.append(label)

  images_arr = np.array(images_arr)
  labels = np.array(labels_list)
    
  return images_arr, labels

IMG_SIZE = Settings().IMG_SIZE
class0_name = 'unselfie'
class1_name = 'selfie'
class0_images_arr, class0_labels = load_sample_images(sample_images_class0, IMG_SIZE, class0_name)
class1_images_arr, class1_labels = load_sample_images(sample_images_class1, IMG_SIZE, class1_name)

#%%
def load_from_saved_models(best_fold, models_dir):

    tune_model_dir = os.path.join(models_dir, 'tune-model')
    tune_model_name = best_fold + 'tune_model.h5'
    tune_model_path = os.path.join(tune_model_dir, tune_model_name)
    tune_model = load_model(tune_model_path)
    
    return tune_model

tune_model = load_from_saved_models(best_fold='', models_dir=models_dir)



histories = tuple(os.listdir(histories_dir))[-2:]


#%%
recall_list = []
precision_list = []
val_recall_list = []
val_precision_list = []
plt.figure(figsize=(10, 10))
historyb_name = histories[0]
historyt_name = histories[1]

historyb_path = os.path.join(histories_dir, historyb_name)
historyt_path = os.path.join(histories_dir, historyt_name)

history_base = np.load(historyb_path, allow_pickle=True)
history_tune = np.load(historyt_path, allow_pickle=True)

history_base = history_base.item()
history_tune = history_tune.item()

historyb = dict()
historyt = dict()

keysb = history_base.keys()
keyst = history_tune.keys()
    
for keyb in keysb:
    if "val_precision" in keyb:
        historyb['val_precision'] = history_base[keyb]
        val_precision = history_base[keyb]
    elif "precision" in keyb:
        historyb['precision'] = history_base[keyb]
        precision = history_base[keyb]
        
    if "val_recall" in keyb:
        historyb['val_recall'] = history_base[keyb]
        val_recall = history_base[keyb]
    elif "recall" in keyb:
        historyb['recall'] = history_base[keyb]
        recall = history_base[keyb]
            
            
for keyt in keyst:
    if "val_precision" in keyt:
        historyt['val_precision'] = history_tune[keyt]
        val_precision += history_tune[keyt]
    elif "precision" in keyt:
        historyt['precision'] = history_tune[keyt]
        precision += history_tune[keyt]
        
    if "val_recall" in keyt:
        historyt['val_recall'] = history_tune[keyt]
        val_recall += history_tune[keyt]
    elif "recall" in keyt:
        historyt['recall'] = history_tune[keyt]
        recall += history_tune[keyt]

f1_score = 2 * np.array(precision) * np.array(recall) / (np.array(precision) + np.array(recall))
val_f1_score = 2 * np.array(val_precision) * np.array(val_recall) / (np.array(val_precision) + np.array(val_recall))

f1_score = np.nan_to_num(f1_score)
val_f1_score = np.nan_to_num(val_f1_score)

# plot recall and percision for train and validation
epochs = np.arange(1, NUM_TOTAL_EPOCHS+1)
plt.plot(epochs, precision, label='Training Precision', marker='d')
plt.plot(epochs, val_precision, label='Validation Precision', marker='x')
plt.plot(epochs, recall, label='Training Recall', marker='o')
plt.plot(epochs, val_recall, label='Validation Recall', marker='P')
plt.plot(epochs, f1_score, label='Training F1 Score', marker='+', linestyle='dashed')
plt.plot(epochs, val_f1_score, label='Validation F1 Score', marker='*', linestyle='dashed')

plt.ylim([0.0, 1])
plt.plot([initial_epochs,initial_epochs],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper left')
plt.xticks(np.arange(1, NUM_TOTAL_EPOCHS+1))

plt.title("Split {}".format(i+1))
    
plt.arrow(1, 0.2, 4, 0, length_includes_head=True,
    head_width=0.1, head_length=0.3)
plt.arrow(4.7, 0.2, -3.7, 0, length_includes_head=True,
    head_width=0.1, head_length=0.3)

plt.arrow(6, 0.2, 4, 0, length_includes_head=True,
    head_width=0.1, head_length=0.3)
plt.arrow(9.7, 0.2, -3.7, 0, length_includes_head=True,
    head_width=0.1, head_length=0.3)

plt.text(2.2, 0.25, 'Base Model')
plt.text(7.2, 0.25, 'Fine Tune Model')

plt.show()
