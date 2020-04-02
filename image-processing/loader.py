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
        self.params = Params(working_dir, COLAB=COLAB)
    
    def load_csv(self, filter_date=False):
        ### Read CSV
        csv = pd.read_csv(self.params.csv_path)
        
        ### Only keep frontal view
        idx_pa = csv["view"].isin(["PA"])
        csv = csv[idx_pa]
        
        ### Make labels by combining findign and survival
        Labes = [f+'_'+'O' if str(s) == 'nan' else f+'_'+str(s) for f, s in zip(csv['finding'], csv['survival'])]
        
        ### Remove unnecessary columns
        cvs_columns = csv.columns
        csv_unremovable_cols = ['patientid', 'sex', 'age', 'finding', 'survival',
                                'date', 'filename']
        
        # removable_cols = ['location', 'doi', ' url', 'license', 'view', 'modality',
        #                    'offset', 'clinical notes', 'other notes', 'Unnamed: 22',
        #                    'finding', 'survival']
        
        removable_cols = list(set(cvs_columns) - set(csv_unremovable_cols))
        for col in removable_cols:
            csv.drop(col, axis=1, inplace=True)
        
        ### Insert label columns in the final column
        num_cols = len(csv.columns)
        csv.insert(num_cols, 'label', Labes)
        
        ### Convert dates from string to datetype
        for ind in csv.index:
            date_str = dateutil.parser.parse(csv['date'][ind]).strftime("%Y-%m-%d")
            csv['date'][ind] = date_str
        
        ### Remove old data 
        if filter_date:
            for ind in csv.index:
                date = datetime.strptime(csv['date'][ind], '%Y-%m-%d').date()
                if date.year < 2020:
                    csv.drop(ind, inplace=True)        
        
        self.csv = csv
        return csv

    def normalize(self, image, maxval=255):
        """Scales images to be roughly [-1024 1024]."""
        image = (2 * (image.astype(np.float32) / maxval) - 1.) * 1024
        #image = image / np.std(image)
        return image
    
    def rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
  
    def get_labels(self):
        return self.csv['label'].values
      
    def load_images(self, change_img_shape=False):
        csv = self.load_csv()
        
        # get image names
        image_names = csv['filename']
        num_images = len(image_names)
        # make image pathes
        image_pathes = [os.path.join(self.params.imageset_dir, img_name) for img_name in image_names]
      
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
        images_mat = np.array([self.rgb2gray(img) for img in images_arr])
        images_vec = images_mat.reshape(num_images, self.params.img_targ_H * self.params.img_targ_W)
        images_list = list(images_arr)
      
        labels = self.get_labels()
        
        return num_images, image_pathes, image_names, images_vec, images_list, images_mat, images_arr, labels
