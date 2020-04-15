# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 23:53:36 2020

@author: rkako
"""

import os


class Params:
    def __init__(self, working_dir, COLAB=False):
        # Directories ans Pathes
        self.current_dir = os.getcwd()
        self.working_dir = self.current_dir if not COLAB else working_dir
        self.imageset_dir = os.path.join(self.working_dir, 'covid-chestxray-dataset/images')
        self.csv_path = os.path.join(self.working_dir, "covid-chestxray-dataset", "metadata.csv")
        
                
        self.log_dir_name = 'logs'
        self.log_dir = os.path.join(self.working_dir, self.log_dir_name)
        self.sprite_image_name = "sprite.png"  
        self.sprite_image_path =  os.path.join(self.log_dir, self.sprite_image_name)
        self.metadata_name = "metadata.tsv"
        self.metadata_path =  os.path.join(self.log_dir, self.metadata_name)

        # Images
        self.img_targ_H = 299
        self.img_targ_W = 299
        self.embeding_images_with = 'sim'# 'sim'# 'feat'# 'img'# 
        self.similarity_metric = 'cosine'
        
        self.IMG_SIZE = 255
        
        self.BATCH_SIZE = 32
            
        self.INITIAL_EPOCHS = 5 # the number of epochs for training the base model
        self.FINE_TUNE_EPOCHS = 15 # the number of epochs for training the fine tune model
        self.TOTAL_EPOCHS =  self.INITIAL_EPOCHS + self.FINE_TUNE_EPOCHS
            
        self.base_learning_rate = 0.0001
        self.tune_learning_rate = self.base_learning_rate/10
        
        self.fine_tune_at = 275
        
        # cross validation
        self.use_cv = 0
        self.n_split = 3
        
        
                
#params = Params()