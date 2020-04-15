# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:46:50 2020

@author: reza
"""
import os
from pathlib import Path
class Settings(object):
    def __init__(self):
        self.imageset_name = 'imageset-large'
        self.testset_name = 'testset'
        self.class_names = ['unselfie', 'selfie']
        self.IMG_SIZE = 255
        
        self.testset_class0_name = self.class_names[0]
        self.testset_class1_name = self.class_names[1]

        # directories
        self.current_dir = os.getcwd()
        self.testset_dir = os.path.join(Path(self.current_dir).parents[0], self.testset_name)

        self.testset_class0_dir = os.path.join(self.testset_dir, self.testset_class0_name)
        self.testset_class1_dir = os.path.join(self.testset_dir, self.testset_class1_name)
        
        self.image_names_class0 = os.listdir(self.testset_class0_dir)
        self.image_names_class1 = os.listdir(self.testset_class1_dir)
        self.image_pathes_class0 = [os.path.join(self.testset_class0_dir, name) for name in self.image_names_class0]
        self.image_pathes_class1 = [os.path.join(self.testset_class1_dir, name) for name in self.image_names_class1]
    
        ##### model name
        self.pretrained_model_name = 'InceptionV3'

        self.saved_model_name = 'tune_model.h5'
        self.saved_model_dir = os.path.join(Path(self.current_dir).parents[0], 'logfiles/InceptionV3/models/tune-model')
        self.saved_model_path = os.path.join(self.saved_model_dir, self.saved_model_name)

        ##### arrays dir
        self.output_dir = os.path.join(self.current_dir, 'output_arrays')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        ### MPL
        self.labels = ['unselfie', 'unselfie', 'unselfie', 'unselfie',
                       'unselfie', 'unselfie', 'unselfie', 'unselfie',
                       'selfie', 'selfie', 'selfie', 'selfie',
                       'selfie', 'selfie', 'selfie', 'selfie']
        
        self.predicts = ['unselfie', 'unselfie', 'unselfie', 'unselfie',
                         'selfie', 'selfie', 'selfie', 'selfie',
                         'unselfie', 'unselfie', 'unselfie', 'unselfie',
                         'selfie', 'selfie', 'selfie', 'selfie']