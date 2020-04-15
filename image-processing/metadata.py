import os
import numpy as np
from setting import Settings

class Classes_Metadata(object):
    def __init__(self):
        ##### imageset info
        self.settings = Settings()
        self.imageset_name = self.settings.imageset_name
        self.class_names = self.settings.class_names
        
        ##### model name
        self.pretrained_model_name = self.settings.pretrained_model_name
        
        ##### hyperparameters
        self.IMG_SIZE = self.settings.IMG_SIZE
        self.BATCH_SIZE = self.settings.BATCH_SIZE
            
        self.INITIAL_EPOCHS = self.settings.INITIAL_EPOCHS
        self.FINE_TUNE_EPOCHS = self.settings.FINE_TUNE_EPOCHS
        self.TOTAL_EPOCHS =  self.settings.TOTAL_EPOCHS
            
        self.base_learning_rate = self.settings.base_learning_rate
        self.tune_learning_rate = self.settings.tune_learning_rate
        
        self.fine_tune_at = self.settings.fine_tune_at
        
        # cross validation
        self.use_cv = self.settings.use_cv
        self.n_split = self.settings.n_split
        
        self.directories()

    def directories(self):
        self.current_dir = os.getcwd()
        self.imageset_dir = os.path.join(self.current_dir, self.imageset_name)
        self.which_imageset = 'smallset' if 'small' in self.imageset_name else 'large'
        
        self.logfiles_dir = os.path.join(self.current_dir, 'logfiles')
        
        if not os.path.exists(self.logfiles_dir):
            os.makedirs(self.logfiles_dir)
        
        self.models_dir = os.path.join(self.logfiles_dir, self.pretrained_model_name, 'models')
        self.base_model_dir = os.path.join(self.models_dir, 'base-model')
        self.tune_model_dir = os.path.join(self.models_dir, 'tune-model')
      
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            
        if not os.path.exists(self.base_model_dir):
            os.makedirs(self.base_model_dir)
    
        if not os.path.exists(self.tune_model_dir):
            os.makedirs(self.tune_model_dir)
      
        self.excel_file_path = os.path.join(self.current_dir, 'data.xlsx')

        #### pathes for saving TB log
        self.logs_dir = os.path.join(self.logfiles_dir, self.pretrained_model_name, 'logs')
        self.histories_dir = os.path.join(self.logfiles_dir, self.pretrained_model_name, 'histories')
#        self.images_info_dir = os.path.join(self.logfiles_dir, "images-info", self.which_imageset)
        
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
            
        if not os.path.exists(self.histories_dir):
            os.makedirs(self.histories_dir)
            
        class0_dir = os.path.join(self.imageset_dir, self.class_names[0])
        class1_dir = os.path.join(self.imageset_dir, self.class_names[1])
        
        class0_names = os.listdir(class0_dir)
        class1_names = os.listdir(class1_dir)
        
        self.class0_pathes = [os.path.join(class0_dir, fc_in)  for fc_in in class0_names]
        self.class1_pathes = [os.path.join(class1_dir, sc_in)  for sc_in in class1_names]
        

# metadata = Classes_Metadata()