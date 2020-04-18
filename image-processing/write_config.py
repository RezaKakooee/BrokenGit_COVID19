# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:58:56 2020

@author: rkako
"""
import os
import configparser

config = configparser.ConfigParser()

config.add_section('General')
config.add_section('Image')
config.add_section('Directory')
config.add_section('Model')


# General
config['General']['COLAB'] = 'False'

# Image
config['Image']['img_h'] = '299'
config['Image']['img_w'] = '299'
config['Image']['img_size'] = '299'
config['Image']['embeding_image_with'] = 'sim' # 'sim'# 'feat'# 'img'# 
config['Image']['similarity_metric'] = 'cosine'

# Directory
current_dir = os.getcwd()
clf_logs = os.path.join(current_dir, 'clf_logs')
config['Directory']['current_dir'] = current_dir
config['Directory']['clf_logs'] = clf_logs
config['Directory']['metadata_pub_path'] =  'metadata_pub.csv'
config['Directory']['metadata_uog_path'] =  'metadata_uog.csv'
PreTrainedModel_Name = 'InceptionV3'
config['Directory']['pretrainedmodel_name'] = PreTrainedModel_Name
config['Directory']['base_model_dir'] = os.path.join(clf_logs, PreTrainedModel_Name, 'models', 'base-model')
config['Directory']['tune_model_dir'] = os.path.join(clf_logs, PreTrainedModel_Name, 'models', 'tune-model')
config['Directory']['histories_dir'] = os.path.join(clf_logs, PreTrainedModel_Name, 'histories')
config['Directory']['logs_dir'] = os.path.join(clf_logs, PreTrainedModel_Name, 'logs')

config['Directory']['tnse_logs'] = os.path.join(current_dir, 'tnse_logs')
config['Directory']['sprite_image_path'] =  os.path.join(current_dir, 'tnse_logs/sprite.png')
config['Directory']['metadata_path'] =  os.path.join(current_dir, 'tnse_logs/metadata.tsv')


# Model
config['Model']['batch_size'] = '32'
config['Model']['epochs'] = '1'
config['Model']['initial_epochs'] = '20'
config['Model']['fine_tune_epochs'] = '20'
config['Model']['total_epochs'] =' 40'
config['Model']['base_learning_rate'] = '0.0001'
config['Model']['tune_learning_rate'] = '0.00001'
config['Model']['fine_tune_at'] = '265'
config['Model']['use_cv'] = '0'
config['Model']['num_splits'] = '3'

with open('settings.ini', 'w') as configfile:
    config.write(configfile)