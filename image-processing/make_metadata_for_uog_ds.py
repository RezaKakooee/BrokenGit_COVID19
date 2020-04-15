# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:06:52 2020

@author: rkako
"""
#%%
import os
import numpy as np
import pandas as pd
import pydicom
from tensorflow.keras.preprocessing.image import save_img

#%%
metadata_uog = pd.read_csv('UoGeneva/metadata_base.csv')
metadata_uog.insert(6, 'path_postfix', 0)

current_dir = os.getcwd()
path_prefix_covid_dcm = 'UoGeneva/images/Covid19-dcm'
path_prefix_covid = 'UoGeneva/images/Covid19'
path_prefix_norm = 'UoGeneva/images/Norm'
covid_dir = os.path.join(current_dir, path_prefix_covid_dcm)
norm_dir = os.path.join(current_dir, path_prefix_norm)

covid_images_names_dcm = os.listdir(path_prefix_covid_dcm)
covid_images_names = [dcm_name[:-4] + '.jpeg' for dcm_name in covid_images_names_dcm]
norm_images_names = os.listdir(norm_dir)
filename = covid_images_names + norm_images_names

num_covids = len(covid_images_names_dcm)
num_norms = len(norm_images_names)
finding = ['COVID-19'] * num_covids + ['Norm'] * num_norms
survival = [np.nan] * num_covids + ['Y'] * num_norms

label = ['COVID-19_O'] * num_covids + ['Norm_Y'] * num_norms


covid_paths_postfix_dcm = [path_prefix_covid_dcm + '/' + img_name for img_name in covid_images_names_dcm]
covid_paths_postfix = [path_prefix_covid + '/' + img_name for img_name in covid_images_names]
norm_paths_postfix = [path_prefix_norm + '/' + img_name for img_name in norm_images_names]
path_postfix = covid_paths_postfix + norm_paths_postfix
path_postfix_dcm_jpg = covid_paths_postfix_dcm + norm_paths_postfix
        

num_imgs = num_covids + num_norms
date = 2020 * np.ones((num_imgs,1))


#%%
for dcm_path, img_name in zip(covid_paths_postfix_dcm, covid_images_names):
    dcm = pydicom.dcmread(dcm_path)
    img_arr = dcm.pixel_array
    img_arr = img_arr[:, :, None] 
    # img = array_to_img(arr)
    img_path = os.path.join(path_prefix_covid, img_name)
    save_img(img_path, img_arr)
    
#%% Prepare datafeae and save to csv
df = pd.DataFrame(np.nan, index=np.arange(0,num_imgs), columns=metadata_uog.columns)

df['finding'] = finding
df['survival'] = survival
df['date'] = date
df['path_postfix'] = path_postfix
df['filename'] = filename
df['label'] = label

df.to_csv('metadata_uog.csv', index=False)

#%%

