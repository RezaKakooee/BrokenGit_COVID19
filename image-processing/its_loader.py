# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 03:01:32 2020

@author: rkako
"""
import os
import pandas as pd
import numpy as np
from skimage.io import imread, imsave

current_dir = os.getcwd()
imageset_dir = os.path.join(current_dir, 'covid-chestxray-dataset/images')
csvpath = os.path.join(current_dir, "covid-chestxray-dataset", "metadata.csv")
views=["PA"]
transform=None
data_aug=None
nrows=None
seed=0
pure_labels=False
unique_patients=True

np.random.seed(seed)  # Reset the seed so all runs are the same.

# defined here to make the code easier to read
pneumonias = ["COVID-19", "SARS", "MERS", "ARDS", "Streptococcus", "Pneumocystis", "Klebsiella"]

pathologies = ["Pneumonia","Viral Pneumonia", "Bacterial Pneumonia", "Fungal Pneumonia", "No Finding"] + pneumonias
pathologies = sorted(pathologies)

mapping = dict()
mapping["Pneumonia"] = pneumonias
mapping["Viral Pneumonia"] = ["COVID-19", "SARS", "MERS"]
mapping["Bacterial Pneumonia"] = ["Streptococcus", "Klebsiella"]
mapping["Fungal Pneumonia"] = ["Pneumocystis"]

# Load data
csvpath = csvpath
csv = pd.read_csv(csvpath, nrows=nrows)
MAXVAL = 255  # Range [0 255]

# Keep only the frontal views.
#idx_pa = csv["view"].isin(["PA", "AP", "AP Supine"])
idx_pa = csv["view"].isin(views)
csv = csv[idx_pa]

labels = []
for pathology in pathologies:
    pass
    mask = csv["finding"].str.contains(pathology)
    if pathology in mapping:
        for syn in mapping[pathology]:
            #print("mapping", syn)
            mask |= csv["finding"].str.contains(syn)
    labels.append(mask.values)
labels = np.asarray(labels).T
labels = labels.astype(np.float32)


def normalize(sample, maxval):
    """Scales images to be roughly [-1024 1024]."""
    sample = (2 * (sample.astype(np.float32) / maxval) - 1.) * 1024
    #sample = sample / np.std(sample)
    return sample

idx = 0
imgid = csv['filename'].iloc[idx]
img_path = os.path.join(imageset_dir, imgid)
print(img_path)
img = imread(img_path)
img = normalize(img, MAXVAL)  

# Check that images are 2D arrays
if len(img.shape) > 2:
    img = img[:, :, 0]
if len(img.shape) < 2:
    print("error, dimension lower than 2 for image")

# Add color channel
img = img[None, :, :]                    
                       
    
{"PA":img, "lab":labels[idx], "idx":idx}