#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from settings import Params
from make_sprite import images_to_sprite
from loader import load_images
from matplots import plotting
from embeder import image_embeding_creator

from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize

import tensorflow as tf

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.contrib.tensorboard.plugins import projector


#%% load model
def get_data(datatype='image'):
    if datatype == 'image':
        return load_images()  

def load_model():
    return InceptionV3(include_top=False, pooling='avg')

# get features
def get_features(model, images):
    return model.predict(images)

def embeding_creator(images):
    image_embeding_creator(images)

def save_metadata(log_dir, metadata_path, labels):
    with open(metadata_path,'w') as f: # "metadata.tsv"
                f.write("Index\tLabel\n")
                for index, label in enumerate(labels):
                    f.write("%d\t%d\n" % (index,label))
                    
def save_model(log_dir):
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(log_dir, "model.ckpt"), 1)

#%% Run
params = Params()
if __name__ == '__main__':
    # directory and file names
    imageset_dir = params.imageset_dir
    log_dir = params.log_dir #'minimalsample'
    sprite_image_path = params.sprite_image_path#"sprit.png"
    metadata_path = params.metadata_path#"metadata.tsv"
    
    # load the whole image set
    num_images, image_pathes, image_names, images_vec, images_list, images_mat, images_arr, labels = get_data()
    IMG_W = params.img_targ_W

    # model
    model = load_model()
    
    # get feature
    features = get_features(model, images_arr)
    
    # similatiris
    if params.similarity_metric == 'cosine':
        similarity_mat = cosine_similarity(features, features)  
    elif params.similarity_metric == 'euclidian':
        distance_mat_euc = euclidean_distances(features, features)
        similarity_mat = 1- normalize(distance_mat_euc)
    
    sprite_image = images_to_sprite(images_arr)
    ### save sprit images
    plt.imsave(params.sprite_image_path , sprite_image, cmap='gray')
    
    # embeding
    embeding_images_with = params.embeding_images_with
    if  embeding_images_with == 'img':
        embd_mat = images_arr
    elif embeding_images_with == 'feat':
        embd_mat = features
    elif embeding_images_with == 'sim':
        embd_mat = similarity_mat 
        
    embeding_creator(embd_mat)
    
    # save metadata
    save_metadata(log_dir, metadata_path, labels)
    
    save_model(log_dir)
    
    # tSNE 
    tsne = TSNE(n_components=3, perplexity=9.0, random_state=0) #, metric='precomputed')
    tsne_val = tsne.fit_transform(1.0001-similarity_mat)
    
    ### MPL
    plotting(tsne_val, images_list)

    #tensorboard --logdir=logs

