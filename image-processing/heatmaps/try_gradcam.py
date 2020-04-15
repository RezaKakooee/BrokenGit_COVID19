# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:46:50 2020

@author: reza
"""
import cv2
import random
import numpy as np
import random
import tensorflow as tf
from setting import Settings
from gradcam import GradCAM
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import  img_to_array, load_img

#%%
def load_from_saved_models(saved_model_path):
    model = load_model(saved_model_path)
    return model

def get_and_predic_one_image(image_pathes_class0, image_pathes_class1, class_name, prediction_name):
    if class_name == 'unselfie':
        classimagepathes = image_pathes_class0
    else:
        classimagepathes = image_pathes_class1
    
    random.shuffle(classimagepathes)
    is_succeed = False
    for imagepath in classimagepathes:
        orig = cv2.imread(imagepath)
        # resized = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
        
        image = load_img(imagepath, target_size=(IMG_SIZE, IMG_SIZE))
        image = img_to_array(image)
        image = image/255.0
        image = np.expand_dims(image, axis=0)
        
        prediction = model.predict(image)
        if prediction_name == 'selfie':
            if prediction[0][0] >= 0.5:
                is_succeed = True
                break
        else:
            if prediction[0][0] < 0.5:
                is_succeed = True
                break
        
    if is_succeed:
        return image, orig, prediction
    else:
         raise ValueError("ClassName-PredictionName pair does not wrok.")
         
def get_and_predic_several_images(num_rand_samps, image_pathes_class0, image_pathes_class1):
    IMAGES = []
    ORIGS = [] 
    PREDS = []
    LABELS = []
    for class_image_pathes in [image_pathes_class0, image_pathes_class1]:
        random.shuffle(class_image_pathes)
        class_image_pathes_half = class_image_pathes[:int(num_rand_samps)/2]
        for imagepath in class_image_pathes_half:
            label = 'unselfie' if 'unselfie' in imagepath else 'selfie'
            orig = cv2.imread(imagepath)
            # resized = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
            
            image = load_img(imagepath, target_size=(IMG_SIZE, IMG_SIZE))
            image = img_to_array(image)
            image = image/255.0
            image = np.expand_dims(image, axis=0)
            
            prediction = model.predict(image)
            
            IMAGES.append(image)
            ORIGS.append(orig)
            PREDS.append(prediction)
            LABELS.append(label)
            
    return IMAGES, ORIGS, PREDS, LABELS
                  

def load_one_image(imagepath):
    orig = cv2.imread(imagepath)
    
    image = load_img(imagepath, target_size=(IMG_SIZE, IMG_SIZE))
    image = img_to_array(image)
    image = image/255.0
    image = np.expand_dims(image, axis=0)
    
    return image, orig

def decode_prediction(prediction):
    if prediction >= 0.5:
        return 1, 'selfie'
    else:
        return 0, 'unselfie'
         
def make_prediction(model, image):
    prediction = model.predict(image)
    pred_class_ind = int(tf.round(prediction).numpy()[0])
    pred_class_name = ['unselfie' if pred_class_ind == 0 else 'selfie']
    return prediction, pred_class_ind, pred_class_name[0]

def get_heatmap(model, image, orig, pred_class_ind):
    cam = GradCAM(model, class_idx=pred_class_ind)
    heatmap = cam.compute_heatmap(image)
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    return heatmap

def get_overlaid(orig, heatmap, alpha=0.5):
    colormap=cv2.COLORMAP_JET#COLORMAP_VIRIDIS
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0])) # We resize the heatmap to have the same size as the original image
    heatmap = cv2.applyColorMap(heatmap, colormap)
    overlaid = cv2.addWeighted(orig, alpha, heatmap, 1-alpha, 0) # heatmap * 0.55 + orig
    
    return (heatmap, overlaid)

def show_heatmap(i, class_ind, pred_class_ind, class_name, pred_class_name, orig, heatmap, overlaid):
    # cv2.rectangle(overlaid, (0, 0), (340, 40), (0, 0, 0), -1)
    
    stacked = np.hstack([orig, heatmap, overlaid])
    cv2.imshow("Output", stacked)
    cv2.putText(orig, class_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(overlaid, pred_class_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite('./output_images/Im{:02d}_Target_{}, Prediction_{}.jpg'.format(i+1, class_name, pred_class_name), stacked)
    cv2.destroyAllWindows()
    # cv2.waitKey(0)

def plot_heatmap(i, class_ind, pred_class_ind, class_name, pred_class_name, orig, heatmap, overlaid):
    plt.figure(figsize=(30, 10))
    stacked = np.hstack([orig, heatmap, overlaid])
#      stacked = imutils.resize(stacked, height=700)
    plt.imshow(stacked)
    TITLE = "Target: {}, Prediction: {}".format(class_name, pred_class_name)
    plt.title(TITLE.title())
    plt.axis('off')
    plt.text(stacked.shape[0]-10, stacked.shape[1]/2,   './output_images/Im{:02d}_Target_{}, Prediction_{}.jpg'.format(i+1, class_name, pred_class_name))
    
    
    plt.show()
    plt.imsave('./output_images/Im{:02d}_Target_{}, Prediction_{}.jpg'.format(i+1, class_name, pred_class_name), stacked)

def plot_heatmaps(class_names, pred_class_names, origs, heatmaps, overlaids):
    plt.figure(figsize=(30, 10))
    subplot_dim = np.sqrt(16)  
    for n in range(16):
      ax = plt.subplot(subplot_dim, subplot_dim, n+1)
      stacked = np.hstack([origs[n], heatmaps[n], overlaids[n]])
#      stacked = imutils.resize(stacked, height=700)
      plt.imshow(stacked)
#      plt.imshow(overlaids[n])
      TITLE = "Target: {}, Prediction: {}".format(class_names[n], pred_class_names[n])
      plt.title(TITLE.title())
      plt.axis('off')

def save_arrays(output_dir, labels, predicts, origs, heatmaps, overlaids):
    np.save(output_dir + '/original_images', origs)
    np.save(output_dir + '/heatmap_images', heatmaps)
    np.save(output_dir + '/overlaid_images', overlaids)
    np.save(output_dir + '/labels', labels)
    np.save(output_dir + '/predictions', predicts)

#%%
if __name__ == '__main__':
    settings = Settings()
    saved_model_path = settings.saved_model_path
    IMG_SIZE = settings.IMG_SIZE
    image_pathes_class0 = settings.image_pathes_class0
    image_pathes_class1 = settings.image_pathes_class1
    
    ### load model
    model = load_from_saved_models(saved_model_path)
    
    ### load image
    ONE = False
    if ONE:
        class_names = ['unselfie', 'selfie']
        prediction_names = ['unselfie', 'selfie']
        class_pred_list = [(a, b) for a in class_names for b in prediction_names]
        i = 0
        for cp in class_pred_list:
            image, orig, prediction = get_and_predic_one_image(image_pathes_class0, image_pathes_class1, cp[0], cp[1])
            pred_class_ind, pred_class_name = decode_prediction(prediction)
            heatmap = get_heatmap(model, image, orig, pred_class_ind)
            heatmap, overlaid = get_overlaid(orig, heatmap, alpha=0.5)
            class_ind = 0 if cp[0]=='unselfie' else 1
            show_heatmap(i, class_ind, pred_class_ind, cp[0], cp[1], orig, heatmap, overlaid)
            i += 1
        
    SEVERAL = False
    if SEVERAL:
        num_rand_samps = 50
        IMAGES, ORIGS, PREDS, LABELS = get_and_predic_several_images(num_rand_samps, image_pathes_class0, image_pathes_class1)
        i = 0
        for image, orig, prediction, label in zip(IMAGES, ORIGS, PREDS, LABELS):
            pred_class_ind, pred_class_name = decode_prediction(prediction)
            heatmap = get_heatmap(model, image, orig, pred_class_ind)
            heatmap, overlaid = get_overlaid(orig, heatmap, alpha=0.5)
            class_name = label
            class_ind = 0 if class_name == 'unselfie' else 'selfie'
            show_heatmap(i, class_ind, pred_class_ind, class_name, pred_class_name, orig, heatmap, overlaid)
            # plot_heatmap(i, class_ind, pred_class_ind, class_name, pred_class_name, orig, heatmap, overlaid)
            i += 1

#%% ======================================================================= ###
### ===================== matplotlib for several images =================== ###
    MPL = False
    if MPL:
        labels = []
        class_names_00 = []
        class_names_01 = []
        class_names_10 = []
        class_names_11 = []
        pred_class_names_00 = []
        pred_class_names_01 = []
        pred_class_names_10 = []
        pred_class_names_11 = []
        orig_00 = []
        orig_01 = []
        orig_10 = []
        orig_11 = []
        imagepathes_00 = []
        imagepathes_01 = []
        imagepathes_10 = []
        imagepathes_11 = []
        heatmap_00 = []
        heatmap_01 = []
        heatmap_10 = []
        heatmap_11 = []
        overlaid_00 = []
        overlaid_01 = []
        overlaid_10 = []
        overlaid_11 = []
        image_pathes = image_pathes_class0 + image_pathes_class1
        random.shuffle(image_pathes)
        for imagepath in image_pathes: 
            # load image
            label = 0 if "unselfie" in imagepath else 1
            labels.append(label)
            image, orig = load_one_image(imagepath)
            # model prediction
            prediction, pred_class_ind, pred_class_name = make_prediction(model, image)
            ### make heatmap
            heatmap  = get_heatmap(model, image, orig, pred_class_ind)
            heatmap, overlaid = get_overlaid(orig, heatmap, alpha=0.5)
            if label == 0 and pred_class_ind == 0:
                if len(overlaid_00) < 4:
                    class_names_00.append('unselfie')
                    pred_class_names_00.append(pred_class_name)
                    overlaid_00.append(overlaid)
                    heatmap_00.append(heatmap)
                    imagepathes_00.append(imagepath)
                    orig_00.append(orig)
            elif label == 0 and pred_class_ind == 1:
                class_names_01.append('unselfie')
                pred_class_names_01.append(pred_class_name)
                if len(overlaid_01) < 4:
                    overlaid_01.append(overlaid)
                    heatmap_01.append(heatmap)
                    imagepathes_01.append(imagepath)
                    orig_01.append(orig)
            elif label == 1 and pred_class_ind == 0:
                class_names_10.append('selfie')
                pred_class_names_10.append(pred_class_name)
                if len(overlaid_10) < 4:
                    overlaid_10.append(overlaid)
                    heatmap_10.append(heatmap)
                    imagepathes_10.append(imagepath)
                    orig_10.append(orig)
            elif label == 1 and pred_class_ind == 1:
                class_names_11.append('selfie')
                pred_class_names_11.append(pred_class_name)
                if len(overlaid_11) < 4:
                    overlaid_11.append(overlaid)
                    heatmap_11.append(heatmap)
                    imagepathes_11.append(imagepath)
                    orig_11.append(orig)
            
            len_arr = np.array([len(overlaid_00), len(overlaid_01), len(overlaid_10), len(overlaid_11)])
            if np.all(len_arr == 4):
                break
            
        # prepaer heatmaps for ploting
        if len(overlaid_00) < 4:
            class_names_00.append(class_names_00[2])
            pred_class_names_00.append(pred_class_names_00[2])
            overlaid_00.append(overlaid_00[2])
            heatmap_00.append(heatmap_00[2])
            imagepathes_00.append(imagepathes_00[2])
            orig_00.append(orig_00[2])
            
        if len(overlaid_01) < 4: 
            class_names_01.append(class_names_01[2])
            pred_class_names_01.append(pred_class_names_01[2])
            overlaid_01.append(overlaid_01[2])
            heatmap_01.append(heatmap_01[2])                
            imagepathes_01.append(imagepathes_01[2])
            orig_01.append(orig_01[2])
            
        if len(overlaid_10) < 4: 
            class_names_10.append(class_names_10[2])
            pred_class_names_10.append(pred_class_names_10[2])
            overlaid_10.append(overlaid_10[2])
            heatmap_10.append(heatmap_10[2])
            imagepathes_10.append(imagepathes_10[2])
            orig_10.append(orig_10[2])
            
        if len(overlaid_11) < 4: 
            class_names_11.append(class_names_11[2])
            pred_class_names_11.append(pred_class_names_11[2])
            overlaid_11.append(overlaid_11[2])
            heatmap_11.append(heatmap_11[2])
            imagepathes_11.append(imagepathes_11[2])
            orig_11.append(orig_11[2])
        
        pred_class_names = pred_class_names_00 + pred_class_names_01 + pred_class_names_10 + pred_class_names_11
        class_names = class_names_00 + class_names_01 + class_names_10 + class_names_11
        overlaids = overlaid_00 + overlaid_01 + overlaid_10 + overlaid_11
        heatmaps = heatmap_00 + heatmap_01 + heatmap_10 + heatmap_11
        # imagepathes = imagepathes_00 + imagepathes_01 + imagepathes_10 + imagepathes_11
        origs = orig_00 + orig_01 + orig_10 + orig_11
        plot_heatmaps(class_names, pred_class_names, origs, heatmaps, overlaids)
    
    