"""
==============================================================================
============ A binary classifeir for Two-Class imageset with DeepNN ========== 
=========================== by Reza Kakooee ==============================
================================ April 2020 ===============================
==============================================================================
"""

### ======================================================================= ###
### ======================================================================= ###


#%% ======================================================================= ###
###### Import packages
### ======================================================================= ###
import os
import datetime
import logging
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from settings import Params
from loader import Loader

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import metrics
from tensorflow.keras.models import load_model

#%% ======================================================================= ###
###### Necessary functions
### ======================================================================= ###

#### Load the imageset
def get_data(loader, datatype='image'):
    if datatype == 'image':      
        return loader.load_images()  

def class_name_to_index(labels):
    unique_labels = np.unique(labels)
    labels_dic = {}
    for i, label in enumerate(unique_labels):
        labels_dic.update({label:i})
    labels_ind = []
    for label in labels:
        labels_ind.append(labels_dic[label])
    return labels_ind

### ======================================================================= ###
#### Keras image generator
def Image_Generator(trainX, trainY, validX, validY, BATCH_SIZE, logger):
  logger.info("=========== Image Generator ==================================")
  print("----- Image Generator")
  
  train_image_generator = ImageDataGenerator(rescale=1./255)

  valid_image_generator = ImageDataGenerator(rescale=1./255)

  train_image_gen = train_image_generator.flow(trainX,
                                               trainY, 
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)

  valid_image_gen = valid_image_generator.flow(validX,
                                               validY,
                                               batch_size=BATCH_SIZE)  
  return train_image_gen, valid_image_gen


### ======================================================================= ###
#### Create the base model
def create_base_model(IMG_SIZE, base_learning_rate, logger):
  logger.info("=========== Create the base model ============================")
  print("----- Create the base model")
  
  IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
  base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE,
                                                 include_top=False,
                                                 weights='imagenet')

  ##### Freeze the whole base model
  base_model.trainable = False
  # print("----- Number of layers in the base_model: ", len(base_model.layers))
  # print('----- Number of trainable variables in the base_model : ', len(base_model.trainable_variables))

  last_layer = base_model.get_layer('mixed10')
  last_output = last_layer.output
  x = tf.keras.layers.GlobalAveragePooling2D()(last_output)
  x = tf.keras.layers.Dense(1, activation='sigmoid')(x) 
  model = tf.keras.Model(base_model.input, x)
  
  ##### define metrics
  precision = metrics.Precision()
  false_negatives = metrics.FalseNegatives()
  false_positives = metrics.FalsePositives()
  recall = metrics.Recall()
  true_positives = metrics.TruePositives()
  true_negatives = metrics.TrueNegatives()

  ##### compile the model
  if Classes_Metadata().use_cv  == 1:
      model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                    loss='binary_crossentropy',
                    metrics=['accuracy', precision, recall, true_positives, true_negatives, false_negatives, false_positives])
  else:
      model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                    loss='binary_crossentropy',
                    metrics=['accuracy', precision, recall, true_positives, true_negatives, false_negatives, false_positives],
                    weighted_metrics=['accuracy'])


  # print("----- Number of layers in the Base Model: ", len(model.layers))
  # print('----- Number of trainable variables in the Base Model : ', len(model.trainable_variables))

  return base_model, model


### ======================================================================= ###
#### Train the base model
def train_base_model(model, train_image_gen, valid_image_gen, num_train, num_valid, class_weight, BATCH_SIZE, INITIAL_EPOCHS, logs_dir, logger):
  logger.info("=========== Train the base model =============================")
  print("----- Train the base model")
  
  STEPS_PER_EPOCH = num_train // BATCH_SIZE
  VALIDATION_STEPS = num_valid // BATCH_SIZE
  # print('----- STEP_PER_EPOCH: {}, VALIDATION_STEPS: {}'.format(STEPS_PER_EPOCH, VALIDATION_STEPS))

  ##### TB callback
  tensorboard_callback = tf.keras.callbacks.TensorBoard(logs_dir, histogram_freq=1)

  ##### model fitting
  if Classes_Metadata().use_cv == 1:
      history_base = model.fit_generator(train_image_gen,
                                         steps_per_epoch=STEPS_PER_EPOCH,
                                         epochs=INITIAL_EPOCHS,
                                         validation_data=valid_image_gen,
                                         validation_steps=VALIDATION_STEPS,
                                         verbose=2,
                                         callbacks=[tensorboard_callback])
  else:
      history_base = model.fit_generator(train_image_gen,
                                         steps_per_epoch=STEPS_PER_EPOCH,
                                         epochs=INITIAL_EPOCHS,
                                         validation_data=valid_image_gen,
                                         validation_steps=VALIDATION_STEPS,
                                         verbose=2,
                                         callbacks=[tensorboard_callback],
                                         class_weight=class_weight)
              

  return model, history_base


### ======================================================================= ###
#### Create the fine tune model
def create_tune_model(base_model, model, fine_tune_at, tune_learning_rate, logger):
  logger.info("=========== Create the fine tune model =====================")
  print("=========== Create the fine tune model")
  
  ##### Un-freeze the top layers of the model
  base_model.trainable = True
  # print("----- Number of layers in the base_model: ", len(base_model.layers))

  # print("----- Fine tune at: ", fine_tune_at)
  for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False
  # print('----- Number of trainable variables in the base_model : ', len(base_model.trainable_variables))

  ##### define metrics
  recall = metrics.Recall()
  precision = metrics.Precision()
  false_negatives = metrics.FalseNegatives()
  false_positives = metrics.FalsePositives()  
  true_positives = metrics.TruePositives()
  true_negatives = metrics.TrueNegatives()

  ##### compile tune model
  if Classes_Metadata().use_cv == 1:
      model.compile(loss='binary_crossentropy',
                    optimizer = tf.keras.optimizers.RMSprop(lr=tune_learning_rate),
                    metrics=['accuracy', precision, recall, true_positives, true_negatives, false_negatives, false_positives])
  else:
      model.compile(loss='binary_crossentropy',
                    optimizer = tf.keras.optimizers.RMSprop(lr=tune_learning_rate),
                    metrics=['accuracy', precision, recall, true_positives, true_negatives, false_negatives, false_positives],
                    weighted_metrics=['accuracy'])


  # print("----- Number of layers in the Tune Model: ", len(model.layers))
  # print('----- Number of trainable variables in the Tune Model : ', len(model.trainable_variables))
  
  return model


### ======================================================================= ###
#### Train the fine tune model
def train_tune_model(model, train_image_gen, valid_image_gen, num_train, num_valid, class_weight, BATCH_SIZE, INITIAL_EPOCHS, TOTAL_EPOCHS, logs_dir, logger):
  logger.info("=========== Train the fine tune model ========================")
  print("----- Train the fine tune model")

  STEPS_PER_EPOCH = num_train // BATCH_SIZE
  VALIDATION_STEPS = num_valid // BATCH_SIZE
  # print('----- STEP_PER_EPOCH: {}, VALIDATION_STEPS: {}'.format(STEPS_PER_EPOCH, VALIDATION_STEPS))

  ##### TB callback
  tensorboard_callback = tf.keras.callbacks.TensorBoard(logs_dir, histogram_freq=1)

  ##### model fitting
  if Classes_Metadata().use_cv == 1:
      history_tune = model.fit_generator(train_image_gen,
                                         steps_per_epoch=STEPS_PER_EPOCH,
                                         initial_epoch = INITIAL_EPOCHS,
                                         epochs=TOTAL_EPOCHS,
                                         validation_data=valid_image_gen,
                                         validation_steps=VALIDATION_STEPS,
                                         verbose=2,
                                         callbacks=[tensorboard_callback])
  else:
      history_tune = model.fit_generator(train_image_gen,
                                         steps_per_epoch=STEPS_PER_EPOCH,
                                         initial_epoch = INITIAL_EPOCHS,
                                         epochs=TOTAL_EPOCHS,
                                         validation_data=valid_image_gen,
                                         validation_steps=VALIDATION_STEPS,
                                         verbose=2,
                                         callbacks=[tensorboard_callback],
                                         class_weight=class_weight)

  return model, history_tune

#%% ======================================================================= ###
#### Main function
def main(logger):
  logger.info("=========== Main function ====================================")
  print("----- Main function")
  
  COLAB = False
  current_dir = os.getcwd()
  working_dir = current_dir
  params = Params(working_dir, COLAB=COLAB)
  #### =================================================================== ####
  #### =================================================================== ####
  #### pathes for loading
  metadata = Classes_Metadata()
  class0_pathes = metadata.class0_pathes
  class1_pathes = metadata.class1_pathes
  
  #### pathes for saving
  base_model_dir = metadata.base_model_dir
  tune_model_dir = metadata.tune_model_dir
  logs_dir = metadata.logs_dir
  histories_dir = metadata.histories_dir
  
  #### looking into
  num_class0 = len(class0_pathes)
  num_class1 = len(class1_pathes) 
   
  num_imgs = num_class0 + num_class1
  print('=========== num_imgs in class0: {}'.format(num_class0))
  print('=========== num_imgs in class1: {}'.format(num_class1))
  logger.info('=========== num_imgs: {}'.format(num_imgs))
   
  n_negatives = num_class0
  n_positives = num_class1
  # logger.info('=========== n_negatives: {}, n_positives: {}'.format(n_negatives, n_positives))
    
  weight_for_0 = (1 / n_negatives)*num_imgs/2.0 
  weight_for_1 = (1 / n_positives)*num_imgs/2.0
  class_weight = {0: weight_for_0, 1: weight_for_1}
  # print('----- Class Weights: ', class_weight)

  #### =================================================================== ####
  ##### Set constants and hyperparameters
  IMG_SIZE   = Classes_Metadata().IMG_SIZE
  BATCH_SIZE = Classes_Metadata().BATCH_SIZE
    
  INITIAL_EPOCHS   = Classes_Metadata().INITIAL_EPOCHS
#  FINE_TUNE_EPOCHS = Classes_Metadata().FINE_TUNE_EPOCHS
  TOTAL_EPOCHS     = Classes_Metadata().TOTAL_EPOCHS
    
  base_learning_rate = Classes_Metadata().base_learning_rate
  tune_learning_rate = Classes_Metadata().tune_learning_rate
   
  fine_tune_at = Classes_Metadata().fine_tune_at
   
  n_split = Classes_Metadata().n_split
    
  #### =================================================================== ####
  ##### Load images
  images_arr, labels = load_images(class0_pathes, class1_pathes, IMG_SIZE, logger)
  
  #### =================================================================== ####
  ##### Training pipline
  history_base_list = []
  history_tune_list = []
  train_index_list = []
  valid_index_list = []
  i = 0
  if Classes_Metadata().use_cv == 1:
      for train_index, valid_index in StratifiedKFold(n_split, random_state=0).split(images_arr, labels):
         i += 1
         logger.info("=========== Split number: {}".format(i))
         # print("----- Split number: {}".format(i))

         ###### Split images
         trainX, validX = images_arr[train_index], images_arr[valid_index]
         trainY, validY = labels[train_index], labels[valid_index]

         num_train = len(train_index)
         num_valid = len(valid_index)

         train_index_list.append(train_index)
         valid_index_list.append(valid_index)

         # print("----- Num of trains in {}'th fold is {}.".format(i, num_train))
         # print("----- Num of valids in {}'th fold is {}.".format(i, num_valid))
         # print("----- Num of total in {}'th fold is {}.".format(i, num_train+num_valid))

         # print("----- Num of train positives in {}'th fold is {}.".format(i, sum(trainY)))
         # print("----- Num of valid positives in {}'th fold is {}.".format(i, sum(validY)))

         ###### image generator
         train_image_gen, valid_image_gen = Image_Generator(trainX, trainY, validX, validY, BATCH_SIZE, logger)

         ###### base model
         base_model, model = create_base_model(IMG_SIZE, base_learning_rate, logger)
         model, history_base = train_base_model(model, train_image_gen, valid_image_gen, num_train, num_valid, class_weight, BATCH_SIZE, INITIAL_EPOCHS, logs_dir, logger)
         history_base_list.append(history_base)

         ###### save model
         model_name = 'fold' + str(i) + '_' + 'base_model.h5'
         base_model_path = os.path.join(base_model_dir, model_name)
         model.save(base_model_path)

         ## save history tune
         history_base_name = 'fold' + str(i) + '_' + 'history-base.npy'
         history_base_path = os.path.join(histories_dir, history_base_name)
         np.save(history_base_path, history_base.history)

         ###### fine tune model
         #base_model = load_model(base_model_path)
         model = create_tune_model(base_model, model, fine_tune_at, tune_learning_rate, logger)
         model, history_tune = train_tune_model(model, train_image_gen, valid_image_gen, num_train, num_valid, class_weight, BATCH_SIZE, INITIAL_EPOCHS, TOTAL_EPOCHS, logs_dir, logger)
         history_tune_list.append(history_tune)

         ###### save model
         model_name = 'fold' + str(i) + '_' + 'tune_model.h5'
         tune_model_path = os.path.join(tune_model_dir, model_name)
         model.save(tune_model_path)

         ###### save history tune
         history_tune_name = 'fold' + str(i) + '_' + 'history-tune.npy'
         history_tune_path = os.path.join(histories_dir, history_tune_name)
         np.save(history_tune_path, history_tune.history)

         return history_base_list, history_tune_list, train_index_list, valid_index_list

  else:
     trainX, validX, trainY, validY = train_test_split(images_arr, labels, test_size=0.3, stratify=labels)
     ###### Split images
     num_train = len(trainY)
     num_valid = len(validY)

     # print("----- Num of trains is {}.".format(num_train))
     # print("----- Num of valids is {}.".format(num_valid))

     # print("----- Num of train positives is {}.".format(sum(trainY)))
     # print("----- Num of valid positives is {}.".format(sum(validY)))

     ###### image generator
     train_image_gen, valid_image_gen = Image_Generator(trainX, trainY, validX, validY, BATCH_SIZE, logger)

     ###### base model
     base_model, model = create_base_model(IMG_SIZE, base_learning_rate, logger)
     model, history_base = train_base_model(model, train_image_gen, valid_image_gen, num_train, num_valid, class_weight, BATCH_SIZE, INITIAL_EPOCHS, logs_dir, logger)
     history_base_list.append(history_base)

     ###### save model
     model_name = 'base_model.h5'
     base_model_path = os.path.join(base_model_dir, model_name)
     model.save(base_model_path)

     ## save history tune
     history_base_name = 'history-base.npy'
     history_base_path = os.path.join(histories_dir, history_base_name)
     np.save(history_base_path, history_base.history)

     ###### fine tune model
     #base_model = load_model(base_model_path)
     model = create_tune_model(base_model, model, fine_tune_at, tune_learning_rate, logger)
     model, history_tune = train_tune_model(model, train_image_gen, valid_image_gen, num_train, num_valid, class_weight, BATCH_SIZE, INITIAL_EPOCHS, TOTAL_EPOCHS, logs_dir, logger)
     history_tune_list.append(history_tune)

     ###### save model
     model_name = 'tune_model.h5'
     tune_model_path = os.path.join(tune_model_dir, model_name)
     model.save(tune_model_path)

     ###### save history tune
     history_tune_name = 'history-tune.npy'
     history_tune_path = os.path.join(histories_dir, history_tune_name)
     np.save(history_tune_path, history_tune.history)

     return history_base_list, history_tune_list, train_index_list, valid_index_list



#%% ======================================================================= ###
###### Run
### ======================================================================= ###
if __name__ == '__main__':
    #### logging 
    ##### Create and configure the logging
    LOG_FORMAT = "%(message)s"
    logging.basicConfig(filename="logs.log", level=logging.INFO, 
                        format=LOG_FORMAT, filemode="w")
    logger = logging.getLogger()
    logger.info("=========== Run at: {}".format(datetime.datetime.now()))
    start_time = datetime.datetime.now()
    print("----- Run at: {}".format(start_time))
    
    ##### Call the main function
    history_base_list, history_tune_list, train_index_list, valid_index_list = main(logger)
    
    print("----- Total time: {}".format(datetime.datetime.now() - start_time))
