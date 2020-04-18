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
import configparser 
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

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
#### Keras image generator
def Image_Generator(trainX, trainY, validX, validY, logger):
  logger.info("=========== Image Generator ==================================")
  BATCH_SIZE = int(config['Model']['BATCH_SIZE'])
  
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
def create_base_model(logger, num_classes):
  logger.info("=========== Create the base model ============================")
  IMG_SHAPE = (int(config['Image']['img_size']), int(config['Image']['img_size']), 3)
  USE_CV = int(config['Model']['use_cv'])
  base_learning_rate = float(config['Model']['base_learning_rate'])
  
  print("----- Create the base model")
  base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE,
                                                 include_top=False,
                                                 weights='imagenet')

  ##### Freeze the whole base model
  base_model.trainable = False
  # print("----- Number of layers in the base_model: ", len(base_model.layers))
  # print('----- Number of trainable variables in the base_model : ', len(base_model.trainable_variables))

  if num_classes == 2:
    n_units_in_last_layer = 1
    ACTIVATION = 'sigmoid'
    LOSS = 'binary_crossentropy'
  else:
    n_units_in_last_layer = num_classes
    ACTIVATION = 'softmax'
    LOSS = 'sparse_categorical_crossentropy'
      
  last_layer = base_model.get_layer('mixed10')
  last_output = last_layer.output
  x = tf.keras.layers.GlobalAveragePooling2D()(last_output)
  x = tf.keras.layers.Dense(n_units_in_last_layer, activation=ACTIVATION)(x) 
  model = tf.keras.Model(base_model.input, x)
  
  ##### define metrics
  precision = metrics.Precision()
  false_negatives = metrics.FalseNegatives()
  false_positives = metrics.FalsePositives()
  recall = metrics.Recall()
  true_positives = metrics.TruePositives()
  true_negatives = metrics.TrueNegatives()

  ##### compile the model
  if USE_CV  == 1:
      model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                    loss=LOSS,
                    metrics=['accuracy', precision, recall, true_positives, true_negatives, false_negatives, false_positives])
  else:
      model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                    loss=LOSS,
                    metrics=['accuracy'],
                    weighted_metrics=['accuracy'])


  # print("----- Number of layers in the Base Model: ", len(model.layers))
  # print('----- Number of trainable variables in the Base Model : ', len(model.trainable_variables))

  return base_model, model


### ======================================================================= ###
#### Train the base model
def train_base_model(model, train_image_gen, valid_image_gen, num_train, num_valid, class_weight, logger):
  logger.info("=========== Train the base model =============================")
  BATCH_SIZE = int(config['Model']['batch_size'])
  INITIAL_EPOCHS = int(config['Model']['initial_epochs'])
  logs_dir = config['Directory']['logs_dir']
  USE_CV = int(config['Model']['use_cv'])
  
  print("----- Train the base model")
  STEPS_PER_EPOCH = num_train // BATCH_SIZE
  VALIDATION_STEPS = num_valid // BATCH_SIZE
  # print('----- STEP_PER_EPOCH: {}, VALIDATION_STEPS: {}'.format(STEPS_PER_EPOCH, VALIDATION_STEPS))

  ##### TB callback
  tensorboard_callback = tf.keras.callbacks.TensorBoard(logs_dir, histogram_freq=1)

  ##### model fitting
  if USE_CV == 1:
      history_base = model.fit_generator(train_image_gen,
                                         steps_per_epoch=STEPS_PER_EPOCH,
                                         epochs=INITIAL_EPOCHS,
                                         validation_data=valid_image_gen,
                                         validation_steps=VALIDATION_STEPS,
                                         verbose=1,
                                         callbacks=[tensorboard_callback])
  else:
      history_base = model.fit_generator(train_image_gen,
                                         steps_per_epoch=STEPS_PER_EPOCH,
                                         epochs=INITIAL_EPOCHS,
                                         validation_data=valid_image_gen,
                                         validation_steps=VALIDATION_STEPS,
                                         verbose=1,
                                         callbacks=[tensorboard_callback],
                                         class_weight=class_weight)
              

  return model, history_base


### ======================================================================= ###
#### Create the fine tune model
def create_tune_model(base_model, model, num_classes, logger):
  logger.info("=========== Create the fine tune model =====================")
  fine_tune_at = int(config['Model']['fine_tune_at'])
  USE_CV = int(config['Model']['use_cv'])
  tune_learning_rate = float(config['Model']['tune_learning_rate'])
  
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

  if num_classes == 2:
    LOSS = 'binary_crossentropy'
  else:
    LOSS = 'sparse_categorical_crossentropy'
  ##### compile tune model
  if USE_CV == 1:
      model.compile(loss=LOSS,
                    optimizer = tf.keras.optimizers.RMSprop(lr=tune_learning_rate),
                    metrics=['accuracy', precision, recall, true_positives, true_negatives, false_negatives, false_positives])
  else:
      model.compile(loss=LOSS,
                    optimizer = tf.keras.optimizers.RMSprop(lr=tune_learning_rate),
                    metrics=['accuracy'],
                    weighted_metrics=['accuracy'])


  # print("----- Number of layers in the Tune Model: ", len(model.layers))
  # print('----- Number of trainable variables in the Tune Model : ', len(model.trainable_variables))
  
  return model


### ======================================================================= ###
#### Train the fine tune model
def train_tune_model(model, train_image_gen, valid_image_gen, num_train, num_valid, class_weight, logger):
  logger.info("=========== Train the fine tune model ========================")
  BATCH_SIZE = int(config['Model']['batch_size'])
  INITIAL_EPOCHS = int(config['Model']['initial_epochs'])
  TOTAL_EPOCHS = int(config['Model']['total_epochs'])
  logs_dir = config['Directory']['logs_dir']
  USE_CV = int(config['Model']['use_cv'])
  
  print("----- Train the fine tune model")
  STEPS_PER_EPOCH = num_train // BATCH_SIZE
  VALIDATION_STEPS = num_valid // BATCH_SIZE
  # print('----- STEP_PER_EPOCH: {}, VALIDATION_STEPS: {}'.format(STEPS_PER_EPOCH, VALIDATION_STEPS))

  ##### TB callback
  tensorboard_callback = tf.keras.callbacks.TensorBoard(logs_dir, histogram_freq=1)

  ##### model fitting
  if USE_CV == 1:
      history_tune = model.fit_generator(train_image_gen,
                                         steps_per_epoch=STEPS_PER_EPOCH,
                                         initial_epoch = INITIAL_EPOCHS,
                                         epochs=TOTAL_EPOCHS,
                                         validation_data=valid_image_gen,
                                         validation_steps=VALIDATION_STEPS,
                                         verbose=1,
                                         callbacks=[tensorboard_callback])
  else:
      history_tune = model.fit_generator(train_image_gen,
                                         steps_per_epoch=STEPS_PER_EPOCH,
                                         initial_epoch = INITIAL_EPOCHS,
                                         epochs=TOTAL_EPOCHS,
                                         validation_data=valid_image_gen,
                                         validation_steps=VALIDATION_STEPS,
                                         verbose=1,
                                         callbacks=[tensorboard_callback],
                                         class_weight=class_weight)

  return model, history_tune

#%% ======================================================================= ###
#### Main function
def main(logger, config):
  logger.info("=========== Main function ====================================")
  print("----- Main function")
  NUM_SPLITS = int(config['Model']['num_splits'])
  USE_CV = int(config['Model']['use_cv'])
  base_model_dir = config['Directory']['base_model_dir'] 
  tune_model_dir = config['Directory']['tune_model_dir'] 
  histories_dir = config['Directory']['histories_dir'] 
  #### =================================================================== ####
  ##### Load images
  loader = Loader()
  num_images, images_arr, labels, labels_ind, num_classes, class_names, class_weight = loader.load_images()
  
  #### =================================================================== ####
  ##### Training pipline
  history_base_list = []
  history_tune_list = []
  train_index_list = []
  valid_index_list = []
  i = 0
  if USE_CV  == 1:
      for train_index, valid_index in StratifiedKFold(NUM_SPLITS, random_state=0).split(images_arr, labels_ind):
         i += 1
         logger.info("=========== Split number: {}".format(i))
         # print("----- Split number: {}".format(i))

         ###### Split images
         trainX, validX = images_arr[train_index], images_arr[valid_index]
         trainY, validY = labels_ind[train_index], labels_ind[valid_index]

         num_train = len(train_index)
         num_valid = len(valid_index)

         train_index_list.append(train_index)
         valid_index_list.append(valid_index)

         ###### image generator
         train_image_gen, valid_image_gen = Image_Generator(trainX, trainY, validX, validY, logger)

         ###### base model
         base_model, model = create_base_model(logger, class_names)
         model, history_base = train_base_model(model, train_image_gen, valid_image_gen, num_train, num_valid, class_weight, logger)
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
         
         model = create_tune_model(base_model, model, num_classes, logger)
         model, history_tune = train_tune_model(model, train_image_gen, valid_image_gen, num_train, num_valid, class_weight, logger)
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
     trainX, validX, trainY, validY = train_test_split(images_arr, labels_ind, test_size=0.3, stratify=labels_ind)
     ###### Split images
     num_train = len(trainY)
     num_valid = len(validY)

     # print("----- Num of trains is {}.".format(num_train))
     # print("----- Num of valids is {}.".format(num_valid))

     # print("----- Num of train positives is {}.".format(sum(trainY)))
     # print("----- Num of valid positives is {}.".format(sum(validY)))

     ###### image generator
     train_image_gen, valid_image_gen = Image_Generator(trainX, trainY, validX, validY, logger)

     ###### base model
     base_model, model = create_base_model(logger, num_classes)
     model, history_base = train_base_model(model, train_image_gen, valid_image_gen, num_train, num_valid, class_weight, logger)
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
     model = create_tune_model(base_model, model, num_classes, logger)
     model, history_tune = train_tune_model(model, train_image_gen, valid_image_gen, num_train, num_valid, class_weight, logger)
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
    
    #### config 
    config = configparser.ConfigParser()
    config.read('settings.ini')
  
    logger.info("=========== Run at: {}".format(datetime.datetime.now()))
    start_time = datetime.datetime.now()
    print("----- Run at: {}".format(start_time))
    
    ##### Call the main function
    history_base_list, history_tune_list, train_index_list, valid_index_list = main(logger, config)
    
    print("----- Total time: {}".format(datetime.datetime.now() - start_time))
