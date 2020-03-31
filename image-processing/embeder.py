# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:38:13 2020

@author: reza
"""
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from settings import Params

class Embeder:
  def __init__(self, working_dir, COLAB=False):
    self.params = Params(working_dir, COLAB=COLAB)    

  def image_embeding_creator(self, images):
      log_dir = self.params.log_dir
      sprite_image_path = self.params.sprite_image_path
      metadata_path = self.params.metadata_path
      image_height = self.params.img_targ_H
      image_width = self.params.img_targ_W

      embedding_var = tf.Variable(images, name="image_embedding")
      summary_writer = tf.summary.FileWriter(log_dir)
      config = projector.ProjectorConfig()
      embedding = config.embeddings.add()
      embedding.tensor_name = embedding_var.name
      embedding.metadata_path = metadata_path
      embedding.sprite.image_path = sprite_image_path
      embedding.sprite.single_image_dim.extend([image_height, image_width])
      projector.visualize_embeddings(summary_writer, config)
