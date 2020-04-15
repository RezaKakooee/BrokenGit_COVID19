# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:46:50 2020

@author: reza
"""
#%%
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

#%%
class GradCAM:
    def __init__(self, model, class_idx, is_pretrain=True, layer_name='mixed10'):
        self.model = model
        self.class_idx = class_idx
        self.layer_name = layer_name
        self.is_pretrain = is_pretrain

        if self.layer_name == None:
            self.layer_name = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                print("Target layer is: ", layer.name)
                return layer.name
        raise ValueError("Could not find suitable conv layer.")

    def compute_heatmap(self, image, eps=1e-8):
        grad_model = Model(inputs=[self.model.inputs],
                           outputs=[self.model.get_layer(self.layer_name).output,
                                    self.model.output])
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (conv_outputs, predictions) = grad_model(inputs)
            if self.is_pretrain:
                loss = predictions[0] if self.class_idx == 1 else 1-predictions[0]
            else:
                loss = predictions[:, self.class_idx]

        grads = tape.gradient(loss, conv_outputs)
        cast_conv_outputs = tf.cast(conv_outputs>0, "float32")
        cast_grads = tf.cast(grads>0, "float32")
        guided_grads = cast_conv_outputs * cast_grads * grads

        conv_outputs = conv_outputs[0]
        guided_grads = guided_grads[0]

        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = tf.reduce_mean(tf.multiply(weights, conv_outputs), axis=-1)

        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        numerator = heatmap - np.min(heatmap)
        denumerator = (heatmap.max() - heatmap.min()) + eps
        heatmap = numerator / denumerator
        heatmap = (heatmap * 255).astype("uint8")

        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=1, colormap=cv2.COLORMAP_VIRIDIS):
        heatmap = cv2.applyColorMap(heatmap, colormap)
        overlaid = cv2.addWeighted(image, alpha, heatmap, 1-alpha, 0)

        return  (heatmap, overlaid)


