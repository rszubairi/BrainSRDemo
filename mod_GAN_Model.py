# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:33:56 2024

@author: amraa
"""

import keras
from keras.models import Model
from keras.layers import Input
import tensorflow as tf
   
def build_sr_gan(generator, discriminator, feature_extractor, lr_image_shape = (160, 160, 1)):
    lr_input = Input(shape=lr_image_shape)  # Low-resolution image input
    
    # Generate high-resolution images #### Output One #####
    generated_hr = generator(lr_input)
    
    ################################### Discriminator #### Output Two #####
    # For the combined model we will only train the generator
    discriminator.trainable = False
    
    # Discriminator determines validity of generated high-resolution images
    validity = discriminator(generated_hr)
    
    
    ################################### Features Extractor #### Output Three #####   
    feature_extractor.trainable = False
    features = feature_extractor(generated_hr)

    ######## Combined model (stacked generator / discriminator / feature extractors)
    sr_gan = Model(inputs=lr_input, outputs=[generated_hr, features, validity])
    return sr_gan

