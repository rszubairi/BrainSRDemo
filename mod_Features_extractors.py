# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:18:02 2024

@author: amraa
"""

from keras.models import Model
from keras import applications
from keras.layers import Input, Lambda
import tensorflow as tf

def vgg19_feature_extractor(hr_image_shape=(640, 640, 1)):
    # Define the input tensor
    input_ = Input(hr_image_shape)
    
    # Apply grayscale to RGB transformation as a Lambda layer
    rgb_input = Lambda(lambda x: tf.image.grayscale_to_rgb(x))(input_)
    
    # Load the pre-trained VGG19 model with weights trained on ImageNet
    vgg = applications.VGG19(include_top=False, weights='imagenet', input_tensor=rgb_input)
    
    # Create a model that outputs these intermediate layers
    feature_extractor = Model(inputs=input_, outputs=vgg.output, name='Feature_extractor')
    
    return feature_extractor

def vgg16_feature_extractor(hr_image_shape=(640, 640, 1)):
    # Define the input tensor
    input_ = Input(hr_image_shape)
    
    # Apply grayscale to RGB transformation as a Lambda layer
    rgb_input = Lambda(lambda x: tf.image.grayscale_to_rgb(x))(input_)
    
    # Load the pre-trained VGG19 model with weights trained on ImageNet
    vgg = applications.VGG16(include_top=False, weights='imagenet', input_tensor=rgb_input)
    
    # Create a model that outputs these intermediate layers
    feature_extractor = Model(inputs=input_, outputs=vgg.output, name='Feature_extractor')
    
    return feature_extractor


def efficientnet_feature_extractor(hr_image_shape=(640, 640, 1)):
    # Define the input tensor
    input_ = Input(hr_image_shape)
    
    # Apply grayscale to RGB transformation as a Lambda layer
    rgb_input = Lambda(lambda x: tf.image.grayscale_to_rgb(x))(input_)
    
    # Load the pre-trained VGG19 model with weights trained on ImageNet
    EfficientNet = applications.EfficientNetB4(include_top=False, weights='imagenet', input_tensor=rgb_input)
    

    # Create a model that outputs these intermediate layers
    feature_extractor = Model(inputs=input_, outputs=EfficientNet.output, name='Feature_extractor')
    
    return feature_extractor


def inception_v3_feature_extractor(hr_image_shape=(640, 640, 1)):
    # Define the input tensor
    input_ = Input(hr_image_shape)
    
    # Apply grayscale to RGB transformation as a Lambda layer
    rgb_input = Lambda(lambda x: tf.image.grayscale_to_rgb(x))(input_)
    
    # Load the pre-trained VGG19 model with weights trained on ImageNet
    inception_v3 = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=rgb_input)
    

    # Create a model that outputs these intermediate layers
    feature_extractor = Model(inputs=input_, outputs=inception_v3.output, name='Feature_extractor')
    
    return feature_extractor


