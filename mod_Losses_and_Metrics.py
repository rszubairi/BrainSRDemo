# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:46:11 2024

@author: amraa
"""
import tensorflow as tf
import keras
import tensorflow.keras.backend as K
import numpy as np
from keras.losses import MeanAbsoluteError


def combined_MAE_SSIM_ASL_loss(y_true, y_pred): 
    y_true = tf.clip_by_value(y_true, -1.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, -1.0, 1.0)

    # Upsampling the images to improve res
    y_true = keras.layers.UpSampling2D((4,4))(y_true)
    y_pred = keras.layers.UpSampling2D((4,4))(y_pred)    
        
    # adversarial sharpening loss
    def adversarial_sharpening_loss(y_true, y_pred):
        def laplacian(x):
            laplacian_filter = tf.constant([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=tf.float32)
            laplacian_filter = tf.reshape(laplacian_filter, [3, 3, 1, 1])
            x = tf.expand_dims(x, axis=-1)
            return tf.nn.conv2d(x, laplacian_filter, strides=[1, 1, 1, 1], padding='SAME')
    
        y_true_laplacian = laplacian(y_true)
        y_pred_laplacian = laplacian(y_pred)
    
        return K.mean(K.square(y_true_laplacian - y_pred_laplacian))
    
    # Stracture Similarity loss
    def ssim_loss(y_true, y_pred):       
        # Compute SSIM with normalized images
        ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
        
        # Compute SSIM loss
        return 1 - tf.reduce_mean(ssim)
    def psnr_loss(y_true, y_pred):
        
        psnr = tf.image.psnr(y_true, y_pred, max_val=1.0)
        return 1 - tf.reduce_mean(psnr)
    
    # Calculate the MAE loss
    mae_loss = MeanAbsoluteError()
    mae = mae_loss(y_true, y_pred)
    
    # Calculate the adversarial sharpening loss
    adv_sharpening_loss = adversarial_sharpening_loss(y_true, y_pred)

    # calculate ssim loss
    ssim = ssim_loss(y_true, y_pred)
    
    # calculate PSNR loss
    psnr = psnr_loss(y_true, y_pred) / 100
    
    # Combine the losses
    alpha, beta, delta, theta = 0.5, 0.3, 0.1, 0.1
    combined_loss = alpha * mae + beta * adv_sharpening_loss + ssim * delta + theta * psnr

    return combined_loss

# custom metrics
def calculate_ssim(image1, image2):
    image1 = np.clip(image1, -1.0, 1.0)
    image2 = np.clip(image2, -1.0, 1.0)
    ssim = tf.image.ssim(image1.astype('float32'), image2.astype('float32'), max_val=1.0)
    return ssim

def calculate_psnr(image1, image2):
    image1 = np.clip(image1, -1.0, 1.0)
    image2 = np.clip(image2, -1.0, 1.0)
    psnr = tf.image.psnr(image1.astype('float32'), image2.astype('float32'), max_val=1.0)
    return psnr

