# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:18:02 2024

@author: amraa
"""

from keras.models import Model
from keras.layers import Input, BatchNormalization, Conv2D, LeakyReLU, Dense, GlobalMaxPooling2D


def build_discriminator(hr_image_shape=(640,640,1)):        
        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d
                
        df = 64
        # Input img
        d0 = Input(shape=hr_image_shape)

        d1 = d_block(d0, df, bn=False)
        d2 = d_block(d1, df, strides=2)
        d3 = d_block(d2, df*2)
        d4 = d_block(d3, df*2, strides=2)
        d5 = d_block(d4, df*4)
        d6 = d_block(d5, df*4, strides=2)
        d7 = d_block(d6, df*8)
        d8 = d_block(d7, df*8, strides=2)

        d9 = Dense(df*16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        d11 = GlobalMaxPooling2D()(d10)
      
        output1 = Dense(1, activation='sigmoid')(d11)

        output = output1

        return Model(d0, output, name='Discriminator')

