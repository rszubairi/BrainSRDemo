# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:47:42 2024

@author: amraa
"""

import keras
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import gaussian_filter, zoom
import glob

dataset1_ar_path = 'I:\Datasets\High Resolution Single Sequences\Schwannoma High Res\\Processed Arrays\\'
dataset2_ar_path = 'I:\\Datasets\\High Resolution Single Sequences\\UPENN-GBM\\'

## data preprocessing and generator class
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, paths, batch_size=16, input_dims=(320,320, 1), output_dims =(640,640, 1), shuffle=True):
        'Initialization'
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.batch_size = batch_size 
        self.list_IDs = self.get_all_files_in_dirs(paths)
        print('total no. found inside the path dirs is: {} files'.format(len(self.list_IDs )))
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        
        return X, y

    def get_all_files_in_dirs(self, dir_list):
        paths = []
        for path in dir_list:
            paths += glob.glob(os.path.join(path, '**', '*.npy'), recursive=True)
        return paths
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def normalisation(self, image):
        image = image.astype('float32')
        min_val = np.min(image)
        max_val = np.max(image)

    
        # Normalize the pixel values
        epsilon = 1e-7
        image = (image - min_val) / (max_val - min_val + epsilon)
        image = image * 2 - 1
        
        return image

    def add_gaussian_noise(self, image, mean=0, std=0.08):
        # Generate Gaussian noise
        noise = np.random.normal(mean, std, image.shape)
        # Add the noise to the image
        noisy_image = image + noise
        # Clip values to be in the valid range [0, 1]
        noisy_image = np.clip(noisy_image, -1, 1)
        return noisy_image

    def apply_gaussian_blur(self, image):
       # Randomly generate kernel size and sigma
       sigma = np.random.uniform(2.0, 4.0)
       
       # Apply Gaussian blur using scipy's gaussian_filter
       blurred_image = gaussian_filter(image, sigma=sigma)

       return blurred_image
   
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.input_dims))
        y = np.empty((self.batch_size, *self.output_dims))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            ## process truth
            scan_im = self.normalisation(np.load(os.path.join(ID))) 
            
            if len(scan_im.shape) > 2:
                scan_im = scan_im[:,:,0]
                
            ## generate noisy & distorted images
            noisy_image = scan_im.copy()
            
            scan_im = zoom(scan_im, (self.output_dims[0]/scan_im.shape[0], self.output_dims[1]/scan_im.shape[1]))
            
            if np.isnan(scan_im).any() or np.isinf(scan_im).any():
                continue
            else: y[i,...,0] = scan_im


            noisy_image = self.apply_gaussian_blur(noisy_image)
            if np.random.uniform(0.0, 1.0) > 0.5:
                noisy_image = self.add_gaussian_noise(noisy_image)
                
            noisy_image = zoom(noisy_image, (self.input_dims[0]/noisy_image.shape[0], self.input_dims[1]/noisy_image.shape[0]))
            
            if np.isnan(noisy_image).any() or np.isinf(noisy_image).any():
                continue
            else:  X[i,...,0] = noisy_image

        return X, y
    
    
    
'''
### Sanity check for datagenv
sanity_dg = DataGenerator(dataset_ar_path)

def gen_sanity_check(gen):
    for i in range(5):
        Noise_images, Clear_images = gen[i] 
        fig, axes = plt.subplots(2, 6, figsize=(26, 8))
        for i in range(6):
            axes[0, i].imshow(Clear_images[i].squeeze(), cmap='gray')
            axes[0, i].axis('off') 
            axes[0, i].set_title('Normal') 
            axes[1, i].imshow(Noise_images[i].squeeze(), cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title('Noisy') 
                            
        plt.show()   
                   
gen_sanity_check(sanity_dg)'''