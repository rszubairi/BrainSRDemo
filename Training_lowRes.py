# -*- coding: utf-8 -*-
"""
Created on Wed May 29 19:03:56 2024

@author: amraa
"""

import numpy as np
from mod_SRGAN_DeepRes import build_generator
from mod_GAN_Model import build_sr_gan
from mod_Discriminator import build_discriminator
from mod_Features_extractors import vgg19_feature_extractor
from mod_Losses_and_Metrics import combined_MAE_SSIM_ASL_loss, calculate_ssim, calculate_psnr
from DataGeneator import DataGenerator
import matplotlib.pyplot as plt
import gc
import keras
import tensorflow as tf
import time
from tqdm import tqdm
import tensorflow.keras.backend as K
import random
import os
import pandas as pd

# Set random seeds for reproducibility and network configuration
random.seed(142)
np.random.seed(142)
tf.random.set_seed(142)

batch_size = 2
xr = 256
lr_image_shape = (xr, xr, 1) 
hr_image_shape = (xr*4, xr*4, 1)
smoothing_factor = 0.07

## optimzer
#optimizer = keras.optimizers.Adagrad(learning_rate=1e-3)
optimizer = keras.optimizers.Adamax(learning_rate=5e-5, clipvalue=0.5)
#optimizer = keras.optimizers.Adam(learning_rate=4e-4, beta_1 =0.5, clipnorm=0.5)

# Build and compile the discriminator
discriminator = build_discriminator(hr_image_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
discriminator.trainable = False
discriminator.summary()

# build feature extractor 
def perceptual_loss(y_true, y_pred): 
    return K.mean(K.abs(y_true - y_pred))

feature_extractor = vgg19_feature_extractor(hr_image_shape) #inception_v3_feature_extractor(hr_image_shape)
feature_extractor.trainable = False
feature_extractor.summary()


# Build the generator
generator = build_generator(lr_image_shape)
generator.summary() 

# Build and compile the combined model
sr_gan = build_sr_gan(generator, discriminator, feature_extractor, lr_image_shape)
# Print the model summary
sr_gan.summary()
sr_gan.load_weights('weight - sr_gan_res128.h5')


sr_gan.compile(loss=[combined_MAE_SSIM_ASL_loss, 'mae', 'binary_crossentropy'], optimizer=optimizer, loss_weights=[1, 1,1e-3]) # loss weights for (generated img / feature / discriminator)


# Initialize the data generator 
dataset_ar_path1 = 'I:\\Datasets\\High Resolution Single Sequences\\Schwannoma High Res\\Processed Arrays\\'
dataset_ar_path2 = 'I:\\Datasets\\High Resolution Single Sequences\\UPENN-GBM\\'

train_generator = DataGenerator([dataset_ar_path1, dataset_ar_path2], batch_size=batch_size, input_dims=lr_image_shape, output_dims= hr_image_shape, shuffle=True)
 

### save history
def save_model_history(new_history, csv_file_path='model_GAN_df.csv'):
    
    # Check if the CSV file exists
    if os.path.exists(csv_file_path):
        # If it exists, read the existing data into a DataFrame
        existing_histories_df = pd.read_csv(csv_file_path, index_col=0)
    else:
        # If it doesn't exist, create an empty DataFrame
        existing_histories_df = pd.DataFrame()

    # Convert the new history into a DataFrame
    new_history_df = pd.DataFrame(new_history)

    # Append the new history to the existing DataFrame
    updated_histories_df = pd.concat([existing_histories_df, new_history_df], ignore_index=True)

    # Save the updated DataFrame back to the CSV file
    updated_histories_df.to_csv(csv_file_path)
    
    
# Initialize parameters of training
batch_size = train_generator.batch_size
epochs = 10000 # number of epochs
max_steps_per_epoch =  len(train_generator) 

## supress steps progress bar for train_on_patch
tf.keras.utils.disable_interactive_logging()


model_histories = { 'Resolution':[],
                   'epochs':[],
                    'Generator SSIM': [],
                   'Generator PSNR': [],
                    'Generator Total loss': [],
                   'Generator BCE loss': [],
                   'Generator MAE_SSIM loss': [],
                   'Generator Features_loss':[],
                   'Discriminator Real loss': [],
                   'Discriminator Fake loss': [],
                   'Discriminator Total loss':[]}


# Cur factor, adjust start_epoch if you are doing continuation of learning
cur_learn_factor = np.linspace(0.0001, 1, epochs)

start_epoch = 0

# Train the SR-GAN 
for epoch in range(start_epoch, epochs):
    epoch += 1
    gc.collect()
    print(f'Epoch {epoch}/{epochs}')
    
    total_generator_loss = 0
    total_generator_MAE_SSIM_loss = 0   
    total_generator_MAE_features_loss = 0
    total_generator_BCE_loss = 0

    total_generator_SSIM = 0
    total_generator_PSNR = 0
    
    total_discriminator_real_loss = 0
    total_discriminator_fake_loss = 0
    total_discriminator_total_loss = 0

    steps = 0
    steps_per_epoch = int(max_steps_per_epoch * cur_learn_factor[epoch])
    print(f'{steps_per_epoch} steps per current epoch')
    
    start_time = time.time()
    
    for step in tqdm(range(steps_per_epoch), total=steps_per_epoch):
        # Generate batch of images
        low_res_images, high_res_images = train_generator.__getitem__(step)

        # Train the discriminator
        discriminator.trainable = True
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        
        # Apply label smoothing
        real_labels_smoothed = real_labels * (1 - smoothing_factor)
        fake_labels_smoothed = fake_labels + smoothing_factor
    
        generated_hr_images = generator.predict(low_res_images)
        
        discriminator_loss_real = discriminator.train_on_batch(high_res_images, real_labels_smoothed)
        discriminator_loss_fake = discriminator.train_on_batch(generated_hr_images, fake_labels_smoothed)    
        discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)
        discriminator.trainable = False
        
        # Train the generator
        feature_extractor.trainable = False
        HR_features = feature_extractor.predict_on_batch(high_res_images)
        generator_loss = sr_gan.train_on_batch(low_res_images, [high_res_images, HR_features, real_labels])
        
        # Evaluate image quality and accumulate metrics
        generator_SSIM = calculate_ssim(high_res_images, generated_hr_images).numpy().mean()
        generator_PSNR = calculate_psnr(high_res_images, generated_hr_images).numpy().mean()

        # Accumulate losses for the epoch
        total_generator_SSIM += generator_SSIM * 100
        total_generator_PSNR += generator_PSNR
        
        total_generator_loss += generator_loss[0]
        total_generator_MAE_SSIM_loss += generator_loss[1]
        total_generator_MAE_features_loss += generator_loss[2]
        total_generator_BCE_loss += generator_loss[3]
        total_discriminator_real_loss += discriminator_loss_real[0]
        total_discriminator_fake_loss += discriminator_loss_fake[0]
        total_discriminator_total_loss += discriminator_loss[0]
        steps += 1
        
        train_generator.on_epoch_end()
        
    if epoch // 10:
        sr_gan.save_weights('weight - sr_gan_res{}.h5'.format(xr))
        discriminator.save_weights('weight - discriminator_res{}.h5'.format(xr))
        generator.save_weights('weight - generator_res{}.h5'.format(xr))
    duration_epoch = (time.time() - start_time)
    
    # Calculate average and store losses for the epoch
    avg_generator_SSIM = total_generator_SSIM / steps
    avg_generator_PSNR = total_generator_PSNR / steps

    
    avg_generator_total_loss = total_generator_loss / steps
    avg_generator_BCE_loss = total_generator_BCE_loss / steps
    avg_generator_MAE_SSIM_loss = total_generator_MAE_SSIM_loss / steps
    avg_generator_MAE_features_loss = total_generator_MAE_features_loss / steps
    avg_discriminator_real_loss = total_discriminator_real_loss / steps
    avg_discriminator_fake_loss = total_discriminator_fake_loss / steps
    avg_discriminator_total_loss = total_discriminator_total_loss / steps
    

    print('|||| The Average Generator SSIM: {:.2f} \n|||| The Average Generator PNSR: {:.2f}'.format(avg_generator_SSIM, avg_generator_PSNR))

    print('| The Average Generator Total Loss: {:.4f}'.format(avg_generator_total_loss))
    print('|||| The Average Generator BCE Loss: {:.4f} \n|||| The Average Generator MAE/SSIM/Sharpening Loss: {:.4f}  \n|||| The Average Generator Features loss: {:.4f} '.format(avg_generator_BCE_loss, avg_generator_MAE_SSIM_loss, avg_generator_MAE_features_loss))

    print('| Total Discriminator Loss: {:.4f} \n|||| Real Discriminator Loss: {:.4f} \n|||| Fake Discriminator Loss: {:.4f} \n| Total Time in {} seconds.'.format(avg_discriminator_total_loss, avg_discriminator_real_loss, avg_discriminator_fake_loss , round(duration_epoch)))

    model_histories['Generator SSIM'].append(avg_generator_SSIM)     
    model_histories['Generator PSNR'].append(avg_generator_PSNR)

    model_histories['Generator Total loss'].append(avg_generator_total_loss)     
    model_histories['Generator BCE loss'].append(avg_generator_BCE_loss)
    model_histories['Generator MAE_SSIM loss'].append(avg_generator_MAE_SSIM_loss)
    model_histories['Generator Features_loss'].append(avg_generator_MAE_features_loss)
    model_histories['Discriminator Real loss'].append(avg_discriminator_real_loss)
    model_histories['Discriminator Fake loss'].append(avg_discriminator_fake_loss)
    model_histories['Discriminator Total loss'].append(avg_discriminator_total_loss)
    model_histories['Resolution'].append(xr)
    model_histories['epochs'].append(epoch)


    # Generate and plot a random image from the generator
    
    n_int = np.random.randint(0, len(train_generator))
    low_res_images, high_res_images = train_generator.__getitem__(n_int)
    noisy_image = low_res_images[0]
    noisy_image = noisy_image[:,:,0]
    generated_image = generator.predict(low_res_images[0].reshape(1, *lr_image_shape))
    truth_image = high_res_images[0]

    
    plt.figure(figsize=(26, 10))
    plt.suptitle('EPOCH {}, Total Duration {} seconds'.format(epoch, round(duration_epoch)))
    plt.subplot(1, 3, 1)
    plt.title('Nosiy Image')      
    #low_res_img = np.ones(hr_image_shape[:-1], dtype=np.float32)
    start_x = (hr_image_shape[0] - lr_image_shape[0]) // 2
    start_y = (hr_image_shape[1] - lr_image_shape[1]) // 2
    #low_res_img[start_x:start_x+lr_image_shape[0], start_y:start_y+lr_image_shape[1]] = noisy_image.reshape(lr_image_shape[:-1])
    plt.imshow(noisy_image, cmap='gray', aspect='auto')


    plt.subplot(1, 3, 2)
    plt.title('Generated Image')
    plt.imshow(generated_image[0].reshape(hr_image_shape[:-1]), cmap='gray', aspect='auto')
        
    plt.subplot(1, 3, 3)
    plt.title('Ground Truth')
    plt.imshow(truth_image.reshape(hr_image_shape[:-1]), cmap='gray', aspect='auto')
    plt.tight_layout(pad=1.5)
    plt.show()
    
    save_model_history(model_histories)
            
#sr_gan.save_weights('sr_gan_res{}.h5'.format(xr))
#discriminator.save_weights('discriminator_res{}.h5'.format(xr))
#generator.save_weights('generator_res{}.h5'.format(xr))


# Plot the values from the lists in model_histories


colors = ['red', 'orange', 'pink', 'maroon', 'blue', 'purple', 'lime', 'darkblue', 'navy', 'indigo', 'lightslategray']

plt.figure(figsize=(20, 14))
for i, (key, values) in enumerate(model_histories.items()):
    if key in ['Generator SSIM', 'Generator PSNR']:
        plt.plot(range(len(values)), values, label=key, color=colors[i])

#plt.plot(range(epochs), cur_learn_factor, label='curiculum Factor', color=colors[-1])
# Add labels and legend
plt.xlabel('Epochs')
plt.ylabel('Metrics in 100%')
plt.title('Model Losses')
plt.legend()
plt.xticks(range(len(values)), map(int, range(len(values))))
# Show plot
#plt.savefig('residual SRGAN.png', dpi = 300)
plt.show()


plt.figure(figsize=(30, 21))
for i, (key, values) in enumerate(model_histories.items()):
    if key not in ['Generator SSIM', 'Generator PSNR']:
        if 'Generator' in key:
            plt.plot(range(len(values)), values, label=key, color=colors[i])
        
#plt.plot(range(epochs), cur_learn_factor, label='curiculum Factor', color=colors[-1])
# Add labels and legend
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Losses')
plt.legend()
plt.yscale('log')
if len(values) < 2:
    plt.xticks(range(len(values)), map(int, range(len(values))))
# Show plot
#plt.savefig('residual SRGAN.png', dpi = 300)
plt.show()


plt.figure(figsize=(20, 14))
for i, (key, values) in enumerate(model_histories.items()):
    if key not in ['Generator SSIM', 'Generator PSNR']:
        if 'Discriminator' in key:
            plt.plot(range(len(values)), values, label=key, color=colors[i])
        
#plt.plot(range(epochs), cur_learn_factor, label='curiculum Factor', color=colors[-1])
# Add labels and legend
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Losses')
plt.legend()
plt.yscale('log')
if len(values) < 2:
    plt.xticks(range(len(values)), map(int, range(len(values))))
# Show plot
#plt.savefig('residual SRGAN.png', dpi = 300)
plt.show()


   
    
save_model_history(model_histories, csv_file_path='model_GAN_df.csv')
