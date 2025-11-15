![image](https://github.com/AmrAMHD/BrainSR/assets/170816158/d9e13c18-7a12-4518-a81b-24e628b14df3)

# BrainSR: Super-resolution and Image Denoising Network for Brain Scans (v1.0.3)
BrainSR is a Python-based software application designed to enhance brain scan images by utilizing deep learning techniques. It generates cleaner images with a resolution four times higher than the original scans, significantly improving the quality of medical imaging.

# Version notes:
Compared to version 1.0.0, version 1.0.3 demonstrates improved robustness to artifacts and noise, along with enhanced performance. This improvement is achieved through training on a dataset of 250,000 high-resolution medical scans.

## Technical Specifications:
- Backend: Python
-	Deep Learning Frameworks: TensorFlow, Keras
-	Medical Image Processing Libraries: Pydicom, NumPy
-	Visualization Library: Matplotlib
#### Hardware Requirements:
-	Minimum: 16 GB RAM (preferred 32 GB RAM)
-	Recommended: NVIDIA GPU with at least 8 GB VRAM (preferred 12 GB VRAM

## Core Technology:
BrainSR relies on Convolutional Neural Networks (CNNs) with residual blocks and up-sampling techniques to achieve superior image resolution and denoising. This architecture helps in retaining important details while effectively reducing noise and enhancing the overall image quality.

## Potential Benefits:
- Improved Image Quality: Corrects poor-quality scans, providing clearer images for better analysis.
- Increased Efficiency: Allows for obtaining low-resolution scans quickly, enabling scanners to process more patients per day and reducing waiting lists. 


## Network Architecture:
BrainSR utilizes a Conditional Generative Adversarial Network (cGAN) architecture to achieve high-quality super-resolution and denoising of brain scan images. The network consists of a generator and a discriminator, each with specific components designed for optimal performance. Additionally, the network architecture is flexible and can be refined to adapt to any input and output resolutions required by the customer. 
![image](https://github.com/AmrAMHD/BrainSR/assets/170816158/d600ebb3-4c06-4b5c-89b4-da0388d34a41)

-	### Generator
The generator in BrainSR is designed to take low-resolution images (as low as 160x160 pixels) and produce high-resolution outputs (up to 640x640 pixels, achieving 4x super-resolution). The generator's structure includes:
- Pre-residual Block: Responsible for initial convolution and detail extraction from the input image, setting the stage for subsequent processing.
-	16 Propagations through Residual Blocks: These blocks help in preserving the image's high-frequency details and mitigating the vanishing gradient problem. Residual blocks allow the network to maintain and enhance fine details by learning residual functions with reference to the input.
-	Post-residual Block: Stabilizes the output by refining and consolidating the enhancements made by the residual blocks.
-	Final Up-sampling: Uses up-sampling layers to scale up the image resolution to the desired 640x640 pixels.
- ### Discriminator
The discriminator evaluates the authenticity of the generated high-resolution images. It comprises:
-	9 Deconvolution Blocks: These blocks progressively deconvolve the input to discriminate between real high-resolution images and the generated ones.
-	Feature Extractors: Two powerful pre-trained networks, VGG19 and InceptionV3, are integrated to provide additional refinement and ensure the high quality of the generated images. These feature extractors help in capturing complex patterns and details that are crucial for distinguishing between real and synthetic images.
- ### Training Process
-	The training process of BrainSR is based on the concept of curriculum learning, which gradually increases the complexity of the training tasks. This approach helps in achieving stability during training, as the network initially learns simpler tasks and progressively tackles more challenging ones. By doing so, the network can better adapt and refine its weights, leading to more robust and accurate image generation.
-	This sophisticated combination of a cGAN architecture, detailed generator and discriminator designs, and a curriculum learning-based training process enables BrainSR to effectively enhance brain scan images, providing significant improvements in resolution and clarity.


## Example Input – Low-Res Moisy and Output: Super-Sampled 4X clean :

![image](https://github.com/AmrAMHD/BrainSR/assets/170816158/44f6d17a-5e8c-4b35-81f7-ef1b2cc3e1b0)
![image](https://github.com/AmrAMHD/BrainSR/assets/170816158/4bd888b2-8a1f-406f-aca8-982ea3dd1c25)
![image](https://github.com/AmrAMHD/BrainSR/assets/170816158/ec915cea-dcf3-4a80-bba5-e3a0b2f8b6d0)
![image](https://github.com/AmrAMHD/BrainSR/assets/170816158/aced86b2-26d4-493d-9264-854c1996bb3f)

## Interactive User friendly web app
Streamlit with PyDicom: Web-based interface for uploading DICOM scans and denoising. Includes noise addition and blur functionality for model potency testing. We've added two sliders to control the amount of noise and blur applied to the low resolution image. This allows you to gradually increase the difficulty of the task submitted to the denoiser, testing its robustness to artifacts.

![image](https://github.com/AmrAMHD/BrainSR/assets/170816158/60030de7-8d0a-4eed-ac1b-c1ce568fc645)


## Adaptability
One of the key features of BrainSR is its adaptability. The network can be refined to accommodate various input and output resolutions as required by the customer. This flexibility ensures that BrainSR can be tailored to meet specific needs, making it a versatile tool for enhancing medical imaging.
This sophisticated combination of a cGAN architecture, detailed generator and discriminator designs, and a curriculum learning-based training process enables BrainSR to effectively enhance brain scan images, providing significant improvements in resolution and clarity.

# Disclaimer:
BrainSR is currently a research tool under development. It is not intended for clinical diagnosis at this stage. Further validation and regulatory approval are required before it can be used in clinical settings.

## Future Developments:
- 3D U-Net Implementation: Explore the use of 3D U-Nets for improved anatomical detail in synthetic contrast scans.
-	Multi-modal Learning: Integrate additional clinical data (e.g., patient history) to refine the contrast synthesis process.
-	Regulatory Compliance: Pursue regulatory approval for clinical use of BrainSR.


*This documentation provides a basic overview of BrainSR v1.0. As the project progresses, this document will be updated to reflect advancements and improvements*

**Dr. Amr Muhammed, M.D FRCR, M.Sc. PhD**
**amr.muhammed@med.sohag.edu.eg**


