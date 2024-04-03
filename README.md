# Anime_GAN
This repository records the bonus part of WPI DS504 Project 3. 

This project is a PyTorch implementation of Deep Convolutional Generative Adversarial Networks (DCGAN), aimed at generating high-quality anime face images. As a variant of Generative Adversarial Networks (GANs), DCGAN improves the training stability and image quality of GANs through the architecture of deep convolutional networks. This project demonstrates the application of DCGAN in the domain of anime image generation, providing model implementations for the Generator and Discriminator along with simple usage examples.



Author: Dong Tang
# Dependencies
* [Python 3.11](https://www.continuum.io/downloads)
* [PyTorch 2.2.2+cu121](http://pytorch.org/)
* [numpy 1.26.4, matplotlib 3.8.3, scipy 1.12.0](https://www.scipy.org/install.html)
* [imageio 2.34.0](https://pypi.org/project/imageio/)
* [tqdm 4.66.2](https://pypi.org/project/tqdm/)


# Dataset
Please download and extract the dataset from the provided link and extract it to the root directory. https://drive.google.com/drive/folders/1mCsY5LEsgCnc0Txv0rpAUhKVPWVkbw5I



# Model

## Generator
The Generator model takes a 100-dimensional latent vector as input and generates 3-channel (RGB) images with dimensions of 96x96 pixels.
It consists of five transposed convolutional layers, each followed by batch normalization (except for the final layer) and ReLU activation (tanh activation in the final layer).
The model gradually upsamples the input vector to the output image through its convolutional layers.

## Discriminator
The Discriminator model evaluates the authenticity of the input images, distinguishing between real and generated (fake) images.
It comprises five convolutional layers. The first layer uses LeakyReLU activation without batch normalization. Subsequent layers include batch normalization and LeakyReLU activation, with the final layer outputting a single value through a sigmoid activation function.
The model progressively downsamples the input image to a single value that represents the probability of the image being real.

The final generated model is saved in saved/generator_epoch.pt and saved/discriminator_epoch.pt

# run the code
```bash 
python main.py 
```

1 epochs result
![epoch_0.png](saved%2Fimg%2Fepoch_0.png)
500 epochs result