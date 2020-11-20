# GAN-Architectures

[![Packagist](https://img.shields.io/packagist/l/doctrine/orm.svg)](LICENSE.md)

### Author
Shantam Bajpai 

### Architectures Implemented
1. DCGAN (Deep Convolutional Generative adversarial network)

2. WGAN (Wasserstein Generative Adversarial Network)

3. WGAN With Gradient Penalty

4. CGAN (Conditional Generative adversarial network)

### Research Paper References
DCGAN: https://arxiv.org/pdf/1511.06434.pdf

WGAN: https://arxiv.org/pdf/1701.07875.pdf

WGAN-GP: https://arxiv.org/pdf/1704.00028v3.pdf

### Dataset Used
The dataset used to train the Generative adversarial networks was the celeba dataset which is a large scale face attributes dataset with more than 200K Celebrity faces.

### Visualizations

All the visualizations pertaining to the generation of fake images and the Generator-Discriminator loss curves will be carried out in tensorboard.  

### Fake Images Generated using DCGAN !
![Screenshot](DCGAN/Fake_images.png) 

### Loss curves for DCGAN
![Screenshot](DCGAN/loss_curves.PNG)

### Fake Images Generated using WGAN for 5 epochs !
![Screenshot](Wasserstein_GAN/fake_images_WGAN.png)

### Loss Curves for WGAN
![Screenshot](Wasserstein_GAN/loss_curves_WGAN.png)
