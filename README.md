# GAN-Architectures

[![Packagist](https://img.shields.io/packagist/l/doctrine/orm.svg)](LICENSE.md)

### Author
Shantam Bajpai 

### Architectures Implemented
1. DCGAN (Deep Convolutional Generative adversarial network)

2. WGAN (Wasserstein Generative Adversarial Network)

3. WGAN With Gradient Penalty

4. CGAN (Conditional Generative adversarial network)

5. EBGAN (Energy Based Generative Adversarial Network)

### Dataset Used
The dataset used to train the Generative adversarial networks was the celeba dataset which is a large scale face attributes dataset with more than 200K Celebrity faces and the MNIST Dataset (For conditional Wasserstein GAN-GP). 

## Tensorboard Visualizations

### Fake Images Generated using DCGAN 
![Screenshot](DCGAN/Fake_images.png) 

### Loss curves for DCGAN
![Screenshot](DCGAN/loss_curves.PNG)

### Fake Images Generated using WGAN for 5 epochs 
![Screenshot](Wasserstein_GAN/fake_images_WGAN.PNG)

### Loss Curves for WGAN (5 Epochs)
![Screenshot](Wasserstein_GAN/loss_curves_WGAN.PNG)

### Fake Images generated using WGAN-GP for 5 epochs 
![Screenshot](Wasserstein_GAN_GP/fake_images.PNG)

### Loss Curves for WGAN-GP
![Screenshot](Wasserstein_GAN_GP/loss_Curves.PNG)

### Fake Images generated using Conditional WGAN-GP after training for 20 Epochs
![Screenshot](Conditional_GAN/loss_curves.PNG)

### Loss Curves for Conditional WGAN-GP 
![Screenshot](Conditional_GAN/fake_images.PNG)
### Research Paper References
DCGAN: https://arxiv.org/pdf/1511.06434.pdf

WGAN: https://arxiv.org/pdf/1701.07875.pdf

WGAN-GP: https://arxiv.org/pdf/1704.00028v3.pdf

CGAN: https://arxiv.org/pdf/1411.1784.pdf

EBGAN: https://arxiv.org/pdf/1609.03126.pdf
