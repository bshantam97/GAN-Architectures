import os 
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
import torch.optim as optim 
import torch.functional as F
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", required=True, help="Location of the dataset folder")
parser.add_argument("-b", "--batch_size", required = True, help = "Batch Size of the dataset for training")
parser.add_argument("-ic","--image_channels", required=True, help="Channels in the input image")
parser.add_argument("-e", "--epochs", required = True, help = "Number of epochs to run the training loop")

args = vars(parser.parse_args())

class Discriminator(nn.Module):
    
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        # Did not use BatchNorm in the last layer of the generator and the first layer of the 
        # discriminator
        # Input: N x channels_img x 64 x 64
        self.discriminator = nn.Sequential(
            nn.Conv2d(channels_img, features_d, kernel_size = 4, stride = 2, padding = 1), #32x32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1),# 16x16
            self._block(features_d*2, features_d*4, 4, 2, 1), #8x8
            self._block(features_d*4, features_d*8, 4, 2, 1), #4x4
            nn.Conv2d(features_d*8, 1, kernel_size = 4, stride = 2, padding = 0), #1x1(Prediction)
            nn.Sigmoid()
        )
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self,x):
        return self.discriminator(x)
    
class Generator(nn.Module):
    
    # Here channels_img is nothing but the inputs channels 
    # and features_g is nothing but the output channels
    def __init__(self, z_dim, channels_img, features_g):
        
        super(Generator, self).__init__()
        
        self.net = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            self._block(z_dim, features_g*16, 4, 1, 0), # N x f_g*16(1024=64*64) x 4 x 4
            self._block(features_g*16, features_g*8, 4, 2, 1), # f_g*16 x f_g*8 x 8 x 8
            self._block(features_g*8, features_g*4, 4, 2, 1), # f_g*8 x f_g*4 x 16 x 16
            self._block(features_g*4, features_g*2, 4, 2, 1), #32 x 32
            nn.ConvTranspose2d(features_g*2, channels_img, 4, 2, 1),
            nn.Tanh() #[-1,1]
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self,x):
        return self.net(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0) 

# def test():
#     N, in_channels, H, W = 8, 3, 64, 64
#     z_dim = 100
#     X = torch.randn((N, in_channels, H, W))
#     disc = Discriminator(in_channels,8)
#     assert disc(X).shape == (N, 1, 1, 1) # One Value per example
#     gen = Generator(z_dim, in_channels, 64)
#     z = torch.randn((N, z_dim, 1, 1))
#     assert gen(z).shape == (N, in_channels, H, W) # Ouput Generated image
#     print("Success")
# test()

## The training Setup
root = "C:\\Users\\shant\\img_align_celeba\\img_align_celeba\\"
LEARNING_RATE = 2e-4
BATCH_SIZE = args["batch_size"]
IMAGE_SIZE = 64
# Image Channels in the generator output and input to the discriminator 
IMAGE_CHANNELS = args["image_channels"]

# Latent Space Dimensions
Z_DIM = 100
NUM_EPOCHS = args["epochs"]

# feature size in the discriminator
FEATURES_DISC = 64

# feature size in the generator
FEATURES_GEN = 64

# Setup Transforms
dataset = datasets.ImageFolder(os.path.join(args["dataset"]),transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(IMAGE_CHANNELS)], [0.5 for _ in range(IMAGE_CHANNELS)])
]))

# DataLoader
dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle=True, num_workers = 2)

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(Z_DIM, IMAGE_CHANNELS, FEATURES_GEN).to(device)
discriminator = Discriminator(IMAGE_CHANNELS,FEATURES_DISC).to(device)

generator.apply(weights_init)
discriminator.apply(weights_init)

generator_optimizer = optim.Adam(generator.parameters(), lr = LEARNING_RATE, betas = (0.5,0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr = LEARNING_RATE, betas = (0.5,0.999))

# Loss 
criterion = nn.BCELoss()

# Batch of Latent Vectors
fixed_noise = torch.randn((64, Z_DIM, 1, 1)).to(device)

writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
loss_curves = SummaryWriter(f"logs/loss_curves")

step = 0 # Printing to tensorboard
# generator.train()
# discriminator.train()

for epoch in range(NUM_EPOCHS):
    
    # Unsupervised
    for batch_idx, (real, _) in enumerate(dataloader):
        discriminator.zero_grad()
        # Latent noise
        noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
        # The real world images
        real = real.to(device)
        # Pass the latent vector through the generator
        fake = generator(noise)
        #####################################################
        # Train the discriminator max log(D(x)) + log(1-D(G(z)))
        #####################################################
        
        disc_real = discriminator(real).view(-1)
        # log(D(x)), y_n = ones hence only logx_n term is left. Refer to Pytorch Documentation
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        
        loss_disc_real.backward()
        
        disc_fake = discriminator(fake.detach()).view(-1)
        # Subtracting 
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        
        # Addition of gradients from the all the real and fake samples
        loss_disc = (loss_disc_fake + loss_disc_real)
        
        loss_disc_fake.backward() # Reutilize fake, but pytorch removes intermeditate results
        discriminator_optimizer.step()
        
        ##################################
        # Train Generator max log(D(G(z)))
        ##################################
        generator.zero_grad()
        output = discriminator(fake).view(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        loss_gen.backward()
        generator_optimizer.step()
        
        if batch_idx % 100 == 0:
            
            print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )
            with torch.no_grad():
                fake = generator(fixed_noise)
            
                # The [:64] prints out the 4-D tensor BxCxHxW
                img_grid_real = torchvision.utils.make_grid(
                    real[:64], normalize = True)
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:64], normalize = True)
                
                ##########################
                # TensorBoard Visualizations
                ##########################
                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                loss_curves.add_scalar("generator", loss_gen, global_step=step)
                loss_curves.add_scalar("discriminator", loss_disc, global_step = step)
            step += 1 # See progression of images
