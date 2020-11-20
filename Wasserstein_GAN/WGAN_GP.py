import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
import torch.functional as F

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", required=True, help="Location of the dataset folder")
parser.add_argument("-b", "--batch_size", required = True, type = int, help = "Batch Size of the dataset for training")
parser.add_argument("-ic","--image_channels", required=True, type = int, help="Channels in the input image")
parser.add_argument("-e", "--epochs", required = True, type = int, help = "Number of epochs to run the training loop")
parser.add_argument("-ci", "--critic_iterations", required = True, type = int, help = "Number of iterations for the Critic")

args = vars(parser.parse_args())

# Implementation of the Wasserstein GAN
# WGAN Has better stability and loss means something for WGAN: Its a termination criteria
# WGAN Also prevents Mode Collapse(Model only outputs specific classes)
# When Discriminator converged to 0 obtained great results

class Generator(nn.Sequential):
    """
    z_dim: 
    channels_img: Input channels(for example for an RGB image this value is 3)
    features_g: Size of the output feature map(In this case its 64x64)
    """
    def __init__(self, z_dim, channels_img, features_g):
        
        modules = [self._block(z_dim, features_g*16, 4, 1, 0),
                   self._block(features_g*16, features_g*8, 4, 2, 1),
                   self._block(features_g*8, features_g*4, 4, 2, 1),
                   self._block(features_g*4, features_g*2, 4, 2, 1),
                   nn.ConvTranspose2d(features_g*2, channels_img, 4, 2, 1),
                   nn.Tanh()]
        
        super(Generator, self).__init__(*modules)
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
class Critic(nn.Sequential):
    
    def __init__(self, channels_img, features_d):
        
        modules = [nn.Conv2d(channels_img, features_d, kernel_size = 4, stride = 2, padding = 1), #32x32
                   nn.LeakyReLU(0.2, inplace=True),
                   self._block(features_d, features_d*2, 4, 2, 1),# 16x16
                   self._block(features_d*2, features_d*4, 4, 2, 1), #8x8
                   self._block(features_d*4, features_d*8, 4, 2, 1), #4x4
                   nn.Conv2d(features_d*8, 1, kernel_size = 4, stride = 2, padding = 0)]
        
        super(Critic, self).__init__(*modules)
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        
        return nn.Sequential(
         nn.Conv2d(in_channels, out_channels,kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1, 0.02)
        nn.init.constant_(m.bias.data, 0)

root = args["dataset"]
LEARNING_RATE = 0.00005
BATCH_SIZE = args["batch_size"]
Z_DIM = 100
FEATURES_DISC = 64
FEATURES_GEN = 64
IMAGE_CHANNELS = args["image_channels"]
NUM_EPOCHS = args["epochs"]
IMAGE_SIZE = 64
CRITIC_ITERATIONS = args["critic_iterations"]
WEIGHT_CLIP = 0.01

dataset = datasets.ImageFolder(root = root , transform=transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))

dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)

device = torch.device("gpu" if torch.cuda.is_available() else "cpu")

########################
# Generator and Discriminator Model objects
########################
generator = Generator(Z_DIM,IMAGE_CHANNELS,FEATURES_GEN).to(device)
critic = Critic(IMAGE_CHANNELS,FEATURES_DISC).to(device)

########################
# Weight Initialization for the model
########################
generator.apply(weights_init)
critic.apply(weights_init)

########################
# Optimizers for Critic and the Generator
########################
optimizer_gen = optim.RMSprop(generator.parameters(), lr = LEARNING_RATE)
optimizer_critic = optim.RMSprop(critic.parameters(), lr = LEARNING_RATE)

#######################
# Create tensorboard SummaryWriter objects to display generated fake images and associated loss curves
#######################
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
loss_curves = SummaryWriter(f"logs/loss_curves")

#######################
# Create a batch of latent vectors. Will be used to to do a single pass through the generator after 
# the training has terminated
#######################
fixed_noise = torch.randn((64, Z_DIM, 1, 1)).to(device)

step = 0 # For printing to tens

for epoch in range(NUM_EPOCHS):
    
    # Unsupervised
    for batch_idx, (real, _) in enumerate(dataloader):
        
        # The real world images
        real = real.to(device)
        
        #####################################################
        # Train the Critic
        #####################################################
        
        for _ in range(CRITIC_ITERATIONS):
            critic.zero_grad()
            # Latent noise
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
            # Pass the latent vector through the generator
            fake = generator(noise)     
            critic_real = critic(real).view(-1)
            critic_fake = critic(fake.detach()).view(-1)
            ## Loss for the critic. Taking -ve because RMSProp are designed to minimize 
            ## Hence to minimize something -ve is equivalent to maximizing that expression
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
            loss_critic.backward()
            optimizer_critic.step()
            
            for p in critic.parameters():
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)
        
        #############################
        # Train the generator minimizing -E[critic(gen_fake)]
        #############################
        generator.zero_grad()
        output = critic(fake).view(-1)
        loss_gen = -torch.mean(output)
        loss_gen.backward()
        optimizer_gen.step()
        
        if batch_idx % 100 == 0:
            
            print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
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
#                 loss_curves.add_scalar("generator", {loss_gen, global_step=step)
                loss_curves.add_scalars("curves", {
                    "generator":loss_gen, "critic":loss_critic
                }, global_step = step)
#                 loss_curves.add_scalar("discriminator", loss_disc, global_step = step)
                
            step += 1 # See progression of images