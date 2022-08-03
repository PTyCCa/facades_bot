import os
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import random
import itertools
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import warnings
import glob
#%matplotlib inline
from torchvision.utils import make_grid


device = torch.device('cpu') #'cuda' if torch.cuda.is_available() else 


dir = 'C:/Users/PTyCCa/Downloads/CGAN/facades'
train_paintings_dir = dir + '/trainA/'
train_photos_dir = dir + '/trainB/'

paintings_addr = [train_paintings_dir+i for i in os.listdir(train_paintings_dir)]
photos_addr = [train_photos_dir+i for i in os.listdir(train_photos_dir)]

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=dimension, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(in_channels=dimension, out_channels=dimension*2, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(dimension * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(in_channels=dimension * 2, out_channels=dimension*4, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(dimension * 4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(in_channels=dimension * 4, out_channels=dimension*8, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(dimension * 8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(in_channels=dimension*8, out_channels=1, kernel_size=2, stride=1, padding=0, bias=False))

    def forward(self, x):

        x = self.main(x)
        x = F.avg_pool2d(x, x.size()[2:])
        return x

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
            )
        self.pool1 =  nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
            )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
            )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  

        self.bottleneck_conv =  nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.upsample0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        self.dec_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
            )
        self.upsample1 =  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
            )
        self.upsample2 =  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        self.upsample3 =  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)
            )

    def forward(self, x):
        e0 = self.enc_conv0(x)
        e1 = self.enc_conv1(self.pool0(e0))
        e2 = self.enc_conv2(self.pool1(e1))
        e3 = self.enc_conv3(self.pool2(e2))

        b = self.bottleneck_conv(self.pool3(e3))

        d0 = self.dec_conv0(torch.cat((self.upsample0(b), e3),dim=1))
        d1 = self.dec_conv1(torch.cat((self.upsample1(d0), e2),dim=1))
        d2 = self.dec_conv2(torch.cat((self.upsample2(d1), e1),dim=1))
        d3 = self.dec_conv3(torch.cat((self.upsample3(d2),e0 ),dim=1))  
        return d3


if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor

def show_image(photo_addr):
    img = Image.open(photo_addr)
    img_np = np.zeros((1, 3, 128, 128), dtype=np.float32)
    temp_np = np.asarray(img.resize((128, 128), Image.ANTIALIAS))
    plt.imshow(temp_np)
    
    img_np[0] = temp_np.transpose(2, 0, 1)
    
    img_np /= 255
    img_np = img_np * 2 - 1
    img_tensor = torch.from_numpy(img_np)
    img_var = Variable(img_tensor).type(dtype)
    
    paint_var = G_fake_2_real(img_var)
    paint = paint_var.data.cpu().numpy()
    paint = paint[0].transpose(1, 2, 0)
    #paint = (paint +1 )/2
    plt.figure()
    plt.imshow(paint)


the_model = torch.load(r"C:\Users\PTyCCa\Desktop\neuronica\cgan\GR2F")



import matplotlib
import cv2

def test_image2(img_addr):
    img = Image.open(img_addr)
    img_np = np.zeros((1, 3, 128, 128), dtype=np.float32)
    temp_np = np.asarray(img.resize((128, 128), Image.ANTIALIAS))
    plt.imshow(temp_np)
    
    img_np[0] = temp_np.transpose(2, 0, 1)
    
    img_np /= 255
    img_np = img_np * 2 - 1
    img_tensor = torch.from_numpy(img_np)
    img_var = Variable(img_tensor).type(dtype)
    

    paint_var = the_model(img_var)
    paint = paint_var.data.cpu().numpy()
    #matplotlib.image.imsave('name.png', paint)
    
    paint = paint[0].transpose(1, 2, 0)
    plt.figure()
    
    plt.imshow(paint)
    large = cv2.resize(paint, (0,0), fx=2, fy=2) 
    cv2.imwrite("filename.png", cv2.cvtColor(large*255, cv2.COLOR_RGB2BGR))
    plt.savefig("test.png")


