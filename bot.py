from pickle import FALSE
from telegram.ext import *
from io import BytesIO
import cv2
import numpy as np
import torch
import tensorflow as tf

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
from torchvision.utils import make_grid

from telegram import ReplyKeyboardMarkup, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.keyboardbutton import KeyboardButton

from random import randint
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

IMG = ''
TOKEN = os.getenv("TOKEN")

def build_menu(buttons, n_cols,
               header_buttons=None,
               footer_buttons=None):
    menu = [buttons[i:i + n_cols] for i in range(0, len(buttons), n_cols)]
    if header_buttons:
        menu.insert(0, [header_buttons])
    if footer_buttons:
        menu.append([footer_buttons])
    return menu


button_list = [
    KeyboardButton("About dataset"),
    KeyboardButton("About my CGAN"),
    KeyboardButton("Try image from liabrary"),
]


reply_markup = ReplyKeyboardMarkup(build_menu(button_list, n_cols=2), resize_keyboard=True)


def start(update, context):
        update.message.reply_photo(open('neuronica_logo.png', 'rb'), caption="Welcome to Neuronica!\nUpload a photo to start immidiatly or use menu to explore more information.", reply_markup=reply_markup)


def help(update, context):
        update.message.reply_text("""
        /start - begin dialogue\n
        /help - commands list
        """)

device = torch.device('cpu') #'cuda' if torch.cuda.is_available() else 


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
     
    paint = paint[0].transpose(1, 2, 0)
    plt.figure()
    
    plt.imshow(paint)
    large = cv2.resize(paint, (0,0), fx=2, fy=2) 
    cv2.imwrite("result.png", cv2.cvtColor(large*255, cv2.COLOR_RGB2BGR))


def handle_cgan(update, context):
        update.message.reply_photo(open('gan_arch.png', 'rb'), caption="StyleGAN - generative adversarial network (GAN) introduced by Nvidia researchers in December 2018.\n\nThe core idea of a GAN is based on the 'indirect' training through the discriminator, another neural network that is able to tell how much an input is 'realistic', which itself is also being updated dynamically. This means that the generator is not trained to minimize the distance to a specific image, but rather to fool the discriminator. This enables the model to learn in an unsupervised manner.\n\nGANs are similar to mimicry in evolutionary biology, with an evolutionary arms race between both networks.")
        update.message.reply_photo(open('GR2F.pt_vert.png', 'rb'), caption="My model scheme.\n\nModel was train for 200 epoches.")

def handle_dataset(update, context):
        update.message.reply_photo(open('facades.png', 'rb'), caption="Name: Facades Dataset\n\nDescription: Facades dataset consists of Building Facades & corresponding Segmentations.\nThis dataset was obtained from the original Pix2Pix Datasets directory available at: https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/. For more details about the dataset refer related Pix2Pix publication at: https://arxiv.org/abs/1611.07004\n\nData size: 506 images with split into train and test subsets.")


def handle_facades(update, context):
        global IMG
        rand = str(randint(1, 106))
        img = '.\\facades\\testA\\' + rand + '.jpg'
        IMG = img
        update.message.reply_photo(open(IMG, "rb"), caption="Type 'use' for segmentation.")

def handle_photo(update, context):
        file = context.bot.get_file(update.message.photo[-1].file_id)
        file.download("income.jpg")
        test_image2('income.jpg')      
        update.message.reply_photo(open('result.png', 'rb'), caption="Here is your segmented facade.")


def handle_use(update, context):
        test_image2(IMG)
        update.message.reply_photo(open('result.png', 'rb'), caption="Here is your segmented facade.")


updater = Updater(TOKEN, use_context=True)
dp = updater.dispatcher


dp.add_handler(CommandHandler("start", start))
dp.add_handler(CommandHandler("help", help))
dp.add_handler(MessageHandler(Filters.text(['About dataset']), handle_dataset))
dp.add_handler(MessageHandler(Filters.text(['About my CGAN']), handle_cgan))
dp.add_handler(MessageHandler(Filters.photo, handle_photo))
dp.add_handler(MessageHandler(Filters.text(['Try image from liabrary']), handle_facades))
dp.add_handler(MessageHandler(Filters.text(['use']), handle_use))


updater.start_polling()
updater.idle()
