# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 22:52:05 2021

@author: Administrateur
"""

#%% Feta Basic model setup
import os
import numpy as np
import os
import nibabel as nib
from skimage import io
import cv2
import numpy as np
from skimage import io, exposure, img_as_uint, img_as_float
from PIL import Image 
import PIL 
import imageio
import scipy.misc
import pandas as pd
import os
import glob
import SimpleITK as sitk
import ast
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
############### design data generator 
import torch
class DatasetFeta(torch.utils.data.Dataset):
    def __init__(self, data, path, trans=None):
        self.path = path
        self.data = data
        self.trans = trans
        self.num_classes = 8

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        patient = self.data.iloc[ix].patient
        channel = self.data.iloc[ix].channel
        pathim1=os.path.join(self.path,patient)
        pathimg1=glob.glob(os.path.join(pathim1, 'anat', '*_T2w.nii.gz'))[0]
        img = nib.load(pathimg1).get_fdata()[...,channel]
        #img=img/img.max()
        
        pathimask=glob.glob(os.path.join(pathim1, 'anat', '*_dseg.nii.gz'))[0]
        mask = nib.load(pathimask).get_fdata()[...,channel].astype(np.int)
        if self.trans:
            t = self.trans(image=img, mask=mask)
            img = t['image']
            mask = t['mask'] 
        img_t = torch.from_numpy(img).float().unsqueeze(0)
        # mask encoding
        mask_oh = torch.nn.functional.one_hot(torch.from_numpy(mask).long(), self.num_classes).permute(2,0,1).float()
        return img_t, mask_oh

import pandas as pd

data_training = pd.read_csv('training_data_new.csv')
data_validtion = pd.read_csv('validation_data_new.csv')
#path = 'C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\feta_2.0\\feta_my'
training_path="/raid/Home/Users/aqayyum/EZProj/FeTA2021/Training/"
import albumentations as A
validation_path="/raid/Home/Users/aqayyum/EZProj/FeTA2021/Validation/"
trans = A.Compose([A.Resize(256, 256)])
import nibabel as nib
import random
import numpy as np
import matplotlib.pyplot as plt

data_train = DatasetFeta(data=data_training,path=training_path, trans=None)
print(len(data_train))
img, mask = data_train[0]
img.shape, mask.shape

data_val = DatasetFeta(data=data_validtion,path=validation_path, trans=None)
print(len(data_val))
img, mask = data_val[0]
img.shape, mask.shape



import matplotlib.pyplot as plt
import random

data={'train':DatasetFeta(data=data_training,path=training_path,trans=trans),
      'val':DatasetFeta(data=data_validtion,path=validation_path,trans=trans)}
## check dataset image shape and mask
imgs, masks = next(iter(data['train']))
imgs.shape, masks.shape
batch_size=12
dataloader = {
    'train': torch.utils.data.DataLoader(data['train'], batch_size=batch_size, shuffle=True, pin_memory=True),
    'val': torch.utils.data.DataLoader(data['val'], batch_size=batch_size, shuffle=False, pin_memory=True),
}
# imgs, masks = next(iter(dl['train']))
# imgs.shape, masks.shape

# import matplotlib.pyplot as plt

# r, c = 5, 5
# fig = plt.figure(figsize=(5*r, 5*c))
# for i in range(r):
#     for j in range(c):
#         ix = c*i + j
#         ax = plt.subplot(r, c, ix + 1)
#         ax.imshow(imgs[ix].squeeze(0), cmap="gray")
#         mask = torch.argmax(masks[ix], axis=0).float().numpy()
#         mask[mask == 0] = np.nan
#         ax.imshow(mask, alpha=0.5)
#         ax.axis('off')
# plt.tight_layout()
# plt.show()

#%% define model
################################ define the model #################
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            ###self.resblock= ResBlock(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bRV1w6eCQFVVW3Q9RZeanm2bC9hjAT7d
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
# New Residule Block    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.down_sample = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.double_conv(x)
        out = self.relu(out + identity)
        return out

class ResUNet(nn.Module):
    """ Full assembly of the parts to form the complete network """
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(ResUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.res1= ResBlock(64,64)
        self.down1 = Down(64, 128)
        self.res2= ResBlock(128, 128)
        self.down2 = Down(128, 256)
        self.res3= ResBlock(256, 256)
        self.down3 = Down(256, 512)
        self.res4= ResBlock(512, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        res1= self.res1(x1) 
        # print("1st conv block", x1.shape)
        # print("1st res block", res1.shape)
        x2 = self.down1(x1)
        res2= self.res2(x2)
        # print("sec conv block", x2.shape)
        # print("sec res block", res2.shape)
        x3 = self.down2(x2)
        res3= self.res3(x3)
        # print("3rd conv block", x3.shape)
        # print("3rd res block", res3.shape)
        x4 = self.down3(x3)
        res4= self.res4(x4)
        # print("4 conv block", x4.shape)
        # print("4 res block", res4.shape)
        x5 = self.down4(x4)
        #print("Base down ", x5.shape)
        x = self.up1(x5, res4)
        #print("1st up block", x.shape)
        x = self.up2(x, res3)
        #print(" sec up block", x.shape)
        x = self.up3(x, res2)
        #print("3rd up block", x.shape)
        x = self.up4(x, res1)
   
        logits = self.outc(x)

        return logits

# generate random input (batch size, channel, height, width)
inp=torch.rand(1,1,256,256)
inp.shape
    
# Giving Classes & Channels
n_classes=8
n_channels=1

#Creating Class Instance of Model Inf_Net_UNet Class
model =ResUNet(n_channels, n_classes)

# Giving random input (inp) to the model
out=model(inp)

print(out.shape)

########### define the training and testing function ###########
import os
import numpy as np
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
# modelUdense=ResUNet(n_channels, n_classes)
# print(modelUdense)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 100
lr = 3e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#%% define training and validation function
#second training function for optimizing the model

def IoU(pr, gt, th=0.5, eps=1e-7):
    pr = torch.sigmoid(pr) > th
    gt = gt > th
    intersection = torch.sum(gt * pr, axis=(-2,-1))
    union = torch.sum(gt, axis=(-2,-1)) + torch.sum(pr, axis=(-2,-1)) - intersection + eps
    ious = (intersection + eps) / union
    return torch.mean(ious).item()


def iou(outputs, labels):
    # check output mask and labels
    outputs, labels = torch.sigmoid(outputs) > 0.5, labels > 0.5
    SMOOTH = 1e-6
    # BATCH x num_classes x H x W
    B, N, H, W = outputs.shape
    ious = []
    for i in range(N-1): # we skip the background
        _out, _labs = outputs[:,i,:,:], labels[:,i,:,:]
        intersection = (_out & _labs).float().sum((1, 2))  
        union = (_out | _labs).float().sum((1, 2))         
        iou = (intersection + SMOOTH) / (union + SMOOTH)  
        ious.append(iou.mean().item())
    return np.mean(ious)

from tqdm import tqdm

def fit(model, dataloader, epochs=10, lr=3e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    model.to(device)
    hist = {'loss': [], 'iou': [], 'test_loss': [], 'test_iou': []}
    for epoch in range(1, epochs+1):
      bar = tqdm(dataloader['train'])
      train_loss, train_iou = [], []
      model.train()
      for imgs, masks in bar:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        y_hat = model(imgs)
        loss = criterion(y_hat, masks)
        loss.backward()
        optimizer.step()
        ious = IoU(y_hat, masks)
        train_loss.append(loss.item())
        train_iou.append(ious)
        bar.set_description(f"loss {np.mean(train_loss):.5f} iou {np.mean(train_iou):.5f}")
      hist['loss'].append(np.mean(train_loss))
      hist['iou'].append(np.mean(train_iou))
      bar = tqdm(dataloader['val'])
      test_loss, test_iou = [], []
      model.eval()
      with torch.no_grad():
        for imgs, masks in bar:
          imgs, masks = imgs.to(device), masks.to(device)
          y_hat = model(imgs)
          loss = criterion(y_hat, masks)
          ious = IoU(y_hat, masks)
          test_loss.append(loss.item())
          test_iou.append(ious)
          bar.set_description(f"test_loss {np.mean(test_loss):.5f} test_iou {np.mean(test_iou):.5f}")
      hist['test_loss'].append(np.mean(test_loss))
      hist['test_iou'].append(np.mean(test_iou))
      print(f"\nEpoch {epoch}/{epochs} loss {np.mean(train_loss):.5f} iou {np.mean(train_iou):.5f} test_loss {np.mean(test_loss):.5f} test_iou {np.mean(test_iou):.5f}")
    return hist
hist=fit(model, dataloader, epochs=100, lr=3e-4)

torch.save(model.state_dict(), 'model_weights.pt')