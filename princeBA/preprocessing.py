import numpy as np
from math import ceil
import matplotlib.pyplot as plt
import sys
import re
import pandas as pd
import torch
import torch.nn as nn
import scipy
from scipy import ndimage
import new_transforms
from torchvision import transforms
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
#from jupyterthemes import jtplot
import random
import cv2
from torchvision import models
from tqdm import tqdm

def convert_to_numpy(data):
    X = data[:,1]
    X = np.asarray([np.asarray(X[i].split(" ")) for i in range(X.shape[0])])
    X = np.asarray([X[i].reshape(48,48).astype(int) for i in range(X.shape[0])])
    y = data[:,0]
    return (X,y)

def normalize(image):
    '''Normalizes an image channel so that the mean pixel is 0 and
    standard deviation is 1'''
    new_image = (image - image.mean())/image.std()
    return new_image

def transform_single_channel_image(image):
    trans = transforms.Compose([transforms.ToPILImage(),
                            transforms.RandomHorizontalFlip(),
                            new_transforms.RandomResizedCrop(48),
                            transforms.ToTensor()
                           ])
    newIm = trans(image)[0].numpy()
    finalIm = normalize(newIm)
    return finalIm


def to_rgb1a(data, w, h):
    '''This function takes a single-channel image and converts it to a
    3-channel image'''
    R, _, _ = data.shape
    temp = np.zeros((R, 3, w, h))
    
    for i in tqdm(range(R)):
        im = data[0]
        ret = np.empty((3, w, h))
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        ret[0, :, :] =  ret[1, :, :] =  ret[2, :, :] =  im
        temp[i] = ret
        data = np.delete(data, 0, 0)
    return temp

def flip_horizontal(image):
    '''Flips picture horizontally with p=0.75 probability'''
    r = np.random.rand()
    if r > 0.25:
        new_image = image[:, ::-1]
    else:
        new_image = image
    return new_image

def recrop(image, w, h):
    '''Resizes an image to 52x52, then takes a random 48x48 section'''
    im = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    #pick upper pixel -- integer from 0 to 4 (inclusive) 
    #pick left pixel -- integer from 0 to 4
    u = np.random.choice([0, 1, 2, 3, 4]) 
    l = np.random.choice([0, 1, 2, 3, 4]) 
    new = im[l:l+48, u:u+48]
    return new

#---------------------------------------------------

#Download data in advance

def preprocess(train_X, valid_X, test_X, train_Y, valid_Y, test_Y, model, transform):
    '''
    @model: can be BK, BK12, ResNet, AlexNet, Inception, or LR
    @transform: can be 0 or 1
    @outputs: train_X, valid_X, test_X, train_Y, valid_Y, test_Y
    '''
    
    if transform == 1:
        print('Transforming Single-Channel Images')
        #Create transformed_set
        transformed_X = np.zeros(len(train_X)*48*48).reshape(len(train_X), 48, 48)
        for i in range(len(train_X)):
            flipped = np.array(flip_horizontal(train_X[i]), dtype=float)
            new_image = np.array(recrop(flipped, 52, 52), dtype=int)
            transformed_X[i] = new_image
    
    if transform == 1:
        #Concatenate transformed images to training set
        #Concatenate labels of transformed images to training set labels
        train_X = np.vstack((train_X, transformed_X))
        train_Y = np.append(train_Y, train_Y)

    if model in ['alexnet', 'resnet', 'inception', 'lreg']:
        sizes = {'alexnet': 227, 'inception':299, 'resnet':224 }
        size = sizes[model]
        print('Converting 1-Channel images to 3-channel images, then Renormalizing')
        #Convert images to 3-channel format
        train_X_single = train_X.copy()
        valid_X_single = valid_X.copy()
        test_X_single = test_X.copy()

        train_X = np.zeros(len(train_X_single)*3*size*size).reshape(len(train_X_single), 3, size, size)
        valid_X = np.zeros(len(valid_X_single)*3*size*size).reshape(len(valid_X_single), 3, size, size)
        test_X = np.zeros(len(test_X_single)*3*size*size).reshape(len(test_X_single), 3, size, size)

        batches = round(len(train_X_single)/100)+1
        vbatches = round(len(valid_X_single)/100)+1
        tbatches = round(len(test_X_single)/100)+1

        print('Converting Training Set')
        for b in range(batches):
            subset = train_X_single[b*100:b*100+100]
            subset3D = to_rgb1a(subset, size, size)
            train_X[b*100:b*100+100, :, :, :] = subset3D

        print('Converting Validation Set')
        for v in range(vbatches):
            subset = valid_X_single[v*100:v*100+100]
            subset3D = to_rgb1a(subset, size, size)
            valid_X[v*100:v*100+100, :, :, :] = subset3D

        print('Converting Test Set')
        for t in range(tbatches):
            subset = test_X_single[t*100:t*100+100]
            subset3D = to_rgb1a(subset, size, size)
            test_X[t*100:t*100+100, :, :, :] = subset3D
        
        del train_X_single
        del valid_X_single
        del test_X_single
        
        #train_X = to_rgb1a(train_X_single, size, size)
        #valid_X = to_rgb1a(valid_X_single, size, size)
        #test_X = to_rgb1a(test_X_single, size, size)

    return train_X, valid_X, test_X, train_Y, valid_Y, test_Y
