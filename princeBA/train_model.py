# -*- coding: utf-8 -*-

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
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import torch.optim as optim
import random
import cv2
from torchvision import models
from tqdm import tqdm
import argparse
import os
import subprocess
import pickle
import preprocessing as pp


parser = argparse.ArgumentParser()
parser.add_argument('--USE_CUDA', action='store_true', help='IF USE CUDA (Default == False)')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--n_epochs', type=int, default=50, help='Learning rate')
parser.add_argument('--experiment', default='facial_recog_eperiment', help='where to store samples and models')
parser.add_argument('--input_data_dir', type=str, default="../fer2013/", help="data location")
parser.add_argument('--model_type', type=str, default="resnet", help='Model type')
parser.add_argument('--data_augment', type=int, default=0, help='to augment data or not')
opt = parser.parse_args()
print(opt)

"""
save experiment
"""

os.system('mkdir experiments')
os.system('mkdir experiments/{0}'.format(opt.experiment))
#os.system('mkdir experiments/{0}/images'.format(opt.experiment))


def convert_to_numpy(data):
    X = data[:,1]
    X = np.asarray([np.asarray(X[i].split(" ")) for i in range(X.shape[0])])
    X = np.asarray([X[i].reshape(48,48).astype(int) for i in range(X.shape[0])])
    y = data[:,0]
    return (X,y)


def data_iter(x, y, batch_size):
    dataset_size = x.shape[0]
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while True:
        start += batch_size
        if start > dataset_size - batch_size:
            break
        batch_indices = order[start:start + batch_size]
        yield np.asarray([x[index] for index in batch_indices]) ,np.asarray([y[index] for index in batch_indices])

def early_stop(val_acc_history, t=2, required_progress=0.0000000000000000000005):
    cnt = 0 # initialize the count --> to store count of cases where difference in
                                    #  accuracy is less than required progress.

    if(len(val_acc_history) > 0): # if list has size > 0
        for i in range(t): # start the loop
            index = len(val_acc_history) - (i+1) # start from the last term in list and move to the left
            if (index >= 1): # to check if index != 0 --> else we can't compare to previous value
                if (abs(val_acc_history[index] - val_acc_history[index-1]) < required_progress):
                    cnt += 1 # increase the count value
                else:
                    break # break if difference is grea-ter

    if(cnt != t): # if count is equal to t, return True
        return False
    else:
        return True


class BKStart(nn.Module):
    """
    From Sang, Dat, Thuan paper (2017)
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8119447
    """
       
    def __init__(self, num_labels, dropout=0.1):
       
        super(BKStart, self).__init__()

        self.conv1 = nn.Sequential(
                            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=1),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=3, stride=2)
                            )
        self.conv2 = nn.Sequential(
                            nn.Conv2d(32, 32, kernel_size=4, stride=1, padding=1),
                            nn.ReLU(),
                            nn.AvgPool2d(kernel_size=3, stride=2)
                            )
        self.conv3 = nn.Sequential(
                            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1),
                            nn.ReLU(),
                            nn.AvgPool2d(kernel_size=3, stride = 2)
                            )
        self.FC = nn.Sequential(nn.Linear(64*3*3, 64),
                                nn.ReLU(),
                                nn.Dropout(dropout)
                               )
        self.out = nn.Linear(64, num_labels)
    
    def forward(self, x):
        x = x.unsqueeze(1) # (N,Ci,W,D)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1) 
        x = self.FC(x)
        x = self.out(x)
        return torch.nn.functional.softmax(x)

class BK12(nn.Module):
    """
    From Sang, Dat, Thuan paper (2017)
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8119447
    """
       
    def __init__(self, num_labels, dropout=0.1):
       
        super(BK12, self).__init__()

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
                            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                            nn.ReLU()
                            )
        self.conv1b = nn.Sequential(
                            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                            nn.ReLU()
                            )
        self.conv2 = nn.Sequential(
                            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                            nn.ReLU()
                            )
        self.conv2b = nn.Sequential(
                            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                            nn.ReLU()
                            )
        self.conv3 = nn.Sequential(
                            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                            nn.ReLU()
                            )
        self.conv3b = nn.Sequential(
                            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                            nn.ReLU()
                            )  
        self.conv4 = nn.Sequential(
                            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                            nn.ReLU()
                            )
        self.conv4b = nn.Sequential(
                            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                            nn.ReLU()
                            )
        self.conv4c = nn.Sequential(
                            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                            nn.ReLU()
                            )
        self.FC1 = nn.Sequential(nn.Linear(256*6*6, 256),
                                nn.ReLU(),
                                nn.Dropout(dropout))
        self.FC2 = nn.Sequential(nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Dropout(dropout))
        self.out = nn.Linear(256, num_labels)
    
    def forward(self, x):
        x = x.unsqueeze(1) # (N,Ci,W,D)
        x = self.MaxPool(self.conv1b(self.conv1(x)))
        x = self.MaxPool(self.conv2b(self.conv2(x)))
        x = self.MaxPool(self.conv3b(self.conv3(x)))
        x = self.conv4c(self.conv4b(self.conv4(x)))
        x = x.view(x.size(0), -1) 
        x = self.FC2(self.FC1(x))
        x = self.out(x)
        return torch.nn.functional.softmax(x)

def train(train_X, train_Y, valid_X, valid_Y, optimizer, model, batch_size, num_epochs, criterion, to_Add_Softmax=False, is_inception=False):
    losses = []
    total_batches = int(train_X.shape[0]/ batch_size)
    validation_losses = []

    eval_every = 0.5
    print_every = 0.5
    validate_every = int((eval_every/100)*total_batches)
    show_every = int((print_every/100)*total_batches)

    if opt.USE_CUDA:
        model = model.cuda()

    print("inside training")
    for epoch in range(1, num_epochs+1):
        stop_training = False
        train_data = data_iter(train_X, train_Y, batch_size)
        for i, (x,y) in enumerate(train_data):
            x = Variable(torch.from_numpy(x).type(torch.FloatTensor))
            y = Variable(torch.from_numpy(y).type(torch.LongTensor))

            if opt.USE_CUDA:
                x = x.cuda()
                y = y.cuda()

            model.train(True)
            optimizer.zero_grad()

            outputs = model(x)
            if is_inception == True:
                outputs = outputs[0]
            if to_Add_Softmax == True:
                outputs = nn.functional.softmax(outputs)
            loss = criterion(outputs, y)
            losses.append(loss.data[0])
            loss.backward()


            optimizer.step()

            if (i+1)%validate_every == 0:
                valid_loss_temp = []
                valid_data = data_iter(valid_X, valid_Y, 97)
                correct=0
                for j, (v_x, v_y) in enumerate(valid_data):
                    v_x = Variable(torch.from_numpy(v_x).type(torch.FloatTensor))
                    v_y = Variable(torch.from_numpy(v_y).type(torch.LongTensor))
                    if opt.USE_CUDA:
                        v_x = v_x.cuda()
                        v_y = v_y.cuda()
                    model.eval()
                    val_outputs = model(v_x)
                    eval_loss = criterion(val_outputs, v_y)
                    valid_loss_temp.append(eval_loss.data[0])
                    pred_vy = torch.max(val_outputs, 1)[1].data.cpu().numpy().squeeze()
                    correct = correct + sum(pred_vy == v_y.data.cpu().numpy())

                validation_losses.append(np.mean(valid_loss_temp))
                valid_acc = round(correct/len(valid_Y)*100, 2)
                stop_training = early_stop(validation_losses, 3)

            if stop_training:
                print("earily stop triggered")
                break
            if (i+1) % show_every == 0:
                print('Epoch: [{0}/{1}], Step: [{2}/{3}], Train loss: {4}, Validation loss:{5}, Valid Acc: {6}%'.format(
                           epoch, num_epochs, i+1, total_batches, np.sum(losses)/(total_batches*(epoch-1)+i), np.mean(np.array(validation_losses)),
                           valid_acc))
        if stop_training == True:
            break


data = pd.read_csv(opt.input_data_dir+"fer2013.csv")

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

train_d = data[data['Usage'] == 'Training']
train_X, train_Y = convert_to_numpy(train_d.values)
valid = data[data['Usage'] == 'PrivateTest']
valid_X, valid_Y = convert_to_numpy(valid.values)
test = data[data['Usage'] == 'PublicTest']
test_X, test_Y = convert_to_numpy(test.values)

num_labels = 7
num_epochs = opt.n_epochs
learning_rate = opt.learning_rate
kernel_size = 3
batch_size = 80

print("before training")

criterion = nn.CrossEntropyLoss()

#TO DELETE-----------------------------------------

def flip_horizontal(image):
    '''Flips picture horizontally with p=0.75 probability'''
    r = np.random.rand()
    if r > 0.25:
        new_image = image[:, ::-1]
    else:
        new_image = image
    return new_image

def recrop(image, w, h):
    '''Resizes an image to wxh, then takes a random 48x48 section'''
    im = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    #pick upper pixel -- integer from 0 to 4 (inclusive) 
    #pick left pixel -- integer from 0 to 4
    u = np.random.choice([0, 1, 2, 3, 4]) #bad code, but works if we set w=52 and h=52 
    l = np.random.choice([0, 1, 2, 3, 4]) 
    new = im[l:l+48, u:u+48]
    return new

if opt.data_augment == 1:
    print('Transforming Single-Channel Images')
    #Create transformed_set
    transformed_X = np.zeros(len(train_X)*48*48).reshape(len(train_X), 48, 48)
    for i in range(len(train_X)):
        flipped = np.array(flip_horizontal(train_X[i]), dtype=float)
        new_image = np.array(recrop(flipped, 52, 52), dtype=int)
        transformed_X[i] = new_image
    
    #Concatenate transformed images to training set
    #Concatenate labels of transformed images to training set labels
    train_X = np.vstack((train_X, transformed_X))
    train_Y = np.append(train_Y, train_Y)
#-------------------------------------------------

old_train_X = train_X.copy()
old_valid_X = valid_X.copy()
old_test_X = test_X.copy()
old_train_Y = train_Y.copy()

train_X, valid_X, test_X, train_Y, valid_Y, test_Y = pp.preprocess(old_train_X, old_valid_X, old_test_X,
                                                                   old_train_Y, valid_Y.copy(), test_Y.copy(),
                                                                   model = opt.model_type.lower(), transform = opt.data_augment)
del old_train_X
del old_valid_X
del old_test_X
del old_train_Y

print(train_X.shape)


print("Data Preprocessing Done")

def get_test_set_performance(test_X, test_Y, model):
    test_data = data_iter(test_X, test_Y, 97)
    correct=0
    predictions=[]
    actuals=[]
    for t, (t_x, t_y) in enumerate(test_data):
        t_x = Variable(torch.from_numpy(t_x).type(torch.FloatTensor))
        t_y = Variable(torch.from_numpy(t_y).type(torch.LongTensor))

        if opt.USE_CUDA:
            t_x = t_x.cuda()
            t_y = t_y.cuda()
        model.eval()
        test_output = model(t_x)
        pred_y = torch.max(test_output, 1)[1].data.cpu().numpy().squeeze()
        actual = t_y.data.cpu().numpy()
        correct = correct + sum(pred_y == actual)
        predictions.extend(pred_y)
        actuals.extend(actual)
    test_accuracy = correct/len(test_X)
    matrix = pd.DataFrame(confusion_matrix(actuals, predictions))
    return test_accuracy, matrix
    

if opt.model_type == "bk":
    BK = BKStart(num_labels, dropout=0.5)
    optimizer = torch.optim.Adam(BK.parameters(), lr=learning_rate)
    train(train_X, train_Y, valid_X, valid_Y, optimizer, BK, batch_size, num_epochs, criterion)
    BK.train(False)
    accuracy, matrix = get_test_set_performance(test_X, test_Y, BK)
    matrix.to_csv('experiments/{0}/confusion_matrix.csv'.format(opt.experiment))
    print(accuracy)
    print(matrix)

elif opt.model_type == "bk12":
    BK12model = BK12(num_labels, dropout=0.5)
    optimizer = torch.optim.Adam(BK12model.parameters(), lr=learning_rate)
    train(train_X, train_Y, valid_X, valid_Y, optimizer, BK12model, batch_size, num_epochs, criterion)
    BK12model.train(False)
    accuracy, matrix = get_test_set_performance(test_X, test_Y, BK12model)
    matrix.to_csv('experiments/{0}/confusion_matrix.csv'.format(opt.experiment))
    print(accuracy)
    print(matrix)

elif opt.model_type == "resnet":

    resnet = models.resnet50(pretrained=True)

    if opt.USE_CUDA:
        resnet = resnet.cuda()
    print("before resnet params")
    # freeze all model parameters
    for param in resnet.parameters():
        param.requires_grad = False

    # new final layer with 7 classes
    num_ftrs = resnet.fc.in_features
    resnet.fc = torch.nn.Linear(num_ftrs, num_labels)
    for param in resnet.fc.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(resnet.fc.parameters(), lr = learning_rate)

    print("before train")
    train(train_X, train_Y, valid_X, valid_Y, optimizer, resnet, batch_size, num_epochs, criterion, to_Add_Softmax=True, is_inception=False)

    #test_output = model(Variable(torch.from_numpy(test_X).type(torch.FloatTensor)))
    resnet.train(False)
    accuracy, matrix = get_test_set_performance(test_X, test_Y, resnet)
    matrix.to_csv('experiments/{0}/confusion_matrix.csv'.format(opt.experiment))
    print(accuracy)
    print(matrix)
    
elif opt.model_type == "alexnet":

    alexnet = models.alexnet(pretrained=True)

    if opt.USE_CUDA:
        alexnet = alexnet.cuda()

    # freeze all model parameters
    for param in alexnet.parameters():
        param.requires_grad = False

    # new final layer with 7 classes
    num_ftrs = alexnet.classifier[6].in_features
    alexnet.classifier._modules['6'] = nn.Linear(num_ftrs, num_labels)

    for param in alexnet.classifier[6].parameters():
        param.requires_grad = True
    optimizer = optim.Adam(alexnet.classifier[6].parameters(), lr=learning_rate)

    train(train_X, train_Y, valid_X, valid_Y, optimizer, alexnet, batch_size, num_epochs, criterion, to_Add_Softmax=True, is_inception=False)

    #test_output = model(Variable(torch.from_numpy(test_X).type(torch.FloatTensor)))
    alexnet.train(False)
    accuracy, matrix = get_test_set_performance(test_X, test_Y, alexnet)
    matrix.to_csv('experiments/{0}/confusion_matrix.csv'.format(opt.experiment))
    print(accuracy)
    print(matrix)

elif opt.model_type == "inception":

    incptn = models.inception_v3(pretrained=True)

    if opt.USE_CUDA:
        incptn = incptn.cuda()

    # freeze all model parameters
    for param in incptn.parameters():
        param.requires_grad = False

    # new final layer with 7 classes
    num_ftrs = incptn.fc.in_features
    incptn.fc = torch.nn.Linear(num_ftrs, num_labels)

    for param in incptn.fc.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(incptn.fc.parameters(), lr=learning_rate)
    train(train_X, train_Y, valid_X, valid_Y, optimizer, incptn, batch_size, num_epochs, criterion, to_Add_Softmax=True, is_inception=True)

    #test_output = model(Variable(torch.from_numpy(test_X).type(torch.FloatTensor)))
    incptn.train(False)
    accuracy, matrix = get_test_set_performance(test_X, test_Y, incptn)
    matrix.to_csv('experiments/{0}/confusion_matrix.csv'.format(opt.experiment))
    print(accuracy)
    print(matrix)
