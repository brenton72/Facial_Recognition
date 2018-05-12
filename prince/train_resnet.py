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


parser = argparse.ArgumentParser()
parser.add_argument('--USE_CUDA', action='store_true', help='IF USE CUDA (Default == False)')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--n_epochs', type=int, default=50, help='Learning rate')
parser.add_argument('--experiment', default='facial_recog_eperiment', help='where to store samples and models')
parser.add_argument('--input_data_dir', type=str, default="/scratch/sb3923/Facial_Recognition/fer2013/", help="data location")
parser.add_argument('--model_type', type=str, default="resnet", help='Model type')
parser.add_argument('--data_augment', type=int, default=0, help='to agment data or not')
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

def early_stop(val_acc_history, t=2, required_progress=0.001):
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




def train(train_X, train_Y, valid_X, valid_Y, optimizer, model, batch_size, num_epochs, criterion, to_Add_Softmax=False, is_inception=False):
    losses = []
    total_batches = int(train_X.shape[0]/ batch_size)
    validation_losses = []

    eval_every = 10
    print_every = 10
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
                valid_data = data_iter(valid_X, valid_Y, batch_size)
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
                validation_losses.append(np.mean(valid_loss_temp))
                stop_training = early_stop(validation_losses, 3)

            if stop_training:
                print("earily stop triggered")
                break
            if (i+1) % show_every == 0:
                print('Epoch: [{0}/{1}], Step: [{2}/{3}], Train loss: {4}, Validation loss:{5}'.format(
                           epoch, num_epochs, i+1, total_batches, np.mean(losses)/(total_batches*epoch), np.mean(np.array(validation_losses))))
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

def to_rgb1a(data, w, h):

    R, _, _ = data.shape
    temp = np.zeros((R, 3, w, h))

    for i in tqdm(range(R)):
        im = data[0]
        ret = np.empty((3, w, h), dtype=np.uint8)
        im = cv2.resize(im.astype(float), (w, h), interpolation=cv2.INTER_LINEAR)
        ret[0, :, :] =  ret[1, :, :] =  ret[2, :, :] =  im
        temp[i] = ret
        data = np.delete(data, 0, 0)

    return temp

print("transform")

if opt.model_type == "resnet":
    old_train_X = train_X
    train_X = to_rgb1a(old_train_X, 227, 227)
    del old_train_X
    old_valid_X = valid_X
    valid_X = to_rgb1a(old_valid_X, 227, 227)
    del old_valid_X
    old_test_X = test_X
    test_X = to_rgb1a(old_test_X, 227, 227)
    del old_test_X


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

    optimizer = optim.Adam(resnet.fc.parameters(), lr=0.0001)
    num_epochs = 1

    print("before train")
    train(train_X, train_Y, valid_X, valid_Y, optimizer, resnet, batch_size, num_epochs, criterion, to_Add_Softmax=True, is_inception=False)

    #test_output = model(Variable(torch.from_numpy(test_X).type(torch.FloatTensor)))
    resnet.train(False)
    test_in = Variable(torch.from_numpy(test_X).type(torch.FloatTensor))
    if opt.USE_CUDA:
        test_in = test_in.cuda()
    test_output = resnet(test_in)
    pred_y = torch.max(test_output, 1)[1].data.cpu().numpy().squeeze()
    accuracy = sum(pred_y == test_Y)/len(test_Y)
    print(accuracy)
    print(confusion_matrix(test_Y.astype(int), pred_y))

elif opt.model_type == "alexnet":
    old_train_X = train_X
    train_X = to_rgb1a(old_train_X, 227, 227)
    del old_train_X
    old_valid_X = valid_X
    valid_X = to_rgb1a(old_valid_X, 227, 227)
    del old_valid_X
    old_test_X = test_X
    test_X = to_rgb1a(old_test_X, 227, 227)
    del old_test_X


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
    optimizer = optim.Adam(alexnet.classifier[6].parameters(), lr=0.0001)

    train(train_X, train_Y, valid_X, valid_Y, optimizer, alexnet, batch_size, num_epochs, criterion, to_Add_Softmax=True, is_inception=False)

    #test_output = model(Variable(torch.from_numpy(test_X).type(torch.FloatTensor)))
    alexnet.train(False)
    test_in = Variable(torch.from_numpy(test_X).type(torch.FloatTensor))
    if opt.USE_CUDA:
        test_in = test_in.cuda()
    test_output = alexnet(test_in)
    pred_y = torch.max(test_output, 1)[1].data.cpu().numpy().squeeze()
    accuracy = sum(pred_y == test_Y)/len(test_Y)
    print(accuracy)
    print(confusion_matrix(test_Y.astype(int), pred_y))

elif opt.model_type == "inception":
    old_train_X = train_X
    train_X = to_rgb1a(old_train_X, 299, 299)
    del old_train_X
    old_valid_X = valid_X
    valid_X = to_rgb1a(old_valid_X, 299, 299)
    del old_valid_X
    old_test_X = test_X
    test_X = to_rgb1a(old_test_X, 299, 299)
    del old_test_X


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

    optimizer = optim.Adam(incptn.fc.parameters(), lr=0.0001)
    train(train_X, train_Y, valid_X, valid_Y, optimizer, incptn, batch_size, num_epochs, criterion, to_Add_Softmax=True, is_inception=True)

    #test_output = model(Variable(torch.from_numpy(test_X).type(torch.FloatTensor)))
    incptn.train(False)
    test_in = Variable(torch.from_numpy(test_X).type(torch.FloatTensor))
    if opt.USE_CUDA:
        test_in = test_in.cuda()
    test_output = incptn(test_in)
    pred_y = torch.max(test_output, 1)[1].data.cpu().numpy().squeeze()
    accuracy = sum(pred_y == test_Y)/len(test_Y)
    print(accuracy)
    print(confusion_matrix(test_Y.astype(int), pred_y))
