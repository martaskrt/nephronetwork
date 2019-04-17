# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 16:45:18 2019

@author: larun
"""
####
###         IMPORT LIBRARIES
####

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import torch.optim as optim
import os
import pandas as pd
import argparse
import time

####
###         HELPER FUNCTIONS
####    

    ##
    ##         Data prep
    ##    


# loads all us triad seq paths
def get_pickle_files(file_dir):
    pickle_files = []
    for subdir, dirs, files in os.walk(file_dir):
        for file in files:
            if file.lower()[-7:] == ".pickle":
                pickle_files.append(os.path.join(subdir, file))
    return pickle_files    

# loads all triad seqs, converts them to a torch tensor 
def get_training_data(opt,split_batches = True,valid_share = 0.1):
    pickle_files = get_pickle_files(opt.train_dir)  
    
    training_stack = []
    label_stack = []
    for file in pickle_files:
        in_df = pd.read_pickle(file)
        training_stack.extend(in_df.triad)
        label_stack.extend(in_df.label)
    
    ts_np = np.array(training_stack)
    tl_np = np.array(label_stack)
    
    n_samples = ts_np.shape[0]
    n_val_samples = round(n_samples*0.1)    
    
    perm = np.random.permutation(n_samples)
    
    ts_perm = ts_np[perm,:,:,:]
    tl_perm = tl_np[perm]
    
    ts_val = ts_perm[:n_val_samples,:,:,:]
    tl_val = ts_perm[0:n_val_samples]

    ts_train = ts_perm[(n_val_samples+1):,:,:,:]    
    tl_train = tl_perm[(n_val_samples+1):]    
    
    if split_batches:
        ts_batch = np.empty((0,opt.batch_size,ts_train.shape[1],ts_train.shape[2],ts_train.shape[3]))
        tl_batch = np.empty((0,opt.batch_size))
        for k in range((n_samples - n_val_samples)//opt.batch_size):
            ts_batch = np.append(ts_batch,[ts_train[k:(k+opt.batch_size),:,:,:]],axis=0)
            tl_batch = np.append(tl_batch,[tl_train[k:(k+opt.batch_size)]],axis=0)
    else:
        ts_batch = ts_perm
        tl_batch = tl_perm
    
    
    training_data = torch.from_numpy(ts_batch)
    training_label = torch.from_numpy(tl_batch)

    validation_data = torch.from_numpy(ts_val)    
    validation_label = torch.from_numpy(tl_val)
    
    if opt.gpu: 
        training_data = training_data.cuda()
        training_label = training_label.cuda()
    
        validation_data = validation_data.cuda()
        validation_label = validation_label.cuda()
    
    return training_data, training_label, validation_data, validation_label

    ##
    ##        Model and training functions
    ##    

## CNN MODEL CLASS
class cnn(nn.Module):
    def __init__(self,opt):
        super(cnn,self).__init__()
        
        ### input data shape: batch, channels, img width, img height
        self.batch_size = opt.batch_size
        
        self.downconv1 = nn.Sequential(
                nn.Conv2d(opt.num_in_channels,opt.num_filters,opt.kernel_size,padding=2),
                nn.BatchNorm2d(opt.num_filters),
                nn.ReLU(),
                nn.MaxPool2d(2)) 
        self.downconv2 = nn.Sequential(
                nn.Conv2d(opt.num_filters,opt.num_filters*2,opt.kernel_size,padding=2),
                nn.BatchNorm2d(opt.num_filters*2),
                nn.ReLU(),
                nn.MaxPool2d(2))
        self.downconv3 = nn.Sequential(
                nn.Conv2d(opt.num_filters*2,opt.num_filters*4,opt.kernel_size,padding=2),
                nn.BatchNorm2d(opt.num_filters*4),
                nn.ReLU(),
                nn.MaxPool2d(2))
        self.linear1 = nn.Sequential(
                nn.Linear(32*opt.num_filters*4,32*opt.num_filters),
                nn.ReLU())
        self.linear2 = nn.Sequential(
                nn.Linear(32*opt.num_filters,32),
                nn.ReLU())
        self.linear3 = nn.Sequential(
                nn.Linear(32*opt.num_filters,1),
                nn.Softmax())        

    def forward(self,x):        
        self.out1 = self.downconv1(x)
        self.out2 = self.downconv2(self.out1)
        self.out3 = self.downconv3(self.out2)
        
        self.flat_out3 = self.out3.reshape(self.batch_size,-1,1,1)
        self.flat_out3 = torch.squeeze(self.flat_out3)
        
        self.out4 = self.linear1(self.flat_out3)
        self.out5 = self.linear2(self.out4)
        self.out_final = self.linear3(self.out5)
        
        return self.out_final

## SIAMESE CNN MODEL CLASS
class siamese_cnn(nn.Module):
    def __init__(self,opt):
        super(siamese_cnn,self).__init__()
        
        ### input data shape: batch, channels, img width, img height
        self.batch_size = opt.batch_size
        self.img_dim = opt.input_dim
        
        self.downconv1 = nn.Sequential(
                nn.Conv2d(opt.num_in_channels,opt.num_filters,opt.kernel_size,padding=2),
                nn.BatchNorm2d(opt.num_filters),
                nn.ReLU(),
                nn.MaxPool2d(2)) ## 128
        self.downconv2 = nn.Sequential(
                nn.Conv2d(opt.num_filters,opt.num_filters*2,opt.kernel_size,padding=2),
                nn.BatchNorm2d(opt.num_filters*2),
                nn.ReLU(),
                nn.MaxPool2d(2)) ## 64
        self.downconv3 = nn.Sequential(
                nn.Conv2d(opt.num_filters*2,opt.num_filters*4,opt.kernel_size,padding=2),
                nn.BatchNorm2d(opt.num_filters*4),
                nn.ReLU(),
                nn.MaxPool2d(2)) ## batch, 32, 
        
        in_dim = int((self.img_dim/8)*(self.img_dim/8)*opt.num_filters*4)
        self.linear1 = nn.Sequential( ## add batch norm here? 
                nn.Linear(in_dim,16*opt.num_filters), ## change these dimensions
                nn.ReLU())
        self.linear2 = nn.Sequential(
                nn.Linear(3*16*opt.num_filters,4*opt.num_filters),
                nn.ReLU())
        self.linear3 = nn.Sequential(
                nn.Linear(4*opt.num_filters,2),
                nn.Softmax())        

    def forward(self,data):
        
        for k in range(3):
            # print(self.batch_size)
            # input("That's your batch size. Now click enter")
            
            # print(data.shape)
            # input("Waiting...")
            
            x = data[:,k,:,:]
            x = torch.unsqueeze(x,1)
            # print(x.shape)
            # input("Waiting...")
            
            self.out1 = self.downconv1(x)
            self.out2 = self.downconv2(self.out1)
            self.out3 = self.downconv3(self.out2)
            
            #print("self.out3.shape: " + str(self.out3.shape))
            # print(self.out3.shape)
            # input("Waiting...")
            #flat_out3 = self.out3.view(-1,opt.kernel_size*opt.kernel_size*opt.num_filters*4)
            linear_out_dim = int((self.img_dim/8)*(self.img_dim/8)*opt.num_filters*4)
            flat_out3 = self.out3.reshape(self.batch_size,linear_out_dim,1,1)
            flat_out3 = torch.squeeze(flat_out3)
            
            out4 = self.linear1(flat_out3)
            # print(out4.shape)
            # input("out4 shape. Press enter.")
            
            if k == 0:                 
                combined_output = out4 
            else:
                combined_output = torch.cat((combined_output,out4),1)

            # print(combined_output.shape)
            # input("combined_output shape. Press enter.")

            
        out5 = self.linear2(combined_output)    
        out6 = self.linear3(out5)
        
        return out6
        
        
        
def train_model(training_data,training_label,valid_data,valid_labels,opt):
    start = time.time()
    n_batches = training_data.shape[0]    
    criterion = nn.CrossEntropyLoss()
    
    train_loss_list = []
    valid_loss_list = []
    
    if opt.siamese: ## 1 channel NN iterated 3 times 
        cnn_mod = siamese_cnn(opt)
        optimizer = torch.optim.Adam(cnn_mod.parameters(), lr=opt.learning_rate)
        for epoch in range(opt.epochs):

            cnn_mod.train()
            
            for  batch in range(n_batches):
                ## select batch input
                inputs = training_data[batch].float()
                labels = training_labels[batch].type(torch.LongTensor)
                
                ## forward pass
                output = cnn_mod(inputs)               
                loss = criterion(output,labels)
                train_loss_list.append(loss.item())
               
                ## backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                    
            ## evaluate model in validation set 
            cnn_mod.eval()
            val_loss = criterion(output,labels)
            valid_loss_list.append(val_loss)

            time_elapsed = time.time() - start        
            print('Epoch [%d/%d], Train Loss: %.4f, Val loss: %.1f, Time(s): %d' % (
                    epoch+1, opt.epochs, loss, val_loss, time_elapsed))
        
    #else: ## 3 channel NN

    return cnn_mod
    
##
##        MAIN FUNCTION 
##    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_dir', default='/home/lauren/hydronephrosis/triad_model/training_data/', help="input directory of training triads")
    parser.add_argument('-test_dir', default='/home/lauren/hydronephrosis/triad_model/test_data/', help="input directory of test triads")
    parser.add_argument('-batch_size', default=32, help="Batch size")
    parser.add_argument('-learning_rate', default=0.0005, help="Learning rate")
    parser.add_argument('-siamese', default=True, help="Train siamese model vs not")
    parser.add_argument('-gpu', default=True, help="Train model using a gpu=True or no=False")
    parser.add_argument('-num_in_channels', default=1, help="Number of channels input, must be 1 if siamese, 3 if not siamese")
    parser.add_argument('-num_filters', default=6, help="Number of filters per convolutional layer")
    parser.add_argument('-kernel_size', default=5, help="Kernel size")
    parser.add_argument('-epochs', default=3, help="Kernel size")
    parser.add_argument('-input_dim', default=256, help="Kernel size")

    training_data, training_labels, val_data, val_labels = get_training_data(opt)
    out_mod = train_model(training_data, training_labels, val_data, val_labels,opt)
    ## find way to save out_mod

if __name__ == "__main__":
    main()
    
## 
##      DEBUGGING MAIN FUNCTION 
##

# =============================================================================
# class options(object):
#     pass
# # 
# =============================================================================
# =============================================================================
# opt = options()
# opt.train_dir = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Image-analysis/triad/training-dir/'
# opt.test_dir = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Image-analysis/triad/test-dir/'
# opt.batch_size = 32
# opt.learning_rate = 0.0005
# opt.siamese = True
# opt.gpu = False
# opt.num_in_channels = 1
# opt.num_filters = 6
# opt.kernel_size = 5
# opt.epochs = 3
# opt.input_dim = 256
# 
# training_data, training_labels, val_data, val_labels = get_training_data(opt)
# out_mod = train_model(training_data, training_labels, val_data, val_labels,opt)
# 
# =============================================================================
