# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 18:20:45 2019

@author: larun
"""

####
###         IMPORT LIBRARIES
####

import numpy as np
import pandas as pd
from skimage import exposure
import os
import argparse
#import matplotlib.pyplot as plt
import random

####
###         HELPER FUNCTIONS
####    

# loads all us array paths
def get_pickle_files(file_dir):
    pickle_files = []
    for subdir, dirs, files in os.walk(file_dir):
        for file in files:
            if file.lower()[-7:] == ".pickle":
                pickle_files.append(os.path.join(subdir, file))
    return pickle_files    

# load us seq's with specified contrast
def load_us_seq(pickle_file,opt):
    us_seq = pd.read_pickle(pickle_file)    
    
    preproc_us = []
    for i in range(len(us_seq['images'])):
        img = us_seq['images'].iloc[i]
        preproc_us.append(set_contrast(img,opt))
    
    return preproc_us

# function to set the image contrast
def set_contrast(image,opt):
    if opt.contrast == 0:
        out_img = image
    elif opt.contrast == 1:
        out_img = exposure.equalize_hist(image)
    elif opt.contrast == 2:
        out_img = exposure.equalize_adapthist(image)
    elif opt.contrast == 3:
        out_img = exposure.rescale_intensity(image)   
        
    return out_img
    
# create an array of positive triad samples and labels
def make_pos_samples(us_seq):
    pos_triads = []
    pos_triad_labels = []
    
    for k in range(len(us_seq) - 2):
        pos_triads.append([us_seq[k], us_seq[k+1], us_seq[k+2]])
        pos_triad_labels.append(1)
        pos_triads.append([us_seq[k+2], us_seq[k+1], us_seq[k]])
        pos_triad_labels.append(1)
    
    return pos_triads,pos_triad_labels

# create an array of negative triad samples and labels
def make_neg_samples(us_seq):
    neg_triads = []
    neg_triad_labels = []
    
    for k in range(len(us_seq) - 2):
        neg_triads.append([us_seq[k], us_seq[k+2], us_seq[k+1]])
        neg_triad_labels.append(0)
        neg_triads.append([us_seq[k+1], us_seq[k+2], us_seq[k]])
        neg_triad_labels.append(0)
        neg_triads.append([us_seq[k+1], us_seq[k], us_seq[k+2]])
        neg_triad_labels.append(0)
        neg_triads.append([us_seq[k+1], us_seq[k+2], us_seq[k]])
        neg_triad_labels.append(0)
        
    neg_triads = random.sample(neg_triads,k=(2*(len(us_seq)-2)))
    neg_triad_labels = random.sample(neg_triad_labels,k=(2*(len(us_seq)-2)))
    
    return neg_triads,neg_triad_labels

# get pandas df of an equal number of pos and neg training examples
def get_training_samples(us_seq):
    
    pos_triads,pos_labels = make_pos_samples(us_seq)
    neg_triads,neg_labels = make_neg_samples(us_seq)
    
    training_triads = pos_triads + neg_triads
    triad_labels = pos_labels + neg_labels    
    
    pre_df = {'triad' : training_triads, 'label' : triad_labels}
    training_df = pd.DataFrame(pre_df)
    
    return training_df

def train_test_assg(n_us_seqs,test_prob = 0.2):
    return np.random.choice(["train","test"],n_us_seqs,p=[1-test_prob,test_prob])
    
## get out file name based on in file name for pandas df using pickle 
def get_out_file_name(in_file,group="train"):
    if group=="train":
        out_file_name = in_file.split("/")[-1][:-7] + "_us_seq_training.pickle"
    elif group=="test":
        out_file_name = in_file.split("/")[-1][:-7] + "_us_seq_test.pickle"
    
    return out_file_name

## function by Marta Skreta to image with different contrast settings
# =============================================================================
# def view_sample_images(img):
#     plt.figure()
#     plt.subplot(2, 2, 1)
#     plt.imshow(img, cmap='gray')
#     rescaled_cropped_image = exposure.equalize_hist(img)
#     plt.subplot(2, 2, 2)
#     plt.imshow(rescaled_cropped_image, cmap='gray')
#     rescaled_cropped_image = exposure.equalize_adapthist(img)
#     plt.subplot(2, 2, 3)
#     plt.imshow(rescaled_cropped_image, cmap='gray')
#     rescaled_cropped_image = exposure.rescale_intensity(img)
#     plt.subplot(2, 2, 4)
#     plt.imshow(rescaled_cropped_image, cmap='gray')
#     plt.show()
# =============================================================================
 
# =============================================================================
# def plot_triads(triad):
#     plt.figure()
#     plt.subplot(1,3,1)
#     plt.imshow(triad[0],cmap='gray')
#     plt.subplot(1,3,2)
#     plt.imshow(triad[1],cmap='gray')
#     plt.subplot(1,3,3)
#     plt.imshow(triad[2],cmap='gray')
#     plt.show()
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_dir', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/train-us-seqs/', help="directory of ultrasound sequences")
    parser.add_argument('-train_dir', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/triad_model/training_samples/', help="output directory of training triads")
    parser.add_argument('-test_dir', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/triad_model/test_samples/', help="output directory of test triads")
    parser.add_argument('-contrast', default=2, help="0 = original image, 1 = exposure.equalize_hist(image), 2 = exposure.equalize_adapthist(image), 3 = exposure.rescale_intensity(image)")

    opt = parser.parse_args()
    
    pickle_files = get_pickle_files(opt.in_dir)
    
    train_test_split = train_test_assg(len(pickle_files))

    i = 0    
    for file in pickle_files:
        train_vs_test = train_test_split[i]        
        us_seq = load_us_seq(file,opt)
        
        try:
            training_df = get_training_samples(us_seq)             
            if train_vs_test == "train":
                out_file = os.path.join(opt.train_dir,get_out_file_name(file,group = "train"))
                #os.makedirs(os.path.dirname(out_file), exist_ok=True)
                training_df.to_pickle(out_file)
            elif train_vs_test == "test":
                out_file = os.path.join(opt.test_dir,get_out_file_name(file,group = "test"))
                #os.makedirs(os.path.dirname(out_file), exist_ok=True)
                training_df.to_pickle(out_file)
        except:
            print("Error getting training samples for: " + file)
        i=i+1

if __name__ == "__main__":
    main()

## DEBUG MAIN_FUNCTION
# =============================================================================
# class options(object):
#     pass
# 
# opt = options()
# opt.in_dir = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Image-analysis/test-dir/'
# opt.contrast = 0
# 
# pickle_files = get_pickle_files(opt.in_dir)
# 
# file = pickle_files[0]
# #for file in pickle_files:
# us_seq = load_us_seq(file,opt)
# 
# get_out_file_name(file)
# 
# for k in range(len(us_seq)):
#     view_sample_images(us_seq[k])
# 
# training_df = get_training_samples(us_seq)
# 
# training_df.label[200]
# plot_triads(training_df.triad[200])
# =============================================================================

# =============================================================================
#class options(object):
#    pass
## 
#opt = options()
#opt.in_dir = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Image-analysis/test-dir/'
#opt.contrast = 2
#opt.train_dir = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Image-analysis/triad/training-dir/'
#opt.test_dir = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Image-analysis/triad/test-dir/'
## 
#pickle_files = get_pickle_files(opt.in_dir)
## 
#train_test_split = train_test_assg(len(pickle_files))
# 
# i = 0    
# for file in pickle_files:
#     train_vs_test = train_test_split[i]        
#     us_seq = load_us_seq(file,opt)
#     training_df = get_training_samples(us_seq) 
#     if train_vs_test == "train":
#         out_file = os.path.join(opt.train_dir,get_out_file_name(file,group = "train"))
#         os.makedirs(os.path.dirname(out_file), exist_ok=True)
#         training_df.to_pickle(out_file)
#     elif train_vs_test == "test":
#         out_file = os.path.join(opt.test_dir,get_out_file_name(file,group = "test"))
#         os.makedirs(os.path.dirname(out_file), exist_ok=True)
#         training_df.to_pickle(out_file)
#     i=i+1
# 
# =============================================================================
