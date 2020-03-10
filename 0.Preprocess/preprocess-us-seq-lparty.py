# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 15:22:43 2019

@author: Lauren
"""


import os
import pydicom
from skimage.color import rgb2gray
# from skimage import exposure
from skimage import img_as_float
# from skimage import transform
import scipy.misc
#import matplotlib.pyplot as plt
import argparse
# import extract_labels
# import pandas as pd
# import numpy as np
# from PIL import Image


#rootdir = os.path.join(os.path.abspath(os.path.join('./', os.pardir)), 'all-images')
#print(rootdir)
#rootdir = '/home/martaskreta/Desktop/CSC2516/all-images/'

# loads all image paths to array
def load_file_names(cab_dir):
    dcm_files = []
    for subdir, dirs, files in os.walk(cab_dir):
        for file in files:
           if file.lower()[-4:] == ".dcm":
               dcm_files.append(os.path.join(subdir, file))
    return sorted(dcm_files)

def load_images(dcm_files, opt):
    #train_rows = round(label_data.shape[0]*0.75)
    #print("train rows " + train_rows)
    imgs=[]
    img_num = 0
    img_counter = []
    img_creation_time = []
    for image_path in dcm_files:
        tokens = image_path.split("/")
        print(tokens)
        
        rootdir_tokens = opt.rootdir.split("/")
        print(rootdir_tokens)
        
        # get mrn and sample number
        print(tokens[len(rootdir_tokens)-1])
        sample_name = tokens[len(rootdir_tokens)-1][1:].split("_")
        print(sample_name)
        mrn = sample_name[0]
        print(mrn)
        
        float_mrn = float(mrn)
        print(float_mrn)
        
        sample_id = float_mrn
        print("sample id: " + str(int(sample_id)))

        try:
            ds = pydicom.dcmread(image_path)
            print("Image read in from dicom")

        except:
            print("IMAGE PATH ERROR")
            print(image_path)

        try:
            inst_num = ds.InstanceNumber
            img_creation_time.append(inst_num)
        except:
            print("Error grabbing instance number")

        img = ds.pixel_array
        print("Image transferred to pixel array")
        img = img_as_float(rgb2gray(img))


        img_file_name = str(int(sample_id)) + "_" + sample_name[1] + "_" + str(inst_num) + ".jpg"
        print("Image file name:" + img_file_name)
        scipy.misc.imsave(os.path.join(opt.jpg_dump_dir,img_file_name), img)
        print('Image file written to' + opt.jpg_dump_dir + img_file_name)

        img_num = img_num + 1

        print("Error processing image array and writing file")

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-params', default=1, help="parameter settings to crop images "
    #                                                "(see IMAGE_PARAMS_1 at top of file)")
    # parser.add_argument('-view', default=0, help="number of images to visualize")
    # parser.add_argument('-output_dim', default=256, help="dimension of output image")
    parser.add_argument('-rootdir', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/test-cabs/', 
                        help="directory of US sequence dicoms")
    parser.add_argument('-dcm_dir', default='D5048003_1', 
                        help="directory of US sequence dicoms")
    parser.add_argument('-jpg_dump_dir', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/all-jpgs-test/',
                        help="directory of US sequence dicoms")
    # parser.add_argument('-out_dir', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/train-us-seqs/',
    #                     help="directory of US sequence dicoms")

    opt = parser.parse_args()
    opt.view = int(opt.view)
    opt.output_dim = int(opt.output_dim)

    cab_dir = os.path.join(opt.rootdir,opt.dcm_dir)
    print("cab_dir: " + cab_dir)

    dcm_files = load_file_names(cab_dir = cab_dir)
    #print(".dcm files: ")
    #print(dcm_files)
    
    # data = load_labels()
    #print("data: " )
    #print(data)
    load_images(dcm_files, opt)


if __name__ == "__main__":
    main()





