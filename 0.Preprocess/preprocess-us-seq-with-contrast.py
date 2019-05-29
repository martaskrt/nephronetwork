# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 15:22:43 2019

@author: Lauren
"""


import os
import pydicom
from skimage.color import rgb2gray
from skimage import exposure
from skimage import img_as_float
from skimage import transform
import scipy.misc
#import matplotlib.pyplot as plt
import argparse
import extract_labels
import pandas as pd
import numpy as np
from PIL import Image

IMAGE_PARAMS = {0: [2.1, 4, 4],
                1: [2.5, 3.2, 3.2]}  # factor to crop images by: crop ratio, translate_x, translate_y

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
    return dcm_files

# load labels and study ids into pandas dataframe
def load_labels(file_with_mrns="/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/label_files/samples_with_studyids_and_mrns.csv"):
    data = pd.read_csv(file_with_mrns)
    return data

# CROP IMAGES #
def crop_image(image, param_num):
    width = image.shape[1]
    height = image.shape[0]

    params = IMAGE_PARAMS[int(param_num)]

    new_dim = int(width // params[0])  # final image will be square of dim width/2 * width/2

    start_col = int(width // params[1])  # column (width position) to start from
    start_row = int(height // params[2])  # row (height position) to start from

    cropped_image = image[start_row:start_row+new_dim, start_col:start_col+new_dim]
    return cropped_image


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

def load_images(label_data, dcm_files, opt):
    #train_rows = round(label_data.shape[0]*0.75)
    #print("train rows " + train_rows)
    label_data = label_data.sort_values('MRN')
    top_mrns = int(round(label_data.shape[0]*0.75))
    label_data = label_data.head(top_mrns)
    
    #print(label_data.MRN)
    print(label_data.columns)

    imgs=[]
    img_num = 0
    img_counter = []
    for image_path in dcm_files:
        tokens = image_path.split("/")
        #print(tokens)

        # get mrn and sample number
        sample_name = tokens[8][1:].split("_")
        #print(sample_name)
        mrn = sample_name[0]
        #print(mrn)
        
        float_mrn = float(mrn)
        #print(float_mrn)
        
        if float_mrn in label_data.MRN.values:
            sample_id = label_data.loc[label_data['MRN'] == float_mrn,'study_id']
            #print("sample id: " + str(int(sample_id)))
            
            print(mrn + " in label data")
            try:
                sample_num = int(sample_name[1])
            except:
                print("SAMPLE NAME ERROR")
                print(image_path)
                print(sample_name)
                continue
                
            try:
                ds = pydicom.dcmread(image_path)
                print("Image read in from dicom")

            except:
                print("IMAGE PATH ERROR")
                print(image_path)
                
            img = ds.pixel_array
            print("Image transferred to pixel array")
            img = img_as_float(rgb2gray(img))
        
            #if opt.view > 0:
            #    opt.view -= 1
    
            cropped_image = crop_image(img, 0)  # crop image with i-th params -- currently 0, use 1 to make images cropped smaller
            resized_image = transform.resize(cropped_image, output_shape=(opt.output_dim, opt.output_dim))
            print("Image cropped")
            
            resized_image = set_contrast(resized_image,opt)
            
            #print(type(str(img_num)))
            #print(type(sample_name))
            
            img_file_name = str(int(sample_id)) + "_" + sample_name[1] + "_" + str(img_num) + ".jpg"
            scipy.misc.imsave(os.path.join(opt.jpg_dump_dir,img_file_name), resized_image)
            print('Image file written to' + opt.jpg_dump_dir + img_file_name)
    
            imgs.append(resized_image)
            img_counter.append(img_num)
            img_num = img_num + 1
            
            out_data = {'images' : imgs}
            out_df = pd.DataFrame(data=out_data, index=img_counter)
            out_file = str(int(sample_id)) + "_" + sample_name[1] + ".pickle"
            #data.columns = data.columns.str.strip().str.lower().str.replace(" ","_")
            #print(data.columns)
            #h5f = h5py.File("preprocessed_images_20190315.csv", 'w')
            #h5f.create_dataset('all_images', data=data)
            #h5f.close()
            #data.to_csv("preprocessed_images_20190315.csv", sep=',')
            #data.to_pickle("preprocessed_images_20190315.pickle")
            #data.to_pickle("preprocessed_images_20190402.pickle")
            out_df.to_pickle(os.path.join(opt.out_dir,out_file))

        else:
            print(mrn + " not in label data")
            continue
    
        
    #print('\U00001F4A5')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-params', default=1, help="parameter settings to crop images "
                                                   "(see IMAGE_PARAMS_1 at top of file)")
    parser.add_argument('-view', default=0, help="number of images to visualize")
    parser.add_argument('-contrast', default=2, help="0 = original image, 1 = exposure.equalize_hist(image), 2 = exposure.equalize_adapthist(image), 3 = exposure.rescale_intensity(image)")
    parser.add_argument('-output_dim', default=256, help="dimension of output image")
    parser.add_argument('-rootdir', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/test-cabs/', 
                        help="directory of US sequence dicoms")
    parser.add_argument('-dcm_dir', default='D5048003_1', 
                        help="directory of US sequence dicoms")
    parser.add_argument('-jpg_dump_dir', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/train-jpgs/contrast2/', 
                        help="directory of US sequence dicoms")
    parser.add_argument('-out_dir', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/train-us-seqs/contrast2/', 
                        help="directory of US sequence dicoms")

    opt = parser.parse_args()
    opt.view = int(opt.view)
    opt.output_dim = int(opt.output_dim)

    cab_dir = os.path.join(opt.rootdir,opt.dcm_dir)
    print("cab_dir: " + cab_dir)

    dcm_files = load_file_names(cab_dir = cab_dir)
    #print(".dcm files: ")
    #print(dcm_files)
    
    data = load_labels()
    #print("data: " )
    #print(data)
    load_images(data, dcm_files, opt)


if __name__ == "__main__":
    main()





