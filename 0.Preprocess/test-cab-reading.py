# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:58:39 2019

@author: Lauren
"""

    ## IMPORT LIBRARIES 
import argparse
import pyunpack
import os
from pyunpack import Archive


    ## FUNCTIONS FOR READING AND PREPROCESSING CABS/DCMS
def unzipfile(filename,destfolder):
    print("Unzipping:",filename," to ",destfolder)
    Archive(filename).extractall(destfolder)

def load_cab_names(rootdir):
    cabs = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.lower()[-4:] == ".cab":
                cabs.append(os.path.join(subdir, file))
    return cabs

def open_cabs(cab_file,rootdir):
    try:
        cab_dest = os.mkdir(os.path.join(rootdir,cab_file[0:-4]))
        print("open_cabs destfolder created: ", cab_dest)
    except: 
        cab_dest = os.path.join(rootdir,cab_file[0:-4])
        print("destfolder already exits, will write to existing folder")    
    unzipfile(cab_file,cab_dest)    

# loads all image paths to array
def load_file_names(cabdir):
    dcm_files = []
    for subdir, dirs, files in os.walk(cabdir):
        for file in files:
            if file.lower()[-4:] == ".dcm":
                dcm_files.append(os.path.join(subdir, file))
    return dcm_files

def preprocess_dcm_files():

    ### FILL IN WITH DCM PREPROCESSING SCRIPTS FROM MARTA
    
    return None

    ## MAIN SCRIPT FUNCTION 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-rootdir', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/test-cabs', 
                        help="directory where cab files are located")
    #parser.add_argument('-view', default=0, help="number of images to visualize")
    #parser.add_argument('-output_dim', default=256, help="dimension of output image")

    opt = parser.parse_args()
    #opt.view = int(opt.view)
    #opt.output_dim = int(opt.output_dim)

    cab_files = load_cab_names(opt.rootdir)
    print(cab_files)
    #data = load_labels()
    #load_images(data, dcm_files, opt)
    
    for cab in cab_files: 
        open_cabs(cab,opt.rootdir)
        mycabdir = os.path.join(os.path.join(opt.rootdir,cab[0:-4]))
        dcm_files = load_file_names(mycabdir)
        
        print(dcm_files)
        
    


if __name__ == "__main__":
    main()


