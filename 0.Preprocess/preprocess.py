
import os
import pydicom
from skimage.color import rgb2gray
from skimage import exposure
from skimage import img_as_float
from skimage import transform
#import matplotlib.pyplot as plt
import argparse
import extract_labels
import pandas as pd
import numpy as np
import scipy
IMAGE_PARAMS = {0: [2.1, 4, 4],
                1: [2.5, 3.2, 3.2],
                2: [1.7, 4.5, 4.5] # [2.5, 3.2, 3.2] # [1.7, 4.5, 4.5]
               }  # factor to crop images by: crop ratio, translate_x, translate_y

#rootdir = os.path.join(os.path.abspath(os.path.join('./', os.pardir)), 'all-images')
#print(rootdir)
#rootdir = '/home/martaskreta/Desktop/CSC2516/all-images/'

# loads all image paths to array
def load_file_names(rootdir):
    dcm_files = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.lower()[-4:] == ".dcm":
                dcm_files.append(os.path.join(subdir, file))
    return dcm_files


# load labels and study ids into pandas dataframe
def load_labels():
    columns_to_extract = ['study_id', "MRN", 'Laterality',"If bilateral, which is the most severe kidney?",
                          'Age at Baseline', 'Gender', 'Date of Ultrasound 1', 'Surgery'] # Other cols that can be added: 'Surgery'
    # data = extract_labels.load_data("kidney_sample_labels.csv", "samples_with_studyids_and_mrns.csv",
    #                                 columns_to_extract, etiology=True)
    data = extract_labels.load_data("20190524_combinedObstRef.csv", "samples_with_studyids_and_mrns.csv",
                                    columns_to_extract, etiology=False)

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


def view_sample_images(img, counter, target_dir):
    #plt.figure()
    #plt.subplot(2, 2, 1)
    #plt.imshow(img, cmap='gray')
    rescaled_cropped_image = exposure.equalize_hist(img)
    #plt.subplot(2, 2, 2)
    #plt.imshow(rescaled_cropped_image, cmap='gray')
    #rescaled_cropped_image = exposure.equalize_adapthist(img)
    #plt.subplot(2, 2, 3)
    #plt.imshow(rescaled_cropped_image, cmap='gray')
    #rescaled_cropped_image = exposure.rescale_intensity(img)
    #plt.subplot(2, 2, 4)
    #plt.imshow(rescaled_cropped_image, cmap='gray')
    #plt.show()
    scipy.misc.imsave("./{}/{}.jpg".format(target_dir, counter), rescaled_cropped_image)

def get_kidney_view(file_name):
    view = ""
    side = ""
    if "rt-sag" in file_name:
        view = "Sag"
        side = "Right"
    elif "rt-trans" in file_name:
        view = "Trans"
        side = "Right"
    elif "lt-sag" in file_name:
        view = "Sag"
        side = "Left"
    elif "lt-trans" in file_name:
        view = "Trans"
        side = "Left"
    return view, side

def load_images(label_data, dcm_files, opt):
    columns = list(label_data.columns)
    columns += ['sample_num', 'crop_style', 'kidney_view', 'kidney_side', 'image','manufacturer','img_acq_date']

    data = pd.DataFrame(columns=columns)
    print(data.columns)
    counter_1 = 0
    counter_2 = 0
    for image_path in dcm_files:
        if "rt-sag" not in image_path and "lt-sag" not in image_path and "lt-trans" not in image_path \
                and "rt-trans" not in image_path:
            continue
        tokens = image_path.split("/")

        # get mrn and sample number
        # sample_name = tokens[9][1:].split("_")
        sample_name = tokens[2][1:].split("_")
        mrn = sample_name[0]
        try:
            sample_num = int(sample_name[1])
        except:
            print("SAMPLE NAME ERROR")
            print(image_path)
            print(sample_name)
            continue
            continue

        kidney_view, kidney_side = get_kidney_view(tokens[-1])
        if kidney_view == "":
            continue

        new_row = label_data.loc[label_data['MRN'] == float(mrn)]

        if new_row.empty:
            continue
        elif float(mrn) == float(2786688) and int(sample_num) == 1 and kidney_side == "Right":
            print("skipping patient_810 sample_1 side_right")
            continue
        elif float(mrn) == float(2707080) and int(sample_num) == 4:
            print("skipping patient_78 sample_4")
            continue

        if new_row.shape[0] == 2:
            new_row = new_row.iloc[0]
        
        try:
            ds = pydicom.dcmread(image_path)
        except:
            print("IMAGE PATH ERROR")
            print(image_path)
            continue
        img = ds.pixel_array
        img = img_as_float(rgb2gray(img))
        manufacturer = ds.Manufacturer
        
        #print("Manufacturer: " + manufacturer)
        #img_acq_date = ds.AcquisitionDate

        # number of cropped params to save
        num_image_crop_params = len(IMAGE_PARAMS)

        for i in range(num_image_crop_params):
            data = data.append(new_row)
            data.iloc[data.shape[0] - 1, data.columns.get_loc('sample_num')] = sample_num
            data.iloc[data.shape[0] - 1, data.columns.get_loc('kidney_view')] = kidney_view
            data.iloc[data.shape[0] - 1, data.columns.get_loc('kidney_side')] = kidney_side
            data.iloc[data.shape[0] - 1, data.columns.get_loc('manufacturer')] = manufacturer
            #data.iloc[data.shape[0] - 1, data.columns.get_loc('img_acq_date')] = img_acq_date
            data.iloc[data.shape[0] - 1, data.columns.get_loc('crop_style')] = i
            cropped_image = crop_image(img, i)  # crop image with i-th params
            resized_image = transform.resize(cropped_image, output_shape=(opt.output_dim, opt.output_dim))  # reduce
 
            # image resolution
            data.iloc[data.shape[0] - 1, data.columns.get_loc('image')] = [resized_image.reshape((1,-1))]  # reshape
            # matrix to row_length=1
          
            # display images
            #if opt.view > 0 and i == 2:
             #   temp_dir = "view_images"
              #  if not os.path.isdir(temp_dir):
               #     os.makedirs(temp_dir)
               # view_sample_images(resized_image, opt.view, temp_dir)
        #if opt.view > 0:
        #    opt.view -= 1

        #if opt.view == 0:
         #   import sys
          #  sys.exit(0)
    data = data.drop(["MRN"], axis=1)

    if 'Date of Ultrasound 1' in data.columns:
        data = data.sort_values(by=['Date of Ultrasound 1', 'study_id', 'sample_num'])
    data.columns = data.columns.str.strip().str.lower().str.replace(" ","_")
    #print(data.columns)
    #print("CROP_1: {} | CROP_2: {}".format(counter_1, counter_2))
    data.to_csv("preprocessed_images_20190612.csv", sep=',')
    data.to_pickle("preprocessed_images_20190612.pickle")

    print('\U0001F4A5')



def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-params', default=1, help="parameter settings to crop images "
    #                                                "(see IMAGE_PARAMS_1 at top of file)")
    parser.add_argument('-view', default=0, help="number of images to visualize")
    parser.add_argument('-output_dim', default=256, help="dimension of output image")
    parser.add_argument('-rootdir', default="/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/GANs/data/all-images", 
                        help="Directory where the images are")

    opt = parser.parse_args()
    opt.view = int(opt.view)
    opt.output_dim = int(opt.output_dim)

    dcm_files = load_file_names(opt.rootdir)
    print("DCM files generated")
    print(dcm_files)
    data = load_labels()
    print("Data loaded")
    load_images(data, dcm_files, opt)


if __name__ == "__main__":
    main()





