
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

IMAGE_PARAMS = {0: [2.1, 4, 4],
                1: [2.5, 3.2, 3.2]}  # factor to crop images by: crop ratio, translate_x, translate_y

rootdir = os.path.join(os.path.abspath(os.path.join('./', os.pardir)), 'all-images')
print(rootdir)
#rootdir = '/home/martaskreta/Desktop/CSC2516/all-images/'

# loads all image paths to array
def load_file_names():
    dcm_files = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.lower()[-4:] == ".dcm":
                dcm_files.append(os.path.join(subdir, file))
    return dcm_files

def load_cab_files_names():
    cab_files = []
    for subdir, dirs, files in os.walk("/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/all-cabs/"):
        for file in files:            
            cab_files.append(os.path.join(subdir,file))
    return cab_files

# load labels and study ids into pandas dataframe
def load_labels():
    columns_to_extract = ['study_id', "MRN", 'Surgery','Laterality',"If bilateral, which is the most severe kidney?",
                          'Age at Baseline', 'Gender', 'Date of Ultrasound 1']
    data = extract_labels.load_data("kidney_sample_labels.csv", "samples_with_studyids_and_mrns.csv",
                                    columns_to_extract)
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


def view_sample_images(img):
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    rescaled_cropped_image = exposure.equalize_hist(img)
    plt.subplot(2, 2, 2)
    plt.imshow(rescaled_cropped_image, cmap='gray')
    rescaled_cropped_image = exposure.equalize_adapthist(img)
    plt.subplot(2, 2, 3)
    plt.imshow(rescaled_cropped_image, cmap='gray')
    rescaled_cropped_image = exposure.rescale_intensity(img)
    plt.subplot(2, 2, 4)
    plt.imshow(rescaled_cropped_image, cmap='gray')
    plt.show()

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
    columns += ['sample_num', 'crop_style', 'kidney_view', 'kidney_side', 'image']

    data = pd.DataFrame(columns=columns)
    print(data.columns)

    for image_path in dcm_files:
        if "rt-sag" not in image_path and "lt-sag" not in image_path and "lt-trans" not in image_path \
                and "rt-trans" not in image_path:
            continue
        tokens = image_path.split("/")

        # get mrn and sample number
        sample_name = tokens[6][1:].split("_")
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

        try:
            ds = pydicom.dcmread(image_path)
        except:
            print("IMAGE PATH ERROR")
            print(image_path)
        img = ds.pixel_array
        img = img_as_float(rgb2gray(img))

        # number of cropped params to save
        num_image_crop_params = len(IMAGE_PARAMS)

        for i in range(num_image_crop_params):
            data = data.append(new_row)
            data.iloc[data.shape[0] - 1, data.columns.get_loc('sample_num')] = sample_num
            data.iloc[data.shape[0] - 1, data.columns.get_loc('kidney_view')] = kidney_view
            data.iloc[data.shape[0] - 1, data.columns.get_loc('kidney_side')] = kidney_side
            data.iloc[data.shape[0] - 1, data.columns.get_loc('crop_style')] = i
            cropped_image = crop_image(img, i)  # crop image with i-th params
            resized_image = transform.resize(cropped_image, output_shape=(opt.output_dim, opt.output_dim))  # reduce
            # image resolution
            data.iloc[data.shape[0] - 1, data.columns.get_loc('image')] = [resized_image.reshape((1,-1))]  # reshape
            # matrix to row_length=1


            # display images
            if opt.view > 0 :
                view_sample_images(resized_image)
        if opt.view > 0:
            opt.view -= 1


    data = data.drop(["MRN"], axis=1)

    if 'Date of Ultrasound 1' in data.columns:
        data = data.sort_values(by=['Date of Ultrasound 1', 'study_id', 'sample_num'])
    data.columns = data.columns.str.strip().str.lower().str.replace(" ","_")
    print(data.columns)
    #h5f = h5py.File("preprocessed_images_20190315.csv", 'w')
    #h5f.create_dataset('all_images', data=data)
    #h5f.close()
    data.to_csv("preprocessed_images_20190315.csv", sep=',')
    data.to_pickle("preprocessed_images_20190315.pickle")

    print('\U00001F4A5')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-params', default=1, help="parameter settings to crop images "
                                                   "(see IMAGE_PARAMS_1 at top of file)")
    parser.add_argument('-view', default=0, help="number of images to visualize")
    parser.add_argument('-output_dim', default=256, help="dimension of output image")

    opt = parser.parse_args()
    opt.view = int(opt.view)
    opt.output_dim = int(opt.output_dim)

    dcm_files = load_file_names()
    data = load_labels()
    load_images(data, dcm_files, opt)


if __name__ == "__main__":
    main()





