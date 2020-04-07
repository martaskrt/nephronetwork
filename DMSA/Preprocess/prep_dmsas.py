import pydicom
import argparse
import os
from PIL import Image
import pandas as pd
import numpy as np

# loads all image paths to array
def load_file_names(dsma_dir):
    dcm_files = []
    for subdir, dirs, files in os.walk(dsma_dir):
        for file in files:
            if file.lower()[-4:] == ".dcm":
                dcm_files.append(os.path.join(subdir, file))
    return sorted(dcm_files)

def read_dcm(dcm_dir):
    my_dcm = pydicom.dcmread(dcm_dir)
    return my_dcm

def format_np_output(np_arr):
    """
        From: https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/misc_functions.py

        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr.astype('uint8')

def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        print(im)
        im = Image.fromarray(im)
    im.save(path)

def write_images(img_pths, my_path, out_csv):
    mrn_list = []
    manu_list = []
    pa_list = []
    img_date_list = []
    img_filename_list = []
    acc_num = []

    for dup_pth in list(set(img_pths)):
        dcm_files = load_file_names(dup_pth)
        for my_file in dcm_files:
            my_dcm = read_dcm(my_file)

            if 'PRIMARY' in my_dcm.ImageType:

                try:
                    my_mrn = my_dcm.PatientID
                except AttributeError:
                    my_mrn = "NA"
                    print("MRN reading error from: " + my_file)

                try:
                    my_manu = my_dcm.Manufacturer
                except AttributeError:
                    my_manu = "NA"
                    print("Manufacturer reading error from: " + my_file)

                try:
                    my_pa = my_dcm.ImageID[0]
                except AttributeError:
                    my_pa = "NA"
                    print("ImageID reading error from: " + my_file)

                try:
                    img_date = my_dcm.StudyDate
                except AttributeError:
                    img_date = "NA"
                    print("Image date reading error from: " + my_file)

                try:
                    img_acc = my_dcm.AccessionNumber
                except AttributeError:
                    img_acc = "NA"
                    print("Image accession reading error from: " + my_file)

                img_filename = my_mrn + "_" + img_date + "_" + my_pa + ".jpg"

                mrn_list.append(my_mrn)
                manu_list.append(my_manu)
                pa_list.append(my_pa)
                img_date_list.append(img_date)
                img_filename_list.append(img_filename)
                acc_num.append(img_acc)

                pix_array = np.array(my_dcm.pixel_array)
                # print(pix_array)
                np_pix_array = format_np_output(pix_array)
                path = my_path + "/dmsa-jpgs/" + img_filename
                # print(np_pix_array.dtype)
                try:
                    my_im = Image.fromarray(np_pix_array.astype(np.uint8)).convert('L')
                    # save_image(pix_array, path)
                    my_im.save(path)
                    print("Image written to: " + path)
                except TypeError:
                    print("Error converting dicom from " + my_file)

    img_df = pd.DataFrame({"MRN": mrn_list, "IMG_NAME": img_filename_list, "IMG_DATE": img_date_list,
                           "PA": pa_list, "MANU": manu_list, "ACC_NUM": acc_num})

    csv_path = my_path + "/" + out_csv
    print(csv_path)
    img_df.to_csv(csv_path)
    # return img_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dcm_folder', default="C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Front-end-test-files/test_dmsa/", help="path to audio file")
    parser.add_argument('-linking_log', default=None, help="MRN-to-deID linking log file path (if necessary) and name")
    parser.add_argument('-out_path', default="C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Front-end-test-files/", help="Where to write curve file")
    parser.add_argument('-out_filename', default="dmsa-jpgs.csv", help="Curve file name")
    # parser.add_argument('-checkpoint', default="C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Uroflow/AU_forward-pass_test-files/Water.pt", help="Network file")
    parser.add_argument('-save_imgs', action='store_true', default=False, help="Generate-images?")

    opt = parser.parse_args()

    my_dcms = load_file_names(opt.dcm_folder)
    print("All .dcm paths")
    print(my_dcms[0])
    my_dcm_pths = ['/'.join(dcm.split("/")[0:-1]) for dcm in my_dcms]
    print(".dcm root paths:")
    print(my_dcm_pths[0])
    ## find duplicated paths and only keep ones that are duplicated
    ## from: https://kite.com/python/answers/how-to-find-duplicates-in-a-list-in-python
    duplicate_pths = []
    for item in my_dcm_pths:
        if my_dcm_pths.count(item) > 1:
            duplicate_pths.append(item)

    ## read in emission data from each duplicated pth

    # dcm_file_list = []
    # for dup_pth in list(set(duplicate_pths)):
    #     dcm_files = load_file_names(dup_pth)
    #     for my_file in dcm_files:
    #         dcm_file_list.append(read_dcm(my_file))

    ## primary emission images
    # dcm_emissions = []
    # for my_dcm in dcm_file_list:
    #     if 'PRIMARY' in my_dcm.ImageType:
    #         dcm_emissions.append(my_dcm)

    # print(".dcm emissions")
    # print(dcm_emissions[0])
    ## write pixel array to jpg image

    # write_images(dcm_emissions, opt.out_path, opt.out_filename)

    write_images(duplicate_pths, opt.out_path, opt.out_filename)

if __name__ == "__main__":
    main()



###
###      DEBUGGING
###

# class opt():
#     dcm_folder = "C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Front-end-test-files/test_dmsa"
#
# my_dcms = load_file_names(opt.dcm_folder)
# my_dcm_pths = ['/'.join(dcm.split("\\")[0:-1]) for dcm in my_dcms]
# ## find duplicated paths and only keep ones that are duplicated
# ## from: https://kite.com/python/answers/how-to-find-duplicates-in-a-list-in-python
# duplicate_pths = []
# for item in my_dcm_pths:
#     if my_dcm_pths.count(item) > 1:
#         duplicate_pths.append(item)
#
# ## read in emission data from each duplicated pth
#
# dcm_file_list = []
# for dup_pth in list(set(duplicate_pths)):
#     dcm_files = load_file_names(dup_pth)
#     for my_file in dcm_files:
#         dcm_file_list.append(read_dcm(my_file))
#
# ## primary emission images
# dcm_emissions = []
# for my_dcm in dcm_file_list:
#     if 'PRIMARY' in my_dcm.ImageType:
#         dcm_emissions.append(my_dcm)
#
# ## write pixel array to jpg image
#
# write_images(dcm_emissions, "C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Front-end-test-files/test_dmsa/")







