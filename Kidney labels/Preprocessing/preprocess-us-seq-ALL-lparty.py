
import os
import pydicom
import pandas as pd
from skimage.color import rgb2gray
from skimage import exposure
from skimage import img_as_float
from skimage import transform
# import imageio
# from scipy.misc import imsave
# import matplotlib.pyplot as plt
import argparse
import numpy as np
from PIL import Image

# Image crop params
IMAGE_PARAMS = {0: [2.1, 4, 4],
                1: [2.5, 3.2, 3.2],
                2: [1.7, 4.5, 4.5]  # [2.5, 3.2, 3.2] # [1.7, 4.5, 4.5]
                }  # factor to crop images by: crop ratio, translate_x, translate_y

# loads all image paths to array
def load_file_names(cab_dir):
    dcm_files = []
    for subdir, dirs, files in os.walk(cab_dir):
        for file in files:
            if file.lower()[-4:] == ".dcm":
                dcm_files.append(os.path.join(subdir, file))
    return sorted(dcm_files)

# CROP IMAGES #
def crop_image(image, param_num):
    width = image.shape[1]
    height = image.shape[0]

    params = IMAGE_PARAMS[int(param_num)]

    new_dim = int(width // params[0])  # final image will be square of dim width/2 * width/2

    start_col = int(width // params[1])  # column (width position) to start from
    start_row = int(height // params[2])  # row (height position) to start from

    cropped_image = image[start_row:start_row + new_dim, start_col:start_col + new_dim]
    return cropped_image

# function to set the image contrast
def set_contrast(image, contrast):
    if contrast == 0:
        out_img = image
    elif contrast == 1:
        out_img = exposure.equalize_hist(image)
    elif contrast == 2:
        out_img = exposure.equalize_adapthist(image)
    elif contrast == 3:
        out_img = exposure.rescale_intensity(image)

    return out_img

def get_filename(opt, linking_log):
    mrn_usnum = opt.dcm_dir[1:]
    my_mrn = mrn_usnum.split("_")[0]
    deid = linking_log.loc[linking_log['mrn'] == np.int64(my_mrn)]['deid']
    my_usnum = mrn_usnum.split("_")[1]
    filename = '_'.join([str(int(deid)), my_usnum])

    return filename

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
    return np_arr

def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)

def load_images(dcm_files, link_log, lab_file, opt):

    img_num = 0
    img_creation_time = []
    manu = []
    acq_date = []
    acq_time = []
    img_id = []
    view_label = []
    surgery_label = []
    function_label = []
    reflux_label = []


    ## debug
    # image_path = dcm_files[0]

    for image_path in dcm_files:
        tokens = image_path.split("/")
        # print(tokens)

        rootdir_tokens = opt.cabs_rootdir.split("/")
        # print(rootdir_tokens)

        # get mrn and sample number
        # print(tokens[len(rootdir_tokens) - 1])
        sample_name = tokens[len(rootdir_tokens) - 1][1:].split("_") ## may need to remove [1:]
        # print(sample_name)
        mrn = sample_name[0]
        print("MRN being processed: ")
        print(mrn)

        float_mrn = float(mrn)
        # print(float_mrn)

        sample_id = float_mrn
        print("sample id: " + str(int(sample_id)))

        # print(link_log.head)
        deid = str(int(link_log.loc[link_log['mrn'] == np.int64(mrn)]['deid']))
        print("Deid: " + str(int(deid)))

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
            inst_num = "NA"
            img_creation_time.append(inst_num)
            print("Error grabbing instance number")

        img = ds.pixel_array
        print("Image transferred to pixel array")
        img = img_as_float(rgb2gray(img))
        cropped_image = crop_image(img, opt.params)  # crop image with i-th params
        resized_image = transform.resize(cropped_image, output_shape=(opt.output_dim, opt.output_dim))  # reduce
        final_img = set_contrast(resized_image, opt.contrast)

        mrn_img_id_val = str(int(mrn)) + "_" + sample_name[1] + "_" + str(inst_num)
        # print(lab_file.head)
        print("Image ID: " + mrn_img_id_val)

        try:
            my_view = lab_file.loc[lab_file['img_id'] == mrn_img_id_val, 'revised_labels'].values[0]
            my_surg = lab_file.loc[lab_file['img_id'] == mrn_img_id_val, 'Surgery'].values[0]
            my_func = lab_file.loc[lab_file['img_id'] == mrn_img_id_val, 'Function'].values[0]
            my_refl = lab_file.loc[lab_file['img_id'] == mrn_img_id_val, 'Reflux'].values[0]

            view_label.append(my_view)
            surgery_label.append(my_surg)
            function_label.append(my_func)
            reflux_label.append(my_refl)

            print("View label: " + my_view)
            print("Surgery label: " + my_surg)
            print("Function label: " + my_func)
            print("Reflux label: " + my_refl)

            img_id_val = str(int(deid)) + "_" + sample_name[1] + "_" + str(inst_num)
            img_id.append(img_id_val)
            print("De-id image ID: " + img_id_val)

            img_file_name = img_id_val + ".jpg"
            print("Image file name:" + img_file_name)

            manufacturer = ds.Manufacturer
            manu.append(manufacturer)
            img_date = ds.ContentDate
            acq_date.append(img_date)
            img_time = ds.ContentTime
            acq_time.append(img_time)
            print("Manufacturer name: " + manufacturer)
            print("Image acquisition date: " + img_date)
            print("Image acquisition time: " + img_time)

            save_image(final_img, os.path.join(opt.jpg_dump_dir, img_file_name))
            print('Image file written to' + opt.jpg_dump_dir + img_file_name)

        except:
            print("Image ID not in label file")

            my_view = 'Missing'
            my_surg = 'Missing'
            my_func = 'Missing'
            my_refl = 'Missing'

            view_label.append(my_view)
            surgery_label.append(my_surg)
            function_label.append(my_func)
            reflux_label.append(my_refl)

            print("View label: " + my_view)
            print("Surgery label: " + my_surg)
            print("Function label: " + my_func)
            print("Reflux label: " + my_refl)

            img_id_val = str(int(deid)) + "_" + sample_name[1] + "_" + str(inst_num)
            img_id.append(img_id_val)
            print("De-id image ID: " + img_id_val)

            img_file_name = img_id_val + ".jpg"
            print("Image file name:" + img_file_name)

            manufacturer = ds.Manufacturer
            manu.append(manufacturer)
            img_date = ds.ContentDate
            acq_date.append(img_date)
            img_time = ds.ContentTime
            acq_time.append(img_time)
            print("Manufacturer name: " + manufacturer)
            print("Image acquisition date: " + img_date)
            print("Image acquisition time: " + img_time)

            save_image(final_img, os.path.join(opt.jpg_dump_dir, img_file_name))
            print('Image file written to' + opt.jpg_dump_dir + img_file_name)

        img_num = img_num + 1

    # print("Image IDs: ")
    # print(img_id)
    # print("Image labels: ")
    # print(img_label)
    # print("Image manufacturers: ")
    # print(manu)
    df_dict = {'image_ids': img_id, 'image_manu': manu,
               'image_acq_date': acq_date, 'image_acq_time': acq_time, 'view_label': view_label,
               'surgery_label': surgery_label, 'function_label': function_label,
               'reflux_label': reflux_label}
    for key in df_dict:
        print(key + ": " + str(len(df_dict[key])))

    out_csv = pd.DataFrame(df_dict)
    return out_csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-params', default=1, help="parameter settings to crop images "
                                                   "(see IMAGE_PARAMS_1 at top of file)")
    parser.add_argument('-output_dim', default=256, help="dimension of output image")
    parser.add_argument('-rootdir', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/data/load_training_test_sets/',
                        help="directory of US sequence dicoms")
    parser.add_argument('-cabs_rootdir', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/all-cabs/',
                        help="directory of US sequence dicoms")
    parser.add_argument('-dcm_dir', default='D5048003_1',
                        help="directory of US sequence dicoms")
    parser.add_argument('-jpg_dump_dir', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/all_label_img/',
                        help="directory of US sequence dicoms")
    parser.add_argument('-csv_out_dir', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/all_label_csv/',
                        help="directory of US sequence dicoms")
    parser.add_argument('-label_filename', default='full_labels_20200211.csv',
                        help="directory of US sequence dicoms")
    parser.add_argument('-id_linking_filename', default='full_linking_log_20200211.csv',
                        help="directory of US sequence dicoms")
    parser.add_argument("-contrast", default=1, type=int, help="Image contrast to train on")

    opt = parser.parse_args() ## comment for debug
    opt.output_dim = int(opt.output_dim)

    cab_dir = os.path.join(opt.cabs_rootdir, opt.dcm_dir + "/")
    print("cab_dir: " + cab_dir)

    my_dcm_files = load_file_names(cab_dir=cab_dir)
    my_linking_log = pd.read_csv(opt.rootdir + '/' + opt.id_linking_filename)
    my_lab_file = pd.read_csv(opt.rootdir + '/' + opt.label_filename)

    csv_out = load_images(dcm_files=my_dcm_files, link_log=my_linking_log, lab_file=my_lab_file, opt=opt)
    csv_filename = get_filename(opt, linking_log=my_linking_log) + ".csv"

    print("Writing csv file to: " + opt.csv_out_dir + "/" + csv_filename)
    csv_out.to_csv(opt.csv_out_dir + "/" + csv_filename)



if __name__ == "__main__":
    main()



####
####
####
    # parser.add_argument('-params', default=1, help="parameter settings to crop images "
    #                                                "(see IMAGE_PARAMS_1 at top of file)")
    # parser.add_argument('-output_dim', default=256, help="dimension of output image")
    # parser.add_argument('-rootdir', default='C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Front-end-test-files/',
    #                     help="directory of US sequence dicoms")
    # parser.add_argument('-dcm_dir', default='D5048003_1',
    #                     help="directory of US sequence dicoms")
    # parser.add_argument('-jpg_dump_dir', default='C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Front-end-test-files/',
    #                     help="directory of US sequence dicoms")
    # parser.add_argument('-csv_out_dir', default='C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Front-end-test-files/',
    #                     help="directory of US sequence dicoms")
    # parser.add_argument('-label_filename', default='linking_log_20191120.csv',
    #                     help="directory of US sequence dicoms")
    # parser.add_argument('-id_linking_filename', default='view_label_df_20191120.csv',
    #                     help="directory of US sequence dicoms")
    # parser.add_argument("--contrast", default=1, type=int, help="Image contrast to train on")


    ## comment out after debug
# class make_opt():
#     def __init__(self):
#         self.params = 1
#         self.output_dim = 256
#         self.rootdir = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Image organization Nov 2019/'
#         self.cabs_rootdir = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Front-end-test-files/'
#         self.dcm_dir = 'D5048003_1'
#         self.jpg_dump_dir = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Front-end-test-files/'
#         self.csv_out_dir = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Front-end-test-files/'
#         self.label_filename = 'view_label_df_20200120.csv'
#         self.id_linking_filename = 'linking_log_20191120.csv'
#         self.contrast = 1
#
# opt = make_opt()

