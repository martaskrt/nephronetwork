# -*- coding: utf-8 -*-

import os
import pydicom ##
import pandas as pd
from skimage.color import rgb2gray ##
from skimage import exposure ##
from skimage import img_as_float ##
from skimage import transform ##
# import imageio
# from scipy.misc import imsave
# import matplotlib.pyplot as plt
import argparse
import numpy as np
from PIL import Image ##
#from torchvision import models ##
import torch
import cv2  ##
from torch import nn

##
## IMAGE PRE-PROCESSING
##

# Image crop params
IMAGE_PARAMS = {0: [2.1, 4, 4],
                1: [2.5, 3.2, 3.2],
                2: [1.7, 4.5, 4.5]  # [2.5, 3.2, 3.2] # [1.7, 4.5, 4.5]
                }  # factor to crop images by: crop ratio, translate_x, translate_y

# loads all image paths to array
def load_file_names(dcms_dir):
    dcm_files = []
    for subdir, dirs, files in os.walk(dcms_dir):
        for file in files:
            if file.lower()[-4:] == ".dcm":
                dcm_files.append(os.path.join(subdir, file))
    return sorted(dcm_files)

def get_dcm_info(dcm_file):
    my_dcm = pydicom.dcmread(dcm_file)

    try:
        my_mrn = my_dcm.PatientID

    except AttributeError:
        try:
            my_mrn = my_dcm.ImageID

        except AttributeError:
            my_mrn = "NA"

    return my_mrn

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

# def get_filename(opt, linking_log):
#     mrn_usnum = opt.dcm_dir[1:]
#     my_mrn = mrn_usnum.split("_")[0]
#     deid = linking_log.loc[linking_log['mrn'] == np.int64(my_mrn)]['deid']
#     my_usnum = mrn_usnum.split("_")[1]
#     filename = '_'.join([str(int(deid)), my_usnum])
#
#     return filename

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
        try:
            im = Image.fromarray(im)
            im.save(path)
        except TypeError:
            pass

def load_images(dcm_files, link_log, opt, my_nn=None):

    img_num = 0
    img_creation_time = []
    manu = []
    acq_date = []
    acq_time = []
    acc_num = []
    img_id = []
    bladder_p = []
    sag_right_p = []
    sag_left_p = []
    trans_right_p = []
    trans_left_p = []



    ## debug
    # image_path = dcm_files[0]

    for image_path in dcm_files:
        # tokens = image_path.split("/")
        # print(tokens)

        # rootdir_tokens = opt.dcm_dir.split("/")
        # print(rootdir_tokens)

        # get mrn and sample number
        # print(tokens[len(rootdir_tokens) - 1])
        # sample_name = tokens[len(rootdir_tokens) - 1][1:].split("_") ## may need to remove [1:]
        # print(sample_name)
        # mrn = sample_name[0]

        try:
            ds = pydicom.dcmread(image_path)
            print("Image read in from dicom")
        except:
            print("IMAGE PATH ERROR")
            print(image_path)

        try:
            mrn = ds.PatientID
        except AttributeError:
            mrn = '00000000'
            print("Patient ID not accessible")

        print("MRN being processed: ")
        print(mrn)

        float_mrn = float(mrn)
        # print(float_mrn)

        sample_id = float_mrn
        print("sample id: " + str(int(sample_id)))

        # print(link_log.head)
        try:
            deid = str(int(link_log.loc[link_log['mrn'] == np.int64(mrn)]['deid']))
        except TypeError:
            deid = mrn
            print("MRN not in linking log")

        print("Deid: " + str(deid))

        try:
            inst_num = ds.InstanceNumber
            img_creation_time.append(inst_num)
        except:
            inst_num = "NA"
            img_creation_time.append(inst_num)
            print("Error grabbing instance number")

        try:
            img_acc = ds.AccessionNumber
        except:
            img_acc = "NA"
            print("Error grabbing accession number")
        acc_num.append(img_acc)

        try:
            img = ds.pixel_array
            print("Image transferred to pixel array")
            img = img_as_float(rgb2gray(img))
            cropped_image = crop_image(img, opt.params)  # crop image with i-th params
            resized_image = transform.resize(cropped_image, output_shape=(opt.output_dim, opt.output_dim))  # reduce
            final_img = set_contrast(resized_image, opt.contrast)
            img = final_img
            if opt.predict_views:
                img = cv2.GaussianBlur(final_img, (5, 5), 0)
                tor_img = torch.tensor(img).to(opt.device).view(1, 256, 256)
                input_img = torch.cat((tor_img, tor_img, tor_img), 0).view(1, 3, 256, 256).to(opt.device)  # .double()

                my_nn = my_nn.double().to(opt.device)
                fwd_pass_out = torch.sigmoid(my_nn(input_img.double())).detach().numpy()

                # print(fwd_pass_out)
                # print(fwd_pass_out.shape)

                p_bladder = fwd_pass_out[0][0]
                p_other = fwd_pass_out[0][1]
                p_sag_left = fwd_pass_out[0][2]
                p_sag_right = fwd_pass_out[0][3]
                p_trans_left = fwd_pass_out[0][4]
                p_trans_right = fwd_pass_out[0][5]

                print("prob bladder = " + str(p_bladder))
                print("prob other = " + str(p_other))
                print("prob sag left = " + str(p_sag_left))
                print("prob sag right = " + str(p_sag_right))
                print("prob trans left = " + str(p_trans_left))
                print("prob trans right = " + str(p_trans_right))

                bladder_p.append(p_bladder)
                sag_right_p.append(p_sag_right)
                sag_left_p.append(p_sag_left)
                trans_right_p.append(p_trans_right)
                trans_left_p.append(p_trans_left)

            try:
                # my_view = lab_file.loc[lab_file['img_id'] == mrn_img_id_val, 'revised_labels'].values[0]
                # my_surg = lab_file.loc[lab_file['img_id'] == mrn_img_id_val, 'Surgery'].values[0]
                # my_func = lab_file.loc[lab_file['img_id'] == mrn_img_id_val, 'Function'].values[0]
                # my_refl = lab_file.loc[lab_file['img_id'] == mrn_img_id_val, 'Reflux'].values[0]
                #
                # view_label.append(my_view)
                # surgery_label.append(my_surg)
                # function_label.append(my_func)
                # reflux_label.append(my_refl)
                #
                # print("View label: " + my_view)
                # print("Surgery label: " + my_surg)
                # print("Function label: " + my_func)
                # print("Reflux label: " + my_refl)

                try:
                    manufacturer = ds.Manufacturer
                except AttributeError:
                    manufacturer = "NA"
                manu.append(manufacturer)

                try:
                    img_date = ds.ContentDate
                except AttributeError:
                    try:
                        img_date = ds.StudyDate
                    except AttributeError:
                        img_date = 'Unknown'

                acq_date.append(img_date)

                try:
                    img_time = ds.ContentTime
                except AttributeError:
                    try:
                        img_time = ds.StudyTime
                    except AttributeError:
                        img_time = 'Unknown'

                acq_time.append(img_time)
                print("Manufacturer name: " + manufacturer)
                print("Image acquisition date: " + img_date)
                print("Image acquisition time: " + img_time)

                img_id_val = str(int(deid)) + "_" + str(inst_num) + "_" + str(img_date)
                img_id.append(img_id_val)
                print("De-id image ID: " + img_id_val)

                img_file_name = img_id_val + ".jpg"
                print("Image file name:" + img_file_name)

                save_image(img, os.path.join(opt.jpg_dump_dir, img_file_name))
                print('Image file written to' + opt.jpg_dump_dir + img_file_name)

            except:
                print("Image ID not in label file")

                # my_view = 'Missing'
                # my_surg = 'Missing'
                # my_func = 'Missing'
                # my_refl = 'Missing'
                #
                # view_label.append(my_view)
                # surgery_label.append(my_surg)
                # function_label.append(my_func)
                # reflux_label.append(my_refl)
                #
                # print("View label: " + my_view)
                # print("Surgery label: " + my_surg)
                # print("Function label: " + my_func)
                # print("Reflux label: " + my_refl)

                try:
                    manufacturer = ds.Manufacturer
                except AttributeError:
                    manufacturer = "NA"
                manu.append(manufacturer)

                try:
                    img_date = ds.ContentDate
                except AttributeError:
                    try:
                        img_date = ds.StudyDate
                    except AttributeError:
                        img_date = 'Unknown'

                acq_date.append(img_date)

                try:
                    img_time = ds.ContentTime
                except AttributeError:
                    try:
                        img_time = ds.StudyTime
                    except AttributeError:
                        img_time = 'Unknown'
                acq_time.append(img_time)

                print("Manufacturer name: " + manufacturer)
                print("Image acquisition date: " + img_date)
                print("Image acquisition time: " + img_time)

                img_id_val = str(int(deid)) + "_" + str(inst_num) + str(img_date)
                img_id.append(img_id_val)
                print("De-id image ID: " + img_id_val)

                img_file_name = img_id_val + ".jpg"
                print("Image file name:" + img_file_name)

                save_image(img, os.path.join(opt.jpg_dump_dir, img_file_name))
                print('Image file written to' + opt.jpg_dump_dir + img_file_name)

        except AttributeError:
            pass

        img_num = img_num + 1

    # print("Image IDs: ")
    # print(img_id)
    # print("Image labels: ")
    # print(img_label)
    # print("Image manufacturers: ")
    # print(manu)
    # df_dict = {'image_ids': img_id, 'image_manu': manu,
    #            'image_acq_date': acq_date, 'image_acq_time': acq_time, 'view_label': view_label,
    #            'surgery_label': surgery_label, 'function_label': function_label,
    #            'reflux_label': reflux_label}

    if opt.predict_views:
        df_dict = {'image_ids': img_id, 'image_manu': manu,
                   'image_acq_date': acq_date, 'image_acq_time': acq_time, 'acc_num': acc_num,
                   'bladder_p': bladder_p, 'sag_right_p': sag_right_p,
                   'sag_left_p': sag_left_p, 'trans_right_p': trans_right_p,
                   'trans_left_p': trans_left_p}
    else:
        df_dict = {'image_ids': img_id, 'image_manu': manu,
                   'image_acq_date': acq_date, 'acc_num': acc_num}

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
    parser.add_argument('-dcm_dir', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/all-cabs_pull20200322/us/1-1000/',
                        help="directory of US sequence dicoms")
    parser.add_argument('-jpg_dump_dir', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/all-jpgs-dmsa/',
                        help="directory of US sequence dicoms")
    parser.add_argument('-csv_out', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/all_label_csv/test_out.csv',
                        help="directory of US sequence dicoms")
    parser.add_argument('-id_linking_filename', default='full_linking_log_20200405.csv',
                        help="directory of US sequence dicoms")
    parser.add_argument('-device', default='cuda', ## don't forget to change to GPU
                        help="directory of US sequence dicoms")
    parser.add_argument('-pretrained_mod_view_weights', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/data/load_training_test_sets/custom_granular_best.pth.tar', ## don't forget to change to GPU
                        help="directory of US sequence dicoms")
    parser.add_argument("-contrast", default=1, type=int, help="Image contrast to train on")
    parser.add_argument("-predict_views", action='store_true', default=False, help="Add view prediction to output csv?")


    opt = parser.parse_args() ## comment for debug
    opt.output_dim = int(opt.output_dim)

    my_dcm_files = load_file_names(dcms_dir=opt.dcm_dir)
    my_linking_log = pd.read_csv(opt.rootdir + '/' + opt.id_linking_filename)

    if opt.predict_views:
        ## From Delvin at: https://github.com/martaskrt/nephronetwork/blob/master/kidney_label_classifier/nephro_net/net.py
        class Net(nn.Module):

            def __init__(self, task, mod='vgg'):
                super().__init__()
                self.task = task
                self.mod = mod

                task_d = {'view': 4, 'granular': 6, 'bladder': 2}
                # in features
                model_d = {'alexnet': 256, 'vgg': 512, 'resnet': 2048, 'densenet': 1024, 'squeezenet': 512,
                           'custom': 0}  # doesn't matter

                in_f = model_d[self.mod]
                out = task_d[self.task]

                # both vgg and alexnet have same final fc layer
                # if self.mod == 'alexnet':
                #     print('alexnet')
                #     self.model = models.alexnet(pretrained=True)
                #     self.gap = nn.AdaptiveAvgPool2d(1)
                #     self.clf_on = nn.Linear(in_f, out)  # 256 if alexnet, 512 if vgg or resnet
                # elif self.mod == 'vgg':
                #     print('vgg')
                #     self.model = models.vgg11_bn(pretrained=True)
                #     self.gap = nn.AdaptiveAvgPool2d(1)
                #     self.clf_on = nn.Linear(in_f, out)  # 256 if alexnet, 512 if vgg or resnet
                # elif self.mod == 'resnet':
                #     print('resnet')
                #     self.model = models.resnet50(pretrained=True)
                #     self.model.fc = nn.Linear(in_features=in_f, out_features=out, bias=True)
                # elif self.mod == 'densenet':
                #     print('densenet')
                #     self.model = models.densenet121(pretrained=True)
                #     self.model.classifier = nn.Linear(in_features=in_f, out_features=out, bias=True)
                # elif self.mod == 'squeezenet':
                #     print('densenet')
                #     self.model = models.squeezenet1_1(pretrained=True)
                #     self.model.classifier[1] = nn.Conv2d(512, out, kernel_size=(1, 1), stride=(1, 1))

                # input is 256x256
                # let's try a simple repeated cbr-maxpool with small filters (5)
                # so we repeat cbr-max 4 times, global average pool into classification

                # 3 x 256 x 256
                # outputfilter w/h = (W−F+2P) / S) + 1, ((256 - 5 + 0)/1) + 1
                # # p = (f - 1)/2 for stride = 1 for p for same padding
                # x = torch.randn([1, 3, 256, 256 ])
                # x1 = (maxp(relu(bn64(conv64(x)))))
                # x2 = (maxp(relu(bn128(conv128(x1)))))
                # x3 = (maxp(relu(bn256(conv256(x2)))))
                # x4 = (maxp(relu(bn512(conv512(x3)))))
                # x5 = classifier(gap(x4).view(x4.size(0), -1))
                # p = 2 for same padding, formula?

                self.conv64 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=0)
                self.conv128 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, stride=1, padding=0)
                self.conv256 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=7, stride=1, padding=0)
                self.conv512 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=7, stride=1, padding=0)
                # outputs filters of w - f + 2p/s + 1, or 251

                self.bn64 = nn.BatchNorm2d(64)  #
                self.bn128 = nn.BatchNorm2d(128)  #
                self.bn256 = nn.BatchNorm2d(256)  #
                self.bn512 = nn.BatchNorm2d(512)  #
                self.relu = nn.ReLU()
                self.maxp = nn.MaxPool2d(kernel_size=3, stride=2)

                # first conv-batch-relu pass gives 32 x 127 x 127

                self.gap = nn.AdaptiveAvgPool2d(1)
                self.classifier = nn.Linear(in_features=512, out_features=out, bias=True)

            def forward(self, x):
                just_forward = ('resnet', 'squeezenet', 'densenet')

                if self.mod in just_forward:
                    x = self.model(x)
                elif self.mod == 'custom':
                    x = self.maxp(self.relu(self.bn64(self.conv64(x))))
                    # print(x.shape)
                    x = self.maxp(self.relu(self.bn128(self.conv128(x))))
                    # print(x.shape)
                    x = self.maxp(self.relu(self.bn256(self.conv256(x))))
                    # print(x.shape)
                    x = self.maxp(self.relu(self.bn512(self.conv512(x))))
                    # print(x.shape)
                    x = self.gap(x).view(x.size(0), -1)  # to 'unroll'
                    # print(x.shape)
                    x = self.classifier(x)
                    # print(x.shape)
                # else:  # alexnet, vgg, feature extractor, with adaptive pooling
                #     x = self.model.features(x)
                #     # print(x.shape)#, mb x256 x 7 x 7 // 512 x 8 x 8
                #     x = self.gap(x).view(x.size(0), -1)  # ADAPTIVE POOL, 'collapses' into mb x 256 x 1, then mb x 256
                #     # print(x.shape) # mb x 256
                #     # x = torch.max(x, 0, keepdim = True)[0] # max of each column
                #     # print(x.shape)
                #     x = self.clf_on(x)
                # print(x.shape)
                return x

        ## INITIALIZE NN
        net = Net(task='granular', mod='custom').to(opt.device)
        # print(net)

        print("Loading: " + opt.pretrained_mod_view_weights)
        pretrained_dict = torch.load(opt.pretrained_mod_view_weights,
                                     map_location=torch.device(opt.device))  # .state_dict#['model_state_dict']
        # print(pretrained_dict)

        # 3. load the new state dict
        net.load_state_dict(pretrained_dict['state_dict'])
        net.eval()

        with torch.no_grad():
            net.zero_grad()
            labels_out=load_images(dcm_files=my_dcm_files, link_log=my_linking_log, my_nn=net, opt=opt)

    else:
        labels_out = load_images(dcm_files=my_dcm_files, link_log=my_linking_log, opt=opt)
    # csv_filename = get_filename(opt, linking_log=my_linking_log) + ".csv"

    print("Writing csv file to: " + opt.csv_out)
    labels_out.to_csv(opt.csv_out)

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
#         self.dcm_dir = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Front-end-test-files/test_us_dmsa/'
#         self.jpg_dump_dir = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Front-end-test-files/'
#         self.csv_out = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Front-end-test-files/test_csv_out.csv'
#         self.id_linking_filename = 'full_linking_log_20200405.csv'
#         self.pretrained_mod_view_weights = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/nephro/custom_granular_best.pth.tar'
#         self.device = 'cpu'
#         self.contrast = 1
#
# opt = make_opt()
#
# opt.output_dim = int(opt.output_dim)
#
# # cab_dir = os.path.join(opt.cabs_rootdir, opt.dcm_dir + "/")
# # print("cab_dir: " + cab_dir)
#
# my_dcm_files = load_file_names(dcms_dir=opt.dcm_dir)
# my_linking_log = pd.read_csv(opt.rootdir + '/' + opt.id_linking_filename)
#
# ## INITIALIZE NN
#
# net = Net(task='granular',mod='custom')
# print(net)
#
# print("Loading: " + opt.pretrained_mod_view_weights)
# pretrained_dict = torch.load(opt.pretrained_mod_view_weights, map_location=torch.device('cpu'))  # .state_dict#['model_state_dict']
# print(pretrained_dict)
# # 3. load the new state dict
# net.load_state_dict(pretrained_dict['state_dict'])
# net.eval()
#
# with torch.no_grad():
#     net.zero_grad()
#     test_out = load_images(dcm_files=my_dcm_files, link_log=my_linking_log, my_nn=net,opt=opt)
#
# # csv_filename = get_filename(opt, linking_log=my_linking_log) + ".csv"
#
# print("Writing csv file to: " + opt.csv_out)
# test_out.to_csv(opt.csv_out)