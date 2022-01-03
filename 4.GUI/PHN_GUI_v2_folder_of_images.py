import PySimpleGUI as sg
from PIL import Image #, ExifTags
# import pydicom
from skimage.color import rgb2gray
from skimage import img_as_float, transform, exposure
from matplotlib import cm
# from skimage.io import imsave
# from skimage import io
# io.use_plugin('imageio', 'imsave')
import matplotlib.cm as mpl_color_map
import numpy as np
import torch
from torch import nn
import os
import copy
import scipy
import scipy.special.cython_special
import pandas as pd
import time
#import string

## HARD CODED: PREDICTION MODEL ARCHITECTURE BEING USEDc
# import Prehdict_CNN

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

# import scipy
# import argparse
#
# from misc_functions import get_example_params, save_class_activation_images
#
# import prep_single_sample

###
###     NETWORK MODULE
###

class PrehdictNet(nn.Module):
    def __init__(self, num_inputs=2, classes=2,dropout_rate=0.5,output_dim=256, jigsaw=False):
        super(PrehdictNet, self).__init__()
        self.num_inputs = num_inputs
        self.output_dim=output_dim
        self.jigsaw = jigsaw
        self.conv1 = nn.Sequential()
        if jigsaw:
            self.conv1.add_module('conv1_s1',nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=0))
        else:
            self.conv1.add_module('conv1_s1',nn.Conv2d(1, 96, kernel_size=11, stride=2, padding=0))
        self.conv1.add_module('batch1_s1', nn.BatchNorm2d(96))
        self.conv1.add_module('relu1_s1',nn.ReLU(inplace=True))
        #self.conv1.add_module('batch1_s1', nn.BatchNorm2d(96))
        self.conv1.add_module('pool1_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        #self.conv1.add_module('lrn1_s1',LRN(local_size=5, alpha=0.0001, beta=0.75))
        #**************** ADDED *******************#
        self.conv1.add_module('pool1_s2', nn.MaxPool2d(kernel_size=2, stride=1))
        self.conv2 = nn.Sequential()
        self.conv2.add_module('conv2_s1',nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2))
        self.conv2.add_module('batch2_s1', nn.BatchNorm2d(256))
        self.conv2.add_module('relu2_s1',nn.ReLU(inplace=True))
        #self.conv2.add_module('batch2_s1', nn.BatchNorm2d(256))
        self.conv2.add_module('pool2_s1',nn.MaxPool2d(kernel_size=3, stride=2))
        #self.conv2.add_module('lrn2_s1',LRN(local_size=5, alpha=0.0001, beta=0.75))
        # ********** added*********** #
        # self.conv2.add_module('pool', nn.MaxPool2d(kernel_size=2, padding=2, stride=1))
        self.conv2.add_module('conv2b', nn.Conv2d(256, 256, kernel_size=2, padding=1, stride=1))
        self.conv2.add_module('batch2_s1', nn.BatchNorm2d(256))
        self.conv2.add_module('relu2_s1',nn.ReLU(inplace=True))
        #self.conv2.add_module('batch2_s1', nn.BatchNorm2d(256))

        self.conv3 = nn.Sequential()
        self.conv3.add_module('conv3_s1',nn.Conv2d(256, 384, kernel_size=3, padding=1))
        self.conv3.add_module('batch3_s1', nn.BatchNorm2d(384))
        self.conv3.add_module('relu3_s1',nn.ReLU(inplace=True))
        #self.conv3.add_module('batch3_s1', nn.BatchNorm2d(384))

        self.conv4 = nn.Sequential()
        self.conv4.add_module('conv4_s1',nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2))
        self.conv4.add_module('batch4_s1', nn.BatchNorm2d(384))
        self.conv4.add_module('relu4_s1',nn.ReLU(inplace=True))
        #self.conv4.add_module('batch4_s1', nn.BatchNorm2d(384))

        self.conv5 = nn.Sequential()
        self.conv5.add_module('conv5_s1',nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2))
        self.conv5.add_module('batch5_s1', nn.BatchNorm2d(256))
        self.conv5.add_module('relu5_s1',nn.ReLU(inplace=True))
        #self.conv5.add_module('batch5_s1', nn.BatchNorm2d(256))
        # self.conv5.add_module('pool5_s1',nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv5.add_module('pool5_s1', nn.MaxPool2d(kernel_size=2, stride=2))
        # *************************** changed layers *********************** #

        self.fc6 = nn.Sequential()
        # self.fc6.add_module('fc6_s1', nn.Conv2d(256, 1024, kernel_size=3, stride=1, padding=1))
        self.fc6.add_module('fc6_s1', nn.Conv2d(256, 1024, kernel_size=2, stride=1, padding=1))
        self.fc6.add_module('batch6_s1', nn.BatchNorm2d(1024))
        self.fc6.add_module('relu6_s1', nn.ReLU(inplace=True))
        #self.fc6.add_module('batch6_s1', nn.BatchNorm2d(1024))
        self.fc6.add_module('pool5_s1', nn.MaxPool2d(kernel_size=2, stride=1))
        # self.fc6b = nn.Sequential()
        # # self.fc6b.add_module('conv6b_s1', nn.Conv2d(1024, 256, kernel_size=3, stride=2))
        # self.fc6b.add_module('conv6b_s1', nn.Conv2d(1024, 96, kernel_size=1, stride=1))
        # self.fc6b.add_module('batch6b_s1', nn.BatchNorm2d(96))
        # self.fc6b.add_module('relu6_s1', nn.ReLU(inplace=True))
        # self.fc6b.add_module('upsample', nn.Upsample(scale_factor=4))
        #
        # # self.fc6b.add_module('pool6b_s1', nn.MaxPool2d(kernel_size=3, stride=2))

        # ***********************************************************************#
        # U-NET #
        # self.uconnect= nn.Sequential()
        # self.uconnect.add_module('conv_sc', nn.Conv2d(2*96, 256, kernel_size=3, stride=2))
        # self.uconnect.add_module('batch_sc', nn.BatchNorm2d(256))
        # self.uconnect.add_module('relu_sc', nn.ReLU(inplace=True))
        # self.uconnect.add_module('pool_sc', nn.MaxPool2d(kernel_size=3, stride=2))
        # ***********************************************************************#

        # ***********************************************************************#
        # U-NET #
        self.uconnect1= nn.Sequential()
        self.uconnect1.add_module('conv', nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1))
        self.uconnect1.add_module('batch', nn.BatchNorm2d(256))
        self.uconnect1.add_module('relu', nn.ReLU(inplace=True))
        #self.uconnect1.add_module('batch', nn.BatchNorm2d(256))
        self.uconnect1.add_module('pool6b_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        #self.uconnect1.add_module('batch', nn.BatchNorm2d(256))
        #self.uconnect1.add_module('upsample', nn.Upsample(scale_factor=2))  # 256 * 30 * 30

        self.fc6c = nn.Sequential()
        # self.fc6c.add_module('fc7', nn.Linear(256*2*2, 512))
        self.fc6c.add_module('fc7', nn.Linear(256*7*7, 512))
        self.fc6c.add_module('relu7', nn.ReLU(inplace=True))
        #self.fc6c.add_module('relu7', nn.Sigmoid())
        #self.fc6c.add_module('batch', nn.BatchNorm1d(1))
        self.fc6c.add_module('drop7', nn.Dropout(p=dropout_rate))

        self.fc7_new = nn.Sequential()
        self.fc7_new.add_module('fc7', nn.Linear(self.num_inputs * 512, self.output_dim))
        self.fc7_new.add_module('relu7', nn.ReLU(inplace=True))
        #self.fc7_new.add_module('drop7', nn.Dropout(p=dropout_rate))

        self.classifier_new = nn.Sequential()
        self.classifier_new.add_module('fc8', nn.Linear(self.output_dim, classes))

    def load(self, checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)
        pretrained_dict = {k: v for k, v in list(pretrained_dict.items()) if k in model_dict and 'fc8' not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print([k for k, v in list(pretrained_dict.items())])

    def save(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

    def forward(self, x):
        if self.num_inputs == 1:
            x = x.unsqueeze(1)
        B, T, C, H = x.size()
        x = x.transpose(0, 1)
        x_list = []

        for i in range(self.num_inputs):
            curr_x = torch.unsqueeze(x[i], 1)
            if self.jigsaw:
                curr_x = curr_x.expand(-1, 3, -1, -1)
            if torch.cuda.is_available():
                input = torch.cuda.FloatTensor(curr_x.to(device))
            else:
                input = torch.FloatTensor(curr_x.to(device))


            out1 = self.conv1(input)
            out2 = self.conv2(out1)
            out3 = self.conv3(out2)
            out4 = self.conv4(out3)
            out5 = self.conv5(out4)
            out6 = self.fc6(out5)

            unet1 = self.uconnect1(out6)

            z = unet1.view([B, 1, -1])
            z = self.fc6c(z)
            z = z.view([B, 1, -1])
            x_list.append(z)

        x = torch.cat(x_list, 1)
        x = self.fc7_new(x.view(B, -1))
        pred = self.classifier_new(x)

        return pred

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                                        stride=1, padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                                        stride=1, padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)

            div = div.mul(self.alpha).add(1.0).pow(self.beta)

        x = x.div(div)
        return x


###
###     FUNCTIONS FOR IMAGE IMPORT
###

def set_contrast(image, contrast=1):
    """
    Set contrast in image

    :param image: input image
    :param contrast: contrast type
    :return: image with revised contrast
    """

    if contrast == 0:
        out_img = image
    elif contrast == 1:
        out_img = exposure.equalize_hist(image)
    elif contrast == 2:
        out_img = exposure.equalize_adapthist(image)
    elif contrast == 3:
        out_img = exposure.rescale_intensity(image)

    return out_img

IMAGE_PARAMS = {0: [2.1, 4, 4],
                1: [2.5, 3.2, 3.2]}  # factor to crop images by: crop ratio, translate_x, translate_y

def crop_image(image, param_num=0):
    """
    Crops image

    :param image: input image
    :param param_num: factor to crop by, 0 is less, 1 is more
    :return: cropped image
    """

    width = image.shape[1]
    height = image.shape[0]

    params = IMAGE_PARAMS[int(param_num)]

    new_dim = int(width // params[0])  # final image will be square of dim width/2 * width/2

    start_col = int(width // params[1])  # column (width position) to start from
    start_row = int(height // params[2])  # row (height position) to start from

    cropped_image = image[start_row:start_row + new_dim, start_col:start_col + new_dim]
    return cropped_image

def process_input_image(image, crop, output_dim=256, convert_grey=True):
    """
    Processes image: crop, convert to greyscale and resize

    :param image: image
    :param output_dim: dimensions to resize image to
    :param crop: True/False whether to crop image
    :param convert_grey: True/False whether to convert image to greyscale
    :return: formatted image
    """

    if convert_grey:
        image_grey = img_as_float(rgb2gray(image))
    else:
        image_grey = img_as_float(image)

    if crop:
        cropped_img = crop_image(image_grey)
    else:
        cropped_img = image_grey

    resized_img = transform.resize(cropped_img, output_shape=(output_dim, output_dim))
    my_img = set_contrast(resized_img) ## ultimately add contrast variable

    return my_img

def read_image_file(file_name, crop):

    """
    Reads in image file, crops it (if needed), converts it to greyscale, and returns image
    :param file_name: path and name of file
    :param crop: True/False if file needs to be cropped
    :return: Image
    """

    # PIL supported image types
    img_types = ("png", "jpg", "jpeg", "tiff", "bmp")

    if file_name[-3:].lower() in img_types:
        image_in = Image.open(file_name).convert('L')
        image_out = process_input_image(image_in, convert_grey=False, crop=crop)
        return image_out

    elif file_name[-4:].lower() in img_types:
        image_in = Image.open(file_name).convert('L')
        image_out = process_input_image(image_in, convert_grey=False, crop=crop)
        return image_out

    else:
        print("Unsure how to read in image file.")

# def get_new_filename(image_file,out_dir):
#     """
#     Creates a new file name (replaces extension with .png)
#     :param image_file: image file name
#     :return: png image file name
#     """
#     new_filename = image_file.split('.')[:-1] + '-preprocessed.png'
#     return new_filename

def get_grad_filename(side,view,date,patient,outdir):
    """
    Creates a new file name (replaces extension with .png)
    :param image_file: image file name
    :return: png image file name
    """

    new_filename = outdir + "/" + patient + "." + side + view + "." + date + '-GradCAM.png'
    return new_filename

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

def create_png_file(image_file,out_file,crop):
    """
    Reads in file name of image and saves it as a png file

    Function returns name of the png file

    :param image_file: image file name
    :param crop: True/False for if the image is already cropped
    :return: Saves png file and returns png file path + name
    """

    my_image = read_image_file(image_file, crop=crop)
    # plt.imshow(my_image)
    # plt.savefig(new_name)
    # im = Image.fromarray(my_image)
    # print(new_name)
    # im.save(new_name) ### FIGURE OUT WHY THIS IS BREAKING

    save_image(my_image, out_file)
    # imageio.imwrite(new_name, my_image)

    return my_image

###
### GRAD CAM FUNCTIONS
###

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, view):
        self.model = model
        self.gradients = None
        self.view = view

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        x = torch.unsqueeze(x, 0)
        conv_output = None

        B, T, C, H = x.size()
        x = x.transpose(0, 1)
        x_list = []

        for i in range(2):
            curr_x = torch.unsqueeze(x[i], 1)
            # curr_x = curr_x.expand(-1, 3, -1, -1)
            if torch.cuda.is_available():
                input = torch.cuda.FloatTensor(curr_x)
            else:
                input = torch.FloatTensor(curr_x)

            out1 = self.model.conv1(input)
            out2 = self.model.conv2(out1)
            out3 = self.model.conv3(out2)
            out4 = self.model.conv4(out3)
            out5 = self.model.conv5(out4)
            out6 = self.model.fc6(out5)

            unet1 = self.model.uconnect1(out6)

            if i == 0 and self.view == 'sag':
                unet1.register_hook(self.save_gradient)
                conv_output = unet1  # Save the convolution output on that layer

            elif i == 1 and self.view == 'trans':
                unet1.register_hook(self.save_gradient)
                conv_output = unet1  # Save the convolution output on that layer

            z = unet1.view([B, 1, -1])
            z = self.model.fc6c(z)
            z = z.view([B, 1, -1])
            x_list.append(z)

        x = torch.cat(x_list, 1)
        x = self.model.fc7_new(x.view(B, -1))
        x = self.model.classifier_new(x)

        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        return conv_output, x

class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, view):
        self.model = model.to(device)
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, view)

    def generate_cam(self, input_image, target_class=None):

        conv_output, model_output = self.extractor.forward_pass(input_image.to(device))
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        if torch.cuda.is_available():
            one_hot_output = torch.cuda.FloatTensor(1, model_output.size()[-1]).zero_()
        else:
            one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        # Zero grads

        self.model.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.cpu().numpy()[0]
        # Get convolution outputs
        target = conv_output.data.cpu().numpy()[0]
        # Get weights from gradients
        # weights = np.mean(guided_gradients, axis=(1, 2)) * -1  # Take averages for each gradient
        weights = np.mean(guided_gradients, axis=(1, 2))

        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0) ### this is relu
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize

        input_image = torch.unsqueeze(input_image[0], 0)
        input_image = torch.unsqueeze(input_image, 1)
        input_image = input_image.expand(-1, 3, -1, -1)
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        return cam

def get_example_params(image_path):
    """
        Gets used variables for almost all visualizations, like the image, model etc.

    Args:
        example_index (int): Image id to use from examples

    returns:
        original_image (numpy arr): Original image read from the file
        prep_img (numpy_arr): Processed image
        target_class (int): Target class for the image
        file_name_to_export (string): File name to export the visualizations
        pretrained_model(Pytorch model): Model to use for the operations
    """


    original_image = Image.open(image_path).convert('RGB')

    # return (image,
    #         original_image,
    #         target_class,
    #         filename,
    #         net)
    return original_image

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

def save_class_activation_images(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        # imageio image (numpy arr) : Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    # Grayscale activation map
    # heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'viridis')
    # for map in ['plasma', 'magma', 'inferno', 'viridis', 'jet']:
    #     for opacity in [0.5, 0.6]:
    map = 'inferno'
    opacity = 0.5
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, map, opacity) ### this will be more similar to papers
            # Save colored heatmap
            #path_to_file = os.path.join('../results', file_name+'_Cam_Heatmap.png')
            #save_image(heatmap, path_to_file)
            # Save heatmap on iamge
    path_to_file = file_name# + '_Cam_On_Image.png'

    # imsave(path_to_file, heatmap_on_image)
    save_image(heatmap_on_image, path_to_file)
    print("Heatmap on image written to: " + path_to_file)
            # Save grayscale heatmap
            #path_to_file = os.path.join('../results', file_name+'_Cam_Grayscale.png')
            #save_image(activation_map, path_to_file)

    return path_to_file

def apply_colormap_on_image(org_im, activation, colormap_name, opacity=0.4):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)

    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = opacity
    heatmap = np.transpose(heatmap, (1, 0, 2)) ########## added since Image grabs them in different order
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    # heatmap.show()
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))
    # no_trans_heatmap.show()

    # Apply heatmap on image
    heatmap_on_image = Image.new("RGBA", org_im.size)
    # heatmap_on_image.show()
    heatmap_on_image = Image.alpha_composite(heatmap_on_image.convert('RGBA'), org_im.convert('RGBA'))
    # heatmap_on_image.show()
    heatmap_on_image = Image.alpha_composite(heatmap_on_image.convert('RGBA'), heatmap.convert('RGBA'))
    # heatmap_on_image.show()

    return no_trans_heatmap, heatmap_on_image


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # orig_sag_img = dir_path + '/Orig-Sag-Img-Spaceholder.png'
    # orig_trans_img = dir_path + '/Orig-Trans-Img-Spaceholder.png'
    # grad_sag_img = dir_path + '/Grad-Sag-Img-Spaceholder.png'
    # grad_trans_img = dir_path + '/Grad-Trans-Img-Spaceholder.png'
    pth_file = dir_path + '/siam_checkpoint_18.pth'

    # orig_sag_img = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Front-end-test-files/Orig-Sag-Img-Spaceholder.png'
    # orig_trans_img = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Front-end-test-files/Orig-Trans-Img-Spaceholder.png'
    # grad_sag_img = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Front-end-test-files/Grad-Sag-Img-Spaceholder.png'
    # grad_trans_img = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Front-end-test-files/Grad-Trans-Img-Spaceholder.png'

    layout = [[sg.Text('Filename')],
              [sg.Text('Images folder: '), sg.Input(), sg.FolderBrowse()],
              [sg.Text('Patient list: '), sg.Input(), sg.FileBrowse()],
              [sg.Text('Image type (e.g. .jpg): '), sg.InputText()],
              [sg.Checkbox('Crop images?', default=False)],
              # [sg.Text('Network file (.pth): '), sg.Input(), sg.FileBrowse()],
              [sg.Text('Output path: '), sg.Input(), sg.FolderBrowse()],
              [sg.OK(), sg.Exit()]]
              # [sg.Text('_' * 80)],
              # [sg.Txt('Probability of surgery: ', size=[30, 1], key='output_prob')],
              # [sg.Image(orig_sag_img, size=[180, 180], key='orig_sag'), sg.Image(grad_sag_img, size=[180, 180], key='gc_sag')],
              # [sg.Image(orig_trans_img, size=[180, 180], key='orig_trans'), sg.Image(grad_trans_img, size=[180, 180], key='gc_trans')]]

    # layout = [[sg.Txt('Enter values to calculate')],
    #           [sg.In(size=(8, 1), key='numerator')],
    #           [sg.Txt('_' * 10)],
    #           [sg.In(size=(8, 1), key='denominator')],
    #           [sg.Txt('', size=(8, 1), key='output')],
    #           [sg.Button('Calculate', bind_return_key=True)]]

    window = sg.Window('PREHDICT', layout)
    # event, values = window.read()
    # window.close()

    while True:  # Event Loop
        event, values = window.read()
        # print(event, values)
        if event in (None, 'Exit'):
            break
        if event == 'OK':


            ##debug
            # sag_file = "C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Front-end-test-files/Capture-sag.png"
            # trans_file = "C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Front-end-test-files/Capture2-trans.png"

            pt_list = pd.read_csv(values[1])
            im_path = values[0]
            im_type = values[2]
            outdir = values[4]
            print(outdir)
            if not os.path.isdir(outdir):
                os.makedirs(outdir)

            out_list = pd.DataFrame({'Patient':[], 'Date':[], 'L_surg':[], 'R_surg':[]})

            for row in range(pt_list.shape[0]):

                patient = str(pt_list.Patients[row]).zfill(3)
                pt_date = str(pt_list.Date[row]).zfill(8)

                # print(patient)
                # print(pt_date)

                print("Evaluating patient:" + patient)

                r_prob = 0
                l_prob = 0

                for side in "R", "L":

                    sag_file = im_path + "/" + patient + "." + side + "S" + "." + pt_date + im_type
                    print(sag_file)
                    trans_file = im_path + "/" + patient + "." + side + "T" + "." + pt_date + im_type

                    ## debug
                    # crop = False
                    crop = values[3]

                    # test_image = read_image_file(sag_file)
                    # print(test_image)

                    if side == "R":
                        sag_png_file = outdir+"/"+ patient + "." + side + "S" + "." + pt_date +"_preprocessed.png"
                        sag_prepd_img = create_png_file(sag_file, out_file=sag_png_file, crop=crop)
                        out_trans = outdir+"/"+ patient + "." + side + "S" + "." + pt_date +"_preprocessed.png"
                        trans_prepd_img = create_png_file(trans_file, out_file=out_trans, crop=crop)
                    elif side == "L":
                        sag_png_file = outdir+"/"+ patient + "." + side + "S" + "." + pt_date +"_preprocessed.png"
                        sag_prepd_img = create_png_file(sag_file, out_file=sag_png_file, crop=crop)
                        trans_png_file = outdir+"/"+ patient + "." + side + "S" + "." + pt_date +"_preprocessed.png"
                        trans_prepd_img = create_png_file(trans_file, out_file=trans_png_file, crop=crop)

                    ##
                    ##      RUN MODEL AND GET GRAD CAM + PROBABILITY OF SURGERY
                    ##

                    # parser = argparse.ArgumentParser()
                    # parser.add_argument('--sag_path', required=True)
                    # parser.add_argument('--trans_path', required=True)
                    # parser.add_argument('--outdir', default="results")
                    # parser.add_argument('-checkpoint', default="./prehdict_20190802_vanilla_siamese_dim256_c1_checkpoint_18.pth")
                    # args = parser.parse_args()

                    CNN = PrehdictNet

                    # checkpoint = args.checkpoint
                    checkpoint = pth_file
                    net = CNN().to(device)

                    pretrained_dict = torch.load(checkpoint, map_location=torch.device('cpu'))['model_state_dict']

                    model_dict = net.state_dict()

                    # 1. filter out unnecessary keys
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

                    # 2. overwrite entries in the existing state dict
                    model_dict.update(pretrained_dict)
                    # 3. load the new state dict
                    net.load_state_dict(pretrained_dict)
                    # for file in files_to_get:
                    #     print(file)

                    # Get params
                    # outdir = args.outdir

                    #*************************
                    #*************************

                    # sag_path = args.sag_path.split("/")[-1].split(".")[0]
                    # trans_path = args.trans_path.split("/")[-1].split(".")[0]

                    # X = prep_single_sample.load_image(args.sag_path, args.trans_path)
                    X = np.array([sag_prepd_img, trans_prepd_img])

                    # img_path_sag = outdir + '/' + sag_path + '_preprocessed.jpg'
                    # img_path_trans = outdir + '/' + trans_path + '_preprocessed.jpg'

                    # scipy.misc.imsave(img_path_sag, X[0])
                    # scipy.misc.imsave(img_path_trans, X[1])

                    X[0] = torch.from_numpy(X[0]).float()
                    X[1] = torch.from_numpy(X[1]).float()
                    combined_image = torch.from_numpy(X).float()
                    net.eval()
                    softmax = torch.nn.Softmax(dim=1)

                    with torch.no_grad():
                        net.zero_grad()
                        output = net(torch.unsqueeze(combined_image, 0).to(device))
                        output_softmax = softmax(output)
                        pred_prob = output_softmax[:, 1]

                        ## calibration, currently hard coded
                        scaled_pred = scipy.special.expit(-4.245+(12.253*pred_prob))
                    ## incorporate calibration in this at some point

                    if side == "R":
                        r_prob = float(scaled_pred)
                        print("Right prob: " + str(r_prob))
                    elif side == "L":
                        l_prob = float(scaled_pred)
                        print("Left prob: " + str(l_prob))

                    # print("Probability of surgery:::{:6.3f}".format(float(pred_prob)))
                    print("Probability of surgery:::{:6.3f}".format(float(scaled_pred)))
                    # target_class = 1 if pred_prob >= 0.5 else 0
                    target_class = 1 if scaled_pred >= 0.05 else 0

                    original_image_sag = get_example_params(sag_file)
                    original_image_trans = get_example_params(trans_file)

                    # Grad cam


                    sag_filename_id = os.path.join(outdir, get_grad_filename(side=side, view="S", patient=patient, date=pt_date, outdir=outdir))
                    trans_filename_id = os.path.join(outdir, get_grad_filename(side=side, view="T", patient=patient, date=pt_date, outdir=outdir))

                    grad_cam = GradCam(net, view='sag')
                    # Generate cam mask
                    cam = grad_cam.generate_cam(combined_image, target_class)
                    # Save mask
                    # sag_cam_img_file = save_class_activation_images(original_image_sag, cam, sag_filename_id)

                    ## MAKE SAVE TO outdir
                    sag_cam_img_file = save_class_activation_images(Image.fromarray(np.uint8(cm.gist_earth(sag_prepd_img)*255)),
                                                                    cam, sag_filename_id)

                    grad_cam = GradCam(net, view='trans')
                    # Generate cam mask
                    cam = grad_cam.generate_cam(combined_image, target_class)
                    # Save mask
                    # trans_cam_img_file = save_class_activation_images(original_image_trans, cam, trans_filename_id)
                    trans_cam_img_file = save_class_activation_images(Image.fromarray(np.uint8(cm.gist_earth(trans_prepd_img)*255)), cam, trans_filename_id)

                out_list = out_list.append({'Patient': patient, 'Date': pt_date, 'L_surg': l_prob, 'R_surg': r_prob}, ignore_index=True)
                print(out_list)

            print(out_list)
            out_list.to_csv(outdir + "/Predicted_probabilities_" + time.strftime("%Y%m%d%H%M%S") + ".csv")
            ##
            ##     GUI OUTPUT FROM MODEL
            ##

            # sag_img = sg.Image(sag_png_file)
            # trans_img = sg.Image(trans_png_file)
            #
            # sag_gc_img = sg.Image(sag_cam_img_file)
            # trans_gc_img = sg.Image(trans_cam_img_file)

            # Update the "output" text element to be the value of "input" element
            # window['output_prob'].update("Probability of surgery: " + str(round(float(pred_prob), 3)))
            # window['output_prob'].update("Probability of surgery: " + str(round(float(scaled_pred), 3)))
            # window['orig_sag'].update(sag_png_file).set_size([180, 180])
            # window['orig_trans'].update(trans_png_file).set_size([180, 180])
            # window['gc_sag'].update(sag_cam_img_file).set_size([180, 180])
            # window['gc_trans'].update(trans_cam_img_file).set_size([180, 180])
            # window['orig_sag'].update(sag_png_file, size=[180, 180])
            # window['orig_trans'].update(trans_png_file, size=[180, 180])
            # window['gc_sag'].update(sag_cam_img_file, size=[180, 180])
            # window['gc_trans'].update(trans_cam_img_file, size=[180, 180])

    window.close()

    # output_form = [[sg.Text("Probability of surgery: " + str(float(pred_prob)))],
    #                [sg.Text("Sag Image: ")], [sag_img], [sag_gc_img],
    #                [sg.Text("Trans Image: ")], [trans_img], [trans_gc_img],
    #                [sg.OK()], [sg.Cancel()]]
    #
    # output_window = sg.Window('Input images', output_form)
    #
    # out_event = output_window.read()
    # output_window.close()

    # sg.Popup("Input images: ",
    #          'Sagittal: ', sag_img,
    #          'Transverse: ', trans_img)


if __name__ == "__main__":
    main()


