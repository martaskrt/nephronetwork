import argparse
import os
import pydicom
from PIL import Image
import json
from skimage.color import rgb2gray
from skimage import img_as_float, exposure
import numpy as np
from skimage.transform import resize
import math
import cv2
import random

def get_image_files(img_dir):
    img_files = []
    for subdir, dirs, files in os.walk(img_dir):
        for file in files:
            if file.lower()[-4:] == ".jpg":
                img_files.append(os.path.join(subdir, file))
    return sorted(img_files)

def set_contrast(image, contrast=2):
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
    else:
        out_img = None
        print("No valid contrast option provided. \n"
              "Please specify one of the numbers 1-4. \n")

    return out_img


def pad_img_le(image_in, dim=256, random_pad = False):
    """

    :param image_in: input image
    :param dim: desired dimensions of padded image (should be bigger or as big as all input images)
    :return: padded image

    """

    im_shape = image_in.shape

    while im_shape[0] > dim or im_shape[1] > dim:
        image_in = resize(image_in, ((im_shape[0]*4)//5, (im_shape[1]*4)//5), anti_aliasing=True)
        im_shape = image_in.shape

    # print(im_shape)
    if random_pad:

        if random.random() >= 0.5:
            image_in = cv2.flip(image_in, 1)

        rand_h = np.random.uniform(0, 1, 1)
        rand_v = np.random.uniform(0, 1, 1)
        right = math.floor((dim - im_shape[1])*rand_h)
        left = math.ceil((dim - im_shape[1])*(1-rand_h))
        bottom = math.floor((dim - im_shape[0])*rand_v)
        top = math.ceil((dim - im_shape[0])*(1-rand_v))
    else:
        right = math.floor((dim - im_shape[1])/2)
        left = math.ceil((dim - im_shape[1])/2)
        bottom = math.floor((dim - im_shape[0])/2)
        top = math.ceil((dim - im_shape[0])/2)

    image_out = cv2.copyMakeBorder(image_in, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    return image_out


def autocrop(image, threshold=0):
    """Crops any edges below or equal to threshold

    from fviktor here: https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
    Crops blank image to 1x1.

    Returns cropped image.

    """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image


def process_input_image(img_file, set_bw=True):
    """
    Processes image: crop, convert to greyscale and resize

    :param image: image
    :return: formatted image
    """
    image = Image.open(img_file).convert('L')

    fit_img = pad_img_le(set_contrast(np.array(image)))

    cv2.imwrite(img_file, fit_img*255)
    # out_img.save(img_file)
    # fit_img = my_img

    # return fit_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-img_dir', default='C:/Users/lauren erdman/Desktop/kidney_img/View_Labeling/images/all_lab_img/', help="directory of US sequence dicoms")
    # parser.add_argument('-out_dir', default='C:/Users/lauren erdman/Desktop/kidney_img/HN/Stanford/std_jpgs0/', help="where to write jpgs")
    # parser.add_argument('-fileroot', default='NULL', help="directory of US sequence dicoms")

    opt = parser.parse_args()

    img_list = get_image_files(opt.img_dir)

    for img_filename in img_list:
        try:
            process_input_image(img_file=img_filename)
        except IndexError:
            pass

if __name__ == "__main__":
    main()