
import numpy as np
import pydicom
from skimage.color import rgb2gray
from skimage import exposure
from skimage import img_as_float
from skimage import transform
from PIL import Image
from matplotlib.pyplot import imread

IMAGE_PARAMS = {0: [1.9, 6, 6],
                1: [2.5, 3.2, 3.2]}

def crop_image(image, param_num):
    width = image.shape[1]
    height = image.shape[0]

    params = IMAGE_PARAMS[int(param_num)]

    new_dim = int(width // params[0])  # final image will be square of dim width/2 * width/2

    start_col = int(width // params[1])  # column (width position) to start from
    start_row = int(height // params[2])  # row (height position) to start from

    cropped_image = image[start_row:start_row+new_dim, start_col:start_col+new_dim]
    return cropped_image

def load_image(sag_path, trans_path, output_dim=256):
    X = []
    for image_path in [sag_path, trans_path]:
        print(image_path)
        img = img_as_float(rgb2gray(imread(image_path)))
        cropped_image = crop_image(img, 0)  # IMAGE_PARAMS 0
        # cropped_image = img
        resized_image = transform.resize(cropped_image, output_shape=(output_dim, output_dim))
        image = resized_image.reshape((output_dim, output_dim))
        X.append(exposure.equalize_hist(image))
    return np.array(X)

