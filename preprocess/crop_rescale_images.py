
import os
import pydicom
from skimage.color import rgb2gray
from skimage import exposure
from skimage import img_as_float
import skimage.io as io
import matplotlib.pyplot as plt

rootdir = '/home/martaskreta/Desktop/CSC2516/all_kidney_images_hashed/'

# loads all image paths to array
def load_file_names():
    dcm_files = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if ".dcm" in file.lower():
                dcm_files.append(os.path.join(subdir, file))
    return dcm_files

def crop_image(image):
    width = image.shape[1]
    height = image.shape[0]

    new_dim = int(width // 2)  # final image will be square of dim width/2 * width/2

    start_col = width // 4  # column (width position) to start from
    start_row = height // 5  # row (height position) to start from
    print(new_dim)

    cropped_image = image[start_row:start_row+new_dim, start_col:start_col+new_dim]
    print(cropped_image[100:150,100:150])
    return cropped_image

def load_images(dcm_files):
    for image_path in dcm_files:
        ds = pydicom.dcmread(image_path)
        img = ds.pixel_array
        img = img_as_float(rgb2gray(img))
        print('\U0001F4A5')
        plt.figure()
        cropped_image = crop_image(img)
        plt.subplot(2,2,1)
        plt.imshow(cropped_image, cmap='gray')
        rescaled_cropped_image = exposure.equalize_hist(cropped_image)
        plt.subplot(2, 2, 2)
        plt.imshow(rescaled_cropped_image, cmap='gray')
        rescaled_cropped_image = exposure.equalize_adapthist(cropped_image)
        plt.subplot(2, 2, 3)
        plt.imshow(rescaled_cropped_image, cmap='gray')
        rescaled_cropped_image = exposure.rescale_intensity(cropped_image)
        plt.subplot(2, 2, 4)
        plt.imshow(rescaled_cropped_image, cmap='gray')

        plt.show()


dcm_files = load_file_names()
load_images(dcm_files[0:10])
load_images(dcm_files[170:180])

