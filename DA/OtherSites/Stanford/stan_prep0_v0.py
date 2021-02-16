import argparse
import os
import pydicom
from PIL import Image
import json
from skimage.color import rgb2gray
from skimage import img_as_float, exposure
import numpy as np
from skimage.transform import resize

def get_pt_scan_ids(fileroot):
    pt_scan_split = fileroot.split("_")[2].split("-")
    pt_id = pt_scan_split[0]
    scan_id = pt_scan_split[1]

    return pt_id, scan_id

def get_image_files(img_dir):
    dcm_files = []
    for subdir, dirs, files in os.walk(img_dir):
        for file in files:
            if file.lower()[-4:] == ".dcm":
                dcm_files.append(os.path.join(subdir, file))
    return sorted(dcm_files)

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


def process_input_image(image, set_bw=False):
    """
    Processes image: crop, convert to greyscale and resize

    :param image: image
    :return: formatted image
    """

    if set_bw:
        image_grey = img_as_float(rgb2gray(image))
    else:
        image_grey = img_as_float(image)
    my_img = set_contrast(image_grey) ## ultimately add contrast variable
    fit_img = autocrop(my_img, threshold=0.75)
    # fit_img = my_img

    return fit_img

def prep_img_files(img_list, in_dir, out_dir):
    pt_id = []
    scan_id = []
    manu_list = []
    manu_model = []
    number_list = []
    my_img_list = []
    view_name_list = []

    for img in img_list:

        my_dcm = pydicom.dcmread(img)

        try:
            my_pt_id = my_dcm.PatientID
        except AttributeError:
            print("patient ID not found")
            my_pt_id = "NA"
        # pt_id = pt_id.append(my_pt_id)
        pt_id.append(my_pt_id)

        try:
            my_acc_num = my_dcm.AccessionNumber
        except AttributeError:
            print("Accession number not found in the dicom")
            my_acc_num = "NA"
        # scan_id = scan_id.append(my_acc_num)
        scan_id.append(my_acc_num)

        try:
            number = my_dcm.InstanceNumber
        except AttributeError:
            number = "NA"
        # number_list = number_list.append(number)
        number_list.append(number)

        out_file_name = str(my_pt_id) + "_" + str(my_acc_num) + "_" + str(number)
        print(out_file_name)


## FIGURE OUT HOW TO DEAL WITH MONOCHROME AND WHATEVER OTHER ERRORS ARE GOING ON

        # try:
        my_img_array = my_dcm.pixel_array
        if len(my_dcm.pixel_array.shape) == 4:
            my_img = Image.fromarray(my_dcm.pixel_array[1, :, :, :])
            print("Pixel array converted to image from array")
            my_img = process_input_image(my_img, set_bw=True)
            print("Image processed")
            Image.fromarray(my_img.astype(np.uint8)).save(out_dir + "/" + out_file_name + ".jpg")
            print("Image saved")
        else:
            if my_dcm.pixel_array.shape[2] > 3:
                print("Image currently not recoverable in dicom: " + in_dir + "/" + img)
                print("Img shape: " + str(my_dcm.pixel_array.shape))
                my_img_array = "NA"
            else:
                my_img = Image.fromarray(my_dcm.pixel_array)#.resize((256, 256))
                print("Pixel array converted to image from array")
                my_img = process_input_image(my_img, set_bw=True)
                print("Image processed")
                Image.fromarray(my_img.astype(np.uint8)).save(out_dir + "/" + out_file_name + ".jpg")
                print("Image saved")
    # except AttributeError:
    #         print("No image found in dicom: " + in_dir + "/" + img)
    #         my_img_array = "NA"
    #     # my_img_list = my_img_list.append(my_img_array)
    #     my_img_list.append(my_img_array)

        try:
            manu = my_dcm.Manufacturer
        except AttributeError:
            manu = "NA"
        # manu_list = manu_list.append(manu)
        manu_list.append(manu)

        try:
            model = my_dcm.ManufacturerModelName
        except AttributeError:
            model = "NA"
        # manu_model = manu_model.append(model)
        manu_model.append(model)

        try:
            my_view = my_dcm.ViewName
        except AttributeError:
            print("View not found")
            my_view = "NA"
        # view_name_list = view_name_list.append(my_view)
        view_name_list.append(my_view)

    scan_dict = {"pt_id": pt_id,
                 "scan_id": scan_id,
                 "manu_list": manu_list,
                 "manu_model": manu_model,
                 "number_list": number_list,
                 "img_list": img_list,
                 "view_name_list": view_name_list}

    # using solution from Peter Mortensen from https://stackoverflow.com/questions/7100125/storing-python-dictionaries
    with open('C:/Users/lauren erdman/Desktop/kidney_img/HN/Stanford/Datasheets/'+ my_pt_id + "_" + my_acc_num +".json", 'w') as fp:
        json.dump(scan_dict, fp, sort_keys=True, indent=4)

    ## To open:
    # with open('data.json', 'r') as fp:
    #     data = json.load(fp)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-img_dir', default='C:/Users/lauren erdman/Desktop/kidney_img/HN/Stanford/Full_dataset/my_dcm/', help="directory of US sequence dicoms")
    parser.add_argument('-out_dir', default='C:/Users/lauren erdman/Desktop/kidney_img/HN/Stanford/std_jpgs0/', help="where to write jpgs")
    # parser.add_argument('-fileroot', default='NULL', help="directory of US sequence dicoms")

    opt = parser.parse_args()

    img_list = get_image_files(img_dir=opt.img_dir)

    prep_img_files(img_list=img_list, in_dir=opt.img_dir, out_dir=opt.out_dir)

if __name__ == "__main__":
    main()
