import pandas as pd
from PIL import Image
from skimage import transform

output_dim = 256
contrast = 1
    ### script params
img_dir = "/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/all-jpgs-lparty/"
out_dir = "/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/img-lbl-jpgs/"
# label_file = "/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/view_label_df_20191120.csv"
label_file = "C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Image organization Nov 2019/view_label_df_20191120.csv"
# linking_log = "/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/linking_log_20191120.csv"
linking_log = "C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Image organization Nov 2019/linking_log_20191120.csv"

IMAGE_PARAMS = {0: [2.1, 4, 4],
                1: [2.5, 3.2, 3.2]}  # factor to crop images by: crop ratio, translate_x, translate_y


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


## CREATE DIFFERENT FOLDER FOR EACH LABEL

## DATALOADER FOR EACH LABEL?

#def set_up_data
labels = pd.read_csv(label_file)
labels.index = labels.image_name
lin_log = pd.read_csv(linking_log)
lin_log.index = lin_log.mrn

files_out = []
labels_out = []

for file_name in labels.image_name:
    image_in = Image.open(file_name)
    image_grey = img_as_float(rgb2gray(image_in))
    cropped_img = crop_image(image_grey)
    resized_img = transform.resize(cropped_img, output_shape=(output_dim, output_dim))
    image_out = set_contrast(resized_img, contrast)

    mrn = file_name.split("_")[0]
    new_id = lin_log.at[int(mrn), 'deid']
    new_img_name = str(new_id) + "_" + "_".join(file_name.split("_")[1:])
    files_out.append(new_img_name)
    labels_out.append(labels.at[file_name,'revised_labels'])

    ###
    ### MAIN FUNCTION
    ###

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_folder", default="./")
    parser.add_argument("--out_folder", default="./")
    parser.add_argument("--out_filename", default="data_sheet.csv")
    parser.add_argument("--crop", default=0, type=int, help="Crop setting (0=big, 1=tight)")
    parser.add_argument("--contrast", default=1, type=int, help="Image contrast to train on")
    args = parser.parse_args()

    print("ARGS" + '\t' + str(args))

    my_nn = MyCNN


if __name__ == "__main__":
    main()


















