import PySimpleGUI as sg
from PIL import Image
import pydicom
import imageio
from skimage.color import rgb2gray
from skimage import img_as_float
from skimage import transform
from skimage import exposure
from skimage.io import imsave
import matplotlib.cm as mpl_color_map
import numpy as np
import torch
import os
import copy

import Prehdict_CNN
CNN = Prehdict_CNN.PrehdictNet

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

# import scipy
# import argparse
#
# from misc_functions import get_example_params, save_class_activation_images
#
# import prep_single_sample


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

def process_input_image(image, output_dim=256, crop=True, convert_grey=True):
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
    image_out = set_contrast(resized_img) ## ultimately add contrast variable

    return image_out

def read_image_file(file_name, crop):

    """
    Reads in image file, crops it (if needed), converts it to greyscale, and returns image
    :param file_name: path and name of file
    :param crop: True/False if file needs to be cropped
    :return: Image
    """

    # PIL supported image types
    img_types = ("png", "jpg", "jpeg", "tiff", "bmp")

    if file_name[-3:] == 'dcm':
        my_dcm = pydicom.dcmread(file_name)
        image_in = my_dcm.pixel_array
        image_out = process_input_image(image_in, crop=crop)

    elif file_name[-3:] in img_types:
        image_in = Image.open(file_name).convert('L')
        # pic = imageio.imread(file_name)
        # gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
        # image_in = gray(pic)
        image_out = process_input_image(image_in, convert_grey=False, crop=crop)

    elif file_name[-4:] in img_types:
        image_in = Image.open(file_name).convert('L')
        # pic = imageio.imread(file_name)
        # gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
        # image_in = gray(pic)
        image_out = process_input_image(image_in, convert_grey=False, crop=crop)

    else:
        print("Unsure how to read in image file.")

    return image_out

def get_new_filename(image_file):
    """
    Creates a new file name (replaces extension with .png)
    :param image_file: image file name
    :return: png image file name
    """

    new_filename = '.'.join(image_file.split('.')[:-1]) + '.png'
    return new_filename

def create_png_file(image_file,crop):
    """
    Reads in file name of image and saves it as a png file

    Function returns name of the png file

    :param image_file: image file name
    :param crop: True/False for if the image is already cropped
    :return: Saves png file and returns png file path + name
    """

    my_image = read_image_file(image_file,crop=crop)
    new_name = get_new_filename(image_file)
    # plt.imshow(my_image)
    # plt.savefig(new_name)
    # im = Image.fromarray(my_image)
    # print(new_name)
    # im.save(new_name) ### FIGURE OUT WHY THIS IS BREAKING

    imsave(new_name, my_image)
    # imageio.imwrite(new_name, my_image)

    return new_name, my_image

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
    path_to_file = '{}_Cam_On_Image_{}.png'.format(file_name, map)

    # imsave(path_to_file, heatmap_on_image)
    save_image(heatmap_on_image, path_to_file)
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
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)

    return no_trans_heatmap, heatmap_on_image


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    orig_sag_img = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Front-end-test-files/Orig-Sag-Img-Spaceholder.png'
    orig_trans_img = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Front-end-test-files/Orig-Trans-Img-Spaceholder.png'
    grad_sag_img = dir_path + '/Grad-Sag-Img-Spaceholder.png'
    grad_trans_img = dir_path + '/Grad-Trans-Img-Spaceholder.png'

    # orig_sag_img = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Front-end-test-files/Orig-Sag-Img-Spaceholder.png'
    # orig_trans_img = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Front-end-test-files/Orig-Trans-Img-Spaceholder.png'
    # grad_sag_img = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Front-end-test-files/Grad-Sag-Img-Spaceholder.png'
    # grad_trans_img = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Front-end-test-files/Grad-Trans-Img-Spaceholder.png'

    layout = [[sg.Text('Filename')],
              [sg.Text('Sagittal: '), sg.Input(), sg.FileBrowse()],
              [sg.Text('Transverse: '), sg.Input(), sg.FileBrowse()],
              [sg.Checkbox('Crop images?', default=False)],
              [sg.Text('Network file (.pth): '), sg.Input(), sg.FileBrowse()],
              [sg.Text('Output path: '), sg.Input(), sg.FolderBrowse()],
              [sg.OK(), sg.Exit()],
              [sg.Text('_' * 80)],
              [sg.Txt('Probability of surgery: ', size=[30, 1], key='output_prob')],
              [sg.Image(orig_sag_img, size=[180, 180], key='orig_sag'), sg.Image(grad_sag_img, size=[180, 180], key='gc_sag')],
              [sg.Image(orig_trans_img, size=[180, 180], key='orig_trans'), sg.Image(grad_trans_img, size=[180, 180], key='gc_trans')]]

    # layout = [[sg.Txt('Enter values to calculate')],
    #           [sg.In(size=(8, 1), key='numerator')],
    #           [sg.Txt('_' * 10)],
    #           [sg.In(size=(8, 1), key='denominator')],
    #           [sg.Txt('', size=(8, 1), key='output')],
    #           [sg.Button('Calculate', bind_return_key=True)]]

    window = sg.Window('Persistent open window', layout)
    event, values = window.read()
    # window.close()

    while True:  # Event Loop
        event, values = window.read()
        print(event, values)
        if event in (None, 'Exit'):
            break
        if event == 'OK':


            sag_file = values[0]
            trans_file = values[1]

            crop = values[2]

            # test_image = read_image_file(sag_file)
            # print(test_image)

            sag_png_file, sag_prepd_img = create_png_file(sag_file,crop=crop)
            trans_png_file, trans_prepd_img = create_png_file(trans_file,crop=crop)

            ##
            ##      RUN MODEL AND GET GRAD CAM + PROBABILITY OF SURGERY
            ##

            # parser = argparse.ArgumentParser()
            # parser.add_argument('--sag_path', required=True)
            # parser.add_argument('--trans_path', required=True)
            # parser.add_argument('--outdir', default="results")
            # parser.add_argument('-checkpoint', default="./prehdict_20190802_vanilla_siamese_dim256_c1_checkpoint_18.pth")
            # args = parser.parse_args()

            # checkpoint = args.checkpoint
            checkpoint = values[3]
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
            outdir = values[4] ## MAKE SAME AS INPUT DIR
            if not os.path.isdir(outdir):
                os.makedirs(outdir)

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
            ## incorporate calibration in this at some point


            print("Probability of surgery:::{:6.3f}".format(float(pred_prob)))
            target_class = 1 if pred_prob >= 0.5 else 0

            original_image_sag = get_example_params(sag_file)
            original_image_trans = get_example_params(trans_file)

            # Grad cam

            sag_filename_id = os.path.join(outdir, 'sag')
            trans_filename_id = os.path.join(outdir, 'trans')

            grad_cam = GradCam(net, view='sag')
            # Generate cam mask
            cam = grad_cam.generate_cam(combined_image, target_class)
            # Save mask
            sag_cam_img_file = save_class_activation_images(original_image_sag, cam, sag_filename_id)

            grad_cam = GradCam(net, view='trans')
            # Generate cam mask
            cam = grad_cam.generate_cam(combined_image, target_class)
            # Save mask
            trans_cam_img_file = save_class_activation_images(original_image_trans, cam, trans_filename_id)


            ##
            ##     GUI OUTPUT FROM MODEL
            ##

            # sag_img = sg.Image(sag_png_file)
            # trans_img = sg.Image(trans_png_file)
            #
            # sag_gc_img = sg.Image(sag_cam_img_file)
            # trans_gc_img = sg.Image(trans_cam_img_file)

            # Update the "output" text element to be the value of "input" element
            window['output_prob'].update("Probability of surgery: " + str(round(float(pred_prob), 3)))
            window['orig_sag'].update(sag_png_file, size=[180, 180])
            window['orig_trans'].update(trans_png_file, size=[180, 180])
            window['gc_sag'].update(sag_cam_img_file, size=[180, 180])
            window['gc_trans'].update(trans_cam_img_file, size=[180, 180])

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


