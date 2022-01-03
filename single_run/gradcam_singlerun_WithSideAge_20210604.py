from PIL import Image
import numpy as np
import torch
import os
import copy
import argparse
from torch import nn
from skimage import exposure
from skimage import img_as_float, transform
import matplotlib.cm as mpl_color_map


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


def format_np_output(np_arr):
    """
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


def save_class_activation_images(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
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
    save_image(heatmap_on_image, path_to_file)
            # SAve grayscale heatmap
            #path_to_file = os.path.join('../results', file_name+'_Cam_Grayscale.png')
            #save_image(activation_map, path_to_file)


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


def special_ST_preprocessing(img_file, output_dim=256):

    image = np.array(Image.open(img_file).convert('L'))
    image_grey = img_as_float(image)

    # if crop:
    #     cropped_img = crop_image(image_grey)
    # else:
    #     cropped_img = image_grey
    cropped_img = image_grey

    resized_img = transform.resize(cropped_img, output_shape=(output_dim, output_dim))
    my_img = set_contrast(resized_img) ## ultimately add contrast variable

    # img_name = img_file.split('/')[len(img_file.split('/')) - 1]
    # img_folder = "/".join(img_file.split('/')[:-2])
    # out_img_filename = img_folder + "/" + img_name.split('.')[0] + "-preprocessed.png"
    # # out_img_filename = 'C:/Users/lauren erdman/Desktop/kidney_img/img_debugging/' + img_name.split('.')[0] + "-preprocessed.png"
    # Image.fromarray(my_img * 255).convert('RGB').save(out_img_filename)
    # my_img = np.array(Image.open(out_img_filename).convert('L'))

    return my_img


def load_image(sag_path, trans_path, output_dim=256):
    X = []
    for image_path in [sag_path, trans_path]:
        print(image_path)
        img = special_ST_preprocessing(image_path, output_dim=output_dim)
        X.append(img)
    return np.array(X)


class SiamNet(nn.Module):
    def __init__(self, classes=2, num_inputs=2, dropout_rate=0.5, output_dim=256, cov_layers=True):
        super(SiamNet, self).__init__()

        self.cov_layers = cov_layers
        self.output_dim = output_dim
        # print("LL DIM: " + str(self.output_dim))
        self.num_inputs = num_inputs

        self.conv = nn.Sequential()
        self.conv.add_module('conv1_s1', nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=0))
        self.conv.add_module('batch1_s1', nn.BatchNorm2d(96))
        self.conv.add_module('relu1_s1', nn.ReLU(inplace=True))
        self.conv.add_module('pool1_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv.add_module('lrn1_s1', LRN(local_size=5, alpha=0.0001, beta=0.75))

        self.conv.add_module('conv2_s1', nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2))
        self.conv.add_module('batch2_s1', nn.BatchNorm2d(256))
        self.conv.add_module('relu2_s1', nn.ReLU(inplace=True))
        self.conv.add_module('pool2_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv.add_module('lrn2_s1', LRN(local_size=5, alpha=0.0001, beta=0.75))

        self.conv.add_module('conv3_s1', nn.Conv2d(256, 384, kernel_size=3, padding=1))
        self.conv.add_module('batch3_s1', nn.BatchNorm2d(384))
        self.conv.add_module('relu3_s1', nn.ReLU(inplace=True))

        self.conv.add_module('conv4_s1', nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2))
        self.conv.add_module('batch4_s1', nn.BatchNorm2d(384))
        self.conv.add_module('relu4_s1', nn.ReLU(inplace=True))

        self.conv.add_module('conv5_s1', nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2))
        self.conv.add_module('batch5_s1', nn.BatchNorm2d(256))
        self.conv.add_module('relu5_s1', nn.ReLU(inplace=True))
        self.conv.add_module('pool5_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        # *************************** changed layers *********************** #
        self.fc6 = nn.Sequential()
        # self.fc6.add_module('fc6_s1', nn.Conv2d(256, 1024, kernel_size=3, stride=1, padding=1))
        self.fc6.add_module('fc6_s1', nn.Conv2d(256, 1024, kernel_size=2, stride=1, padding=1))
        # self.fc6.add_module('fc6_s1', nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=1))
        self.fc6.add_module('batch6_s1', nn.BatchNorm2d(1024))
        self.fc6.add_module('relu6_s1', nn.ReLU(inplace=True))

        self.fc6b = nn.Sequential()
        self.fc6b.add_module('conv6b_s1', nn.Conv2d(1024, 256, kernel_size=3, stride=2))
        self.fc6b.add_module('batch6b_s1', nn.BatchNorm2d(256))
        self.fc6b.add_module('relu6_s1', nn.ReLU(inplace=True))
        self.fc6b.add_module('pool6b_s1', nn.MaxPool2d(kernel_size=3, stride=2))

        self.fc6c = nn.Sequential()
        # self.fc6c.add_module('fc7', nn.Linear(256*2*2, 512))
        self.fc6c.add_module('fc7', nn.Linear(256 * 3 * 3, 512))
        self.fc6c.add_module('relu7', nn.ReLU(inplace=True))
        # self.fc6c.add_module('drop7', nn.Dropout(p=dropout_rate))

        self.fc7_new = nn.Sequential()
        self.fc7_new.add_module('fc7', nn.Linear(self.num_inputs * 512, self.output_dim))
        self.fc7_new.add_module('relu7', nn.ReLU(inplace=True))
        # self.fc7_new.add_module('drop7', nn.Dropout(p=dropout_rate))

        self.classifier_new = nn.Sequential()
        self.classifier_new.add_module('fc8', nn.Linear(self.output_dim, classes))

        if self.cov_layers:
            self.classifier_new.add_module('relu8', nn.ReLU(inplace=True))

            self.add_covs1 = nn.Sequential()
            self.add_covs1.add_module('fc9', nn.Linear(classes + 2, classes + 126))
            self.add_covs1.add_module('relu9', nn.ReLU(inplace=True))

            self.add_covs2 = nn.Sequential()
            self.add_covs2.add_module('fc10', nn.Linear(classes + 126, classes))

    def load(self, checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)
        pretrained_dict = {k: v for k, v in list(pretrained_dict.items()) if k in model_dict and 'fc8' not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print([k for k, v in list(pretrained_dict.items())])

    def save(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

    def forward(self, x, age, left):

        if self.num_inputs == 1:
            x = x.unsqueeze(1)
        #   B, C, H = x.size()
        # else:
        B, T, C, H = x.size()
        x = x.transpose(0, 1)
        x_list = []
        for i in range(self.num_inputs):
            if self.num_inputs == 1:
                curr_x = torch.unsqueeze(x[i], 1)
            else:
                curr_x = torch.unsqueeze(x[i], 1)
            # if self.num_inputs == 1:
            #   curr_x = curr_x.expand(-1, 3, -1)
            # else:
            curr_x = curr_x.expand(-1, 3, -1, -1) ## expanding 1 channel to 3 duplicate channels
            if torch.cuda.is_available():
                input = torch.cuda.FloatTensor(curr_x.to(device))
            else:
                input = torch.FloatTensor(curr_x.to(device))
            z = self.conv(input)
            z = self.fc6(z) ## convolution
            z = self.fc6b(z) ## convolution
            z = z.view([B, 1, -1])
            z = self.fc6c(z) ## fully connected layer
            ### LAUREN CHECK THIS -- shouldn't need to .view, no?
            # z = z.view([B, 1, -1])
            x_list.append(z)

        x = torch.cat(x_list, 1)
        # x = torch.sum(x, 1)
        x = self.fc7_new(x.view(B, -1))
        pred = self.classifier_new(x)

        if self.cov_layers:
            age = torch.tensor(age).type(torch.FloatTensor).to(device).view(B, 1)
            # print("Age: ")
            # print(age)
            side = torch.tensor(left).type(torch.FloatTensor).to(device).view(B, 1)
            # print("Side: ")
            # print(side)
            mid_in = torch.cat((pred, age, side), 1)

            x = self.add_covs1(mid_in)
            pred = self.add_covs2(x)

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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    def forward_pass_on_convolutions(self, x, age, left):
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
            curr_x = curr_x.expand(-1, 3, -1, -1)
            if torch.cuda.is_available():
                input = torch.cuda.FloatTensor(curr_x)
            else:
                input = torch.FloatTensor(curr_x)

            out1 = self.model.conv(input)

            # z = out1.view([B, 1, -1])
            z = self.model.fc6(out1) ## convolution
            z = self.model.fc6b(z) ## convolution

            if i == 0 and self.view == 'sag':
                z.register_hook(self.save_gradient)
                conv_output = z  # Save the convolution output on that layer

            elif i == 1 and self.view == 'trans':
                z.register_hook(self.save_gradient)
                conv_output = z  # Save the convolution output on that layer

            z = z.view([B, 1, -1])
            z = self.model.fc6c(z) ## fully connected layer
            ### LAUREN CHECK THIS -- shouldn't need to .view, no?
            # z = z.view([B, 1, -1])
            x_list.append(z)

        x = torch.cat(x_list, 1)
        x = self.model.fc7_new(x.view(B, -1))
        pred = self.model.classifier_new(x)

        age = torch.tensor(age).type(torch.FloatTensor).to(device).view(B, 1)
        left = torch.tensor(left).type(torch.FloatTensor).to(device).view(B, 1)

        mid_in = torch.cat((pred, age, left), 1)

        x = self.model.add_covs1(mid_in)
        x = self.model.add_covs2(x)

        return conv_output, x

    def forward_pass(self, x, age, left):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x=x, age=age, left=left)
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

    def generate_cam(self, input_image, age, left, target_class=None):

        conv_output, model_output = self.extractor.forward_pass(x=input_image.to(device), age=age, left=left)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--sag_path', default='C:/Users/lauren erdman/Desktop/kidney_img/HN/repos/SingleRun_20210706/test_RS_cropped2.png')
    # parser.add_argument('--trans_path', default='C:/Users/lauren erdman/Desktop/kidney_img/HN/repos/SingleRun_20210706/test_RT_cropped2.png')
    # parser.add_argument('--age_wks', default=143)
    parser.add_argument('--sag_path', default='C:/Users/lauren erdman/Desktop/kidney_img/HN/repos/SingleRun_20210706/test_RS_cropped1.png')
    parser.add_argument('--trans_path', default='C:/Users/lauren erdman/Desktop/kidney_img/HN/repos/SingleRun_20210706/test_RT_cropped1.png')
    parser.add_argument('--age_wks', default=34)
    parser.add_argument('--left_kidney', action="store_true", help="Flag for left kidney")
    parser.add_argument('--outdir', default="C:/Users/lauren erdman/Desktop/kidney_img/HN/silent_trial/gradcam_test/")
    parser.add_argument('-checkpoint', default="C:/Users/lauren erdman/Desktop/kidney_img/HN/repos/SingleRun_20210706/SickKids_origST_TrainOnly_40epochs_bs16_lr0.001_RCFalse_covTrue_OSFalse_30thEpoch_20210614_v5.pth") ## update with best model
    args = parser.parse_args()
   
    checkpoint = args.checkpoint
    net = SiamNet().to(device)

    if torch.cuda.is_available():
        pretrained_dict = torch.load(checkpoint)['model_state_dict']
    else:
        pretrained_dict = torch.load(checkpoint, map_location='cpu')['model_state_dict']

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
    outdir = args.outdir
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    sag_path = args.sag_path.split("/")[-1].split(".")[0]
    trans_path = args.trans_path.split("/")[-1].split(".")[0]

    X = load_image(args.sag_path, args.trans_path)

    img_path_sag = outdir + '/' + sag_path + '_preprocessed.jpg'
    img_path_trans = outdir + '/' + trans_path + '_preprocessed.jpg'

    save_image(X[0], img_path_sag)
    save_image(X[1], img_path_trans)

    X[0] = torch.from_numpy(X[0]).float()
    X[1] = torch.from_numpy(X[1]).float()
    combined_image = torch.from_numpy(X).float()
    net.eval()
    softmax = torch.nn.Softmax(dim=1)
    # sigmoid = torch.nn.Sigmoid(dim=1)

    with torch.no_grad():
        net.zero_grad()
        output = net(torch.unsqueeze(combined_image, 0).to(device), age=args.age_wks, left=args.left_kidney)
        output_softmax = softmax(output)
        pred_prob = output_softmax[:, 1]
        pred_prob = np.exp(-2.3103 + 3.5598*float(pred_prob))

    print("Un-scaled probability of surgery:::{:6.3f}".format(float(output_softmax[:, 1])))
    print("Probability of surgery:::{:6.3f}".format(float(pred_prob)))
    target_class = 1 if pred_prob >= 0.1 else 0

    original_image_sag = Image.open(img_path_sag)
    original_image_trans = Image.open(img_path_trans)

    sag_filename_id = os.path.join(outdir, sag_path)
    trans_filename_id = os.path.join(outdir, trans_path)

    # Grad cam
    grad_cam = GradCam(net, view='sag')
    # Generate cam mask
    cam = grad_cam.generate_cam(input_image=combined_image, age=args.age_wks, left=int(args.left_kidney), target_class=target_class)
    # Save mask
    save_class_activation_images(original_image_sag, cam, sag_filename_id)

    grad_cam = GradCam(net, view='trans')
    # Generate cam mask
    cam = grad_cam.generate_cam(input_image=combined_image, age=args.age_wks, left=int(args.left_kidney), target_class=target_class)
    # Save mask
    save_class_activation_images(original_image_trans, cam, trans_filename_id)
    print('Grad cam completed')
