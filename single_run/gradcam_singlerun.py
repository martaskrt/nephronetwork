from PIL import Image
import numpy as np
import torch
import os
import scipy
import argparse

from misc_functions import get_example_params, save_class_activation_images

import prep_single_sample
import Prehdict_CNN
CNN = Prehdict_CNN.PrehdictNet

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sag_path', required=True)
    parser.add_argument('--trans_path', required=True)
    parser.add_argument('--outdir', default="results")
    args = parser.parse_args()
   
    checkpoint = "./prehdict_20190802_vanilla_siamese_dim256_c1_checkpoint_18.pth"
    net = CNN().to(device)

    pretrained_dict = torch.load(checkpoint)['model_state_dict']

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

    X = prep_single_sample.load_image(args.sag_path, args.trans_path)

    img_path_sag = outdir + '/' + sag_path + '_preprocessed.jpg'
    img_path_trans = outdir + '/' + trans_path + '_preprocessed.jpg'

    scipy.misc.imsave(img_path_sag, X[0])
    scipy.misc.imsave(img_path_trans, X[1])


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

    print("Probability of surgery:::{:6.3f}".format(float(pred_prob)))
    target_class = 1 if pred_prob >= 0.5 else 0

    original_image_sag = get_example_params(img_path_sag)
    original_image_trans = get_example_params(img_path_trans)

    # Grad cam

    sag_filename_id = os.path.join(outdir, 'sag')
    trans_filename_id = os.path.join(outdir, 'trans')

    grad_cam = GradCam(net, view='sag')
    # Generate cam mask
    cam = grad_cam.generate_cam(combined_image, target_class)
    # Save mask
    save_class_activation_images(original_image_sag, cam, sag_filename_id)

    grad_cam = GradCam(net, view='trans')
    # Generate cam mask
    cam = grad_cam.generate_cam(combined_image, target_class)
    # Save mask
    save_class_activation_images(original_image_trans, cam, trans_filename_id)
    print('Grad cam completed')
