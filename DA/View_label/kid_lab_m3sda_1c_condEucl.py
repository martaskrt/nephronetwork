import numpy as np
import os
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import transforms
# import importlib.machinery
import torchvision.models as models
# from torch.utils.data import Dataset, DataLoader
from matplotlib import cm
import matplotlib.cm as mpl_color_map
import torch.utils.data as data
from PIL import Image
import pandas as pd
import argparse
from sklearn import metrics
from sklearn import metrics
import json
import math
import copy
from random import sample

import datetime
from sklearn.metrics import roc_auc_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###
### GRAD CAM FUNCTIONS
###

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


##From Marta

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

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, G, C1, args):
        self.G = G
        self.C1 = C1
        self.gradients = None
        self.args = args

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None

        if len(x.shape) == 4:
            B, C, H, W = x.shape
        elif len(x.shape) == 3:
            C, H, W = x.shape
            B = 1
            x = x.unsqueeze(0)
            # print(x.shape)

        x = x.type(torch.FloatTensor).to(self.args.device)

        if self.args.batch_size > 1:
            x = F.max_pool2d(F.relu(self.G.bn1(self.G.conv1(x))), stride=2, kernel_size=3, padding=1)
            x = F.max_pool2d(F.relu(self.G.bn3(self.G.conv3(x))), stride=2, kernel_size=3, padding=1)
            conv_out = F.max_pool2d(F.relu(self.G.bn4(self.G.conv4(x))), stride=2, kernel_size=3, padding=1)
        else:
            x = F.max_pool2d(F.relu(self.G.conv1(x)), stride=2, kernel_size=3, padding=1)
            x = F.max_pool2d(F.relu(self.G.conv3(x)), stride=2, kernel_size=3, padding=1)
            conv_out = F.max_pool2d(F.relu(self.G.conv4(x)), stride=2, kernel_size=3, padding=1)

        conv_out.register_hook(self.save_gradient)
        conv_output = conv_out  # Save the convolution output on that layer
        # print(x.shape)

        x = conv_out.view(conv_out.size(0), 2048)
        x = F.relu(self.G.bn2_fc(self.G.fc2(x)))

        model_out = self.C1(x)

        return model_out, conv_output

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        x, conv_output = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        return conv_output, x

class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, G, C1, args):
        self.G = G.to(device)
        self.C1 = C1.to(device)
        self.G.eval()
        self.C1.eval()

        # Define extractor
        self.extractor = CamExtractor(self.G, self.C1, args=args)

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

        self.G.zero_grad()
        self.C1.zero_grad()
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
        cam = np.rot90(np.uint8(cam * 255))  # Scale between 0-255 to visualize

        input_image = torch.unsqueeze(input_image[0], 0)
        input_image = torch.unsqueeze(input_image, 1)
        input_image = input_image.expand(-1, 3, -1, -1)
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        return cam

### From marta: @Lauren to update these!!
def save_class_activation_images(org_img, activation_map, file_name,
                                 file_path = "C:/Users/lauren erdman/Desktop/kidney_img/View_Labeling/OutputNov2020/cams_out/"):
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

    path_to_file = file_path + file_name + '_Cam_On_Image.png'

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

###
###    DATA PROCESSING FUNCTIONS
###

def read_image_file(file_name):
    """
    Reads in image file, crops it (if needed), converts it to greyscale, and returns image
    :param file_name: path and name of file

    :return: Torch tensor of image
    """
    my_img = Image.open(file_name).convert('L')
    my_arr = np.asarray(my_img)
    my_img.close()

    return torch.from_numpy(my_arr).type(torch.DoubleTensor)


def flatten_list(in_list):
    flat_list = []
    for sublist in in_list:
        for item in sublist:
            flat_list.append(item)

    return flat_list


## ORIGINALLY FROM MARTA SKRETA
def get_metrics(y_score, y_true):

    fpr, tpr, auroc_thresholds = metrics.roc_curve(y_score=y_score, y_true=y_true)
    auc = metrics.auc(fpr, tpr)
    auprc = metrics.average_precision_score(y_score=y_score, y_true=y_true)
    precision, recall, auprc_thresholds = metrics.precision_recall_curve(probas_pred=y_score, y_true=y_true)

    results = {'pred_score': y_score,
               'fp': y_true,
               'auc': auc,
               'auprc': auprc,
               'auroc_thresholds': auroc_thresholds,
               'precision': precision,
               'recall': recall,
               'auprc_thresholds': auprc_thresholds,
               }

    return results


## ORIGINALLY FROM VisionLearningGroup.github.io
def return_dataset(args):
    """

    Args:
        datasheet: Datasheet specifying:
            - sources: "machine"
            - test: {0,1} data in test set "test"
            - image file name: "IMG_FILE"
            - label: must match what's specified in args.lab_col
        args: args specifying:
            - label column: args.lab_col
            - target machine: args.target
            - us_dir: args.lab_us_dir
            - dimensions of the images: args.dim

    Returns:
        (1,2) source training and test dictionaries organzied by machine (source)
        (3,4) target training and test dictionaries

    An aside:
        In this code, I'm using this function to return the full dataset (all domains)
        but in VisionLearningGroup.github.io, this is used to read in each domain
        so they can be iteratively added in the "dataset_read" function to a dictionary and
        ultimately made into a data loader.

        My take on this function: it does everything except creating the data loader

    """

    datasheet = pd.read_csv(args.datasheet)

    full_source_set = list(datasheet['machine'].unique())
    source_set = [source for source in full_source_set if source != args.target]

    ### from my own code
    sub_keys = ["img", "labels"]

    target_dict = {my_key: dict() for my_key in sub_keys}
    target_test_dict = {my_key: dict() for my_key in sub_keys}
    source_dict = {source_key: {sub_key: dict() for sub_key in sub_keys} for source_key in source_set}
    source_test_dict = {source_key: {sub_key: dict() for sub_key in sub_keys} for source_key in source_set}

    # target_dict = dict.fromkeys(, dict())
    # target_test_dict = dict.fromkeys(["img", "labels"], dict())
    # source_dict = dict.fromkeys(source_set, dict.fromkeys(["img", "labels"], dict()))
    # source_test_dict = dict.fromkeys(source_set, dict.fromkeys(["img", "labels"], dict()))

    for index, row in datasheet.iterrows():
        # print(row)

        if not math.isnan(row['test']):
            # print(row)

            ## check if machine is the target domain (this will make it easy to rotate targets)
            if row['machine'] == args.target:
                ## split machine-dictionary into train and test
                if row['test'] == 0:
                    ## read_image_file returns 256x256 (args.dim) torch tensor
                    target_dict["img"][row['IMG_ID']] = read_image_file(args.lab_us_dir + "/" + row['IMG_FILE']).view(1,
                                                                                                                      args.dim,
                                                                                                                      args.dim)
                    target_dict["labels"][row['IMG_ID']] = row[args.lab_col]
                else:
                    target_test_dict["img"][row['IMG_ID']] = read_image_file(args.lab_us_dir + "/" + row['IMG_FILE']).view(
                        1, args.dim, args.dim)
                    target_test_dict["labels"][row['IMG_ID']] = row[args.lab_col]

            else:
                if row['test'] == 0:
                    source_dict[row['machine']]["img"][row['IMG_ID']] = read_image_file(
                        args.lab_us_dir + "/" + row['IMG_FILE']).view(1, args.dim, args.dim)
                    source_dict[row['machine']]["labels"][row['IMG_ID']] = row[args.lab_col]

                    # print(row['machine'])
                    # print(row['IMG_ID'])
                    # source_dict[row['machine']]["img"][row['IMG_ID']], source_dict[row['machine']]["labels"][row['IMG_ID']] = \
                    #     read_image_file(args.lab_us_dir + "/" + row['IMG_FILE']).view(1, args.dim, args.dim), row[args.lab_col]

                else:
                    source_test_dict[row['machine']]["img"][row['IMG_ID']] = read_image_file(
                        args.lab_us_dir + "/" + row['IMG_FILE']).view(1, args.dim, args.dim)
                    source_test_dict[row['machine']]["labels"][row['IMG_ID']] = row[args.lab_col]

    print("Data dictionaries created.")
    # print(source_dict)
    # print(source_dict[list(source_dict.keys())])
    ### I'M SOMEHOW GETTING THE SAME IDS (AND IMAGES, ETC) IN MY OUTPUT... HOW IS IT GETTING THERE?

    return source_dict, source_test_dict, target_dict, target_test_dict


## ORIGINALLY FROM VisionLearningGroup.github.io
def dataset_read(args):
    ### TRAINING AND TEST DATA
    S, S_test, T, T_test = return_dataset(args=args)

    scale = 32

    train_loader = UnalignedDataLoader()
    train_loader.initialize(source=S, target=T, scale=scale, args=args)
    dataset = train_loader.load_data()

    test_loader = UnalignedDataLoader()
    test_loader.initialize(source=S_test, target=T_test, scale=scale, args=args)
    dataset_test = test_loader.load_data()

    return dataset, dataset_test

###
###    M3SDA FUNCTIONS
###

## ORIGINALLY FROM VisionLearningGroup.github.io
def euclidean(x1, x2, ep=1e-7):
    euc = ((x1 - x2) ** 2).sum() + torch.tensor(ep)
    return euc.sqrt()


## ORIGINALLY FROM VisionLearningGroup.github.io
def k_moment(source_dict, target_dict, k):
    eucl_sum = 0
    target_k_moment_mean = (target_dict["centered_feat"] ** k).mean(0)
    for my_key in source_dict.keys():
        source_dict[my_key]["k_moment_mean"] = (source_dict[my_key]["centered_feat"] ** k).mean(0)
        eucl_sum = eucl_sum + euclidean(target_k_moment_mean, source_dict[my_key]["k_moment_mean"])

    key_list = list(source_dict.keys())
    for i in range(len(source_dict)):
        for j in range(i):
            eucl_sum = eucl_sum + euclidean(source_dict[key_list[i]]["k_moment_mean"],
                                            source_dict[key_list[j]]["k_moment_mean"])

    return eucl_sum

## ORIGINALLY FROM VisionLearningGroup.github.io
def cond_k_moment(source_dict, target_dict, k):
    eucl_sum = 0

    for label in [0, 1]:
        if "k_moment_mean_" + str(label) in target_dict.keys():
            target_dict["k_moment_mean_" + str(label)] = (target_dict["centered_feat_" + str(label)] ** k).mean(0)

    for my_key in source_dict.keys():
        for label in [0, 1]:
            if "centered_feat_" + str(label) in source_dict[my_key].keys():
                source_dict[my_key]["k_moment_mean_" + str(label)] = (source_dict[my_key]["centered_feat_" + str(label)] ** k).mean(0)
                if "k_moment_mean_" + str(label) in target_dict.keys():
                    eucl_sum += euclidean(target_dict["k_moment_mean_" + str(label)], source_dict[my_key]["k_moment_mean_" + str(label)])

    ### Add all combinations of source feature k-moment distances, conditional on label
    key_list = list(source_dict.keys())
    for i in range(len(source_dict)):
        for j in range(i):
            for label in [0, 1]:
                if "k_moment_mean_" + str(label) in source_dict[key_list[i]].keys() and \
                        "k_moment_mean_" + str(label) in source_dict[key_list[j]].keys():
                    eucl_sum += euclidean(source_dict[key_list[i]]["k_moment_mean_" + str(label)],
                                          source_dict[key_list[i]]["k_moment_mean_" + str(label)])

    return eucl_sum


def split_features(source_dict=None, target_dict=None, bs=4):

    if not source_dict==None:

        for my_key in source_dict.keys():
            my_zip = zip(source_dict[my_key]['label'], source_dict[my_key]['feat'])

            for label, feat in my_zip:  ## hard coding 2 labels
                if label == 0:
                    if "feat_" + str(label) in source_dict[my_key].keys():
                        source_dict[my_key]["feat_" + str(label)] = torch.cat((
                            source_dict[my_key]["feat_" + str(label)], feat.view(-1, 1024)))
                    else:
                        source_dict[my_key]["feat_" + str(label)] = feat.view(-1, 1024)

                elif label == 1:
                    if "feat_" + str(label) in source_dict[my_key].keys():
                        source_dict[my_key]["feat_" + str(label)] = torch.cat((
                            source_dict[my_key]["feat_" + str(label)], feat.view(-1, 1024)))
                    else:
                        source_dict[my_key]["feat_" + str(label)] = feat.view(-1, 1024)

                else:
                    print("Non-0/1 label output. Time to debug.")

                ## UPDATE THIS
                ## make conditional on label -- upsample minority source (i.e. if there are few 1's 0's OR skip if there are none)

                for label in [0, 1]:
                    if "feat_" + str(label) in source_dict[my_key].keys():
                        if source_dict[my_key]["feat_" + str(label)].shape[0] < bs:
                            my_feats = source_dict[my_key]["feat_" + str(label)]
                            source_dict[my_key]["feat_" + str(label)] = torch.cat(
                                (my_feats.view(-1, 1024), my_feats[torch.randint(my_feats.shape[0], (bs - my_feats.shape[0],))].view(-1, 1024)))

                        source_dict[my_key]["centered_feat_" + str(label)] = source_dict[my_key]["feat" + str(label)] - source_dict[my_key]["feat" + str(label)].mean(0)

    if not target_dict==None:
        target_zip = zip(target_dict['C1'].max(1)[1], target_dict['feat'])
            ## ^ proxy target label with C1 label
        for t_label, t_feat in target_zip:
            if t_label == 0:
                if "feat_" + str(t_label) in target_dict.keys():
                    target_dict["feat_" + str(t_label)] = torch.cat(
                        (target_dict["feat_" + str(t_label)].view(-1, 1024), t_feat.view(-1, 1024)))
                else:
                    target_dict["feat_" + str(t_label)] = t_feat.view(-1, 1024)

            elif t_label == 1:
                if "feat_" + str(t_label) in target_dict.keys():
                    target_dict["feat_" + str(t_label)] = torch.cat((
                        target_dict["feat_" + str(t_label)].view(-1, 1024), t_feat.view(-1, 1024)))
                else:
                    target_dict["feat_" + str(t_label)] = t_feat.view(-1, 1024)

            else:
                print("Non-0/1 label output. Time to debug.")

            for label in [0, 1]:
                if "feat_" + str(label) in target_dict.keys():
                    if target_dict["feat_" + str(label)].shape[0] < bs:
                        t_feats = target_dict["feat_" + str(label)].view(-1, 1024)
                        target_dict["feat_" + str(label)] = torch.cat(
                            (t_feats, t_feat[torch.randint(t_feats.shape[0], (bs - t_feats.shape[0],))].view(-1, 1024)))

                    target_dict["centered_feat_" + str(label)] = target_dict["feat" + str(label)] - target_dict["feat" + str(label)].mean(0)

    return source_dict, target_dict


## ORIGINALLY FROM VisionLearningGroup.github.io
def msda_regulizer(source_dict, target_dict,
                   belta_moment, c_msda, bs=None):  ## CHECK BELTA_MOMENT -- DOES IT CORRESPOND WITH THE NUMBER OF SOURCES?
    # print('s1:{}, s2:{}, s3:{}, s4:{}'.format(output_s1.shape, output_s2.shape, output_s3.shape, output_t.shape))

    eucl_sum = 0

    if c_msda:
        source_dict, target_dict = split_features(source_dict=source_dict, target_dict=target_dict, bs=bs)
    else:
        target_dict["centered_feat"] = target_dict["feat"] - target_dict["feat"].mean(0)
        for my_key in source_dict.keys():
            source_dict[my_key]["centered_feat"] = source_dict[my_key]["feat"] - source_dict[my_key]["feat"].mean(0)

    reg_info = 0
    # print(reg_info)
    for i in range(belta_moment - 1):
        if c_msda:
            reg_info += cond_k_moment(source_dict=source_dict, target_dict=target_dict, k=i + 1) ## MAYBE JUST START AT K = i + 1?
        else:
            reg_info += k_moment(source_dict=source_dict, target_dict=target_dict, k=i + 1) ## MAYBE JUST START AT K = i + 1?

    return reg_info / 6

def compute_performance(loss_in, pred_in, lab_in, id_in, feat_in,
                        out_file_name, epoch, data_split, write_file=True):

        #score_in = self.softmax(pred_in)[:, 1]
        # lab_in = lab_in.tolist()
        # score_in = score_in.tolist()

        pred_01 = np.round(pred_in)
        acc_in = np.sum(pred_01 == np.array(lab_in))/len(pred_01)
        # acc_in = torch.sum(torch.argmax(pred_in, dim=1) == torch.tensor(lab_in).to('cuda')).cpu()/len(lab_in)

        # print("debug")
        # print("pred_in:")
        # print(pred_in)
        # print("lab_in:")
        # print(lab_in)
        results_train = get_metrics(y_score=pred_in,
                                    y_true=lab_in)

        print('\n' + data_split + '\tEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
              'AUPRC\t{:.6f}'.format(epoch,
                                     acc_in,
                                     loss_in,
                                     results_train['auc'],
                                     results_train['auprc']))
        if write_file:
            out_dict = {data_split + "_" + str(epoch): {
                "epoch": epoch,
                "accuracy": acc_in,
                "loss": loss_in,
                "AUC": results_train['auc'],
                "AUPRC": results_train['auprc'],
                "pred_vals": pred_in,
                "labels": lab_in,
                "ids": id_in
            }}

            print("writing json file to: " + out_file_name + "_Detailed.json")
            if not os.path.isfile(out_file_name + "_Detailed.json"):
                with open(out_file_name + "_Detailed.json", 'w+') as of:
                    # print(out_dict)
                    # print("AUC")
                    json.dump(out_dict, of, indent=4)
                of.close()
            else:
                with open(out_file_name + "_Detailed.json", "r+") as fi:
                    data = json.load(fi)
                    data.update(out_dict)
                    fi.seek(0)
                    # fi.truncate()
                    # print(out_dict)
                    json.dump(data, fi, indent=4)
                fi.close()
            print("json file written")

            print("writing text file")
            outfile = open(out_file_name + "_Loss.txt", 'a+')
            outfile.write('\n' + data_split + '\tEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}'.format(epoch,
                                         acc_in,
                                         loss_in,
                                         results_train['auc'],
                                         results_train['auprc']))
            outfile.close()
            print("text file written")

# def get_auc():


###

###
###        DATASET
###

## ORIGINALLY FROM VisionLearningGroup.github.io
class Dataset(data.Dataset):
    """Args:
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, data, label,
                 transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.labels = label
        self.keys = list(data.keys())

    def __getitem__(self, index):
        """
         Args:
             index (int): Index
         Returns:
             tuple: (image, target) where target is index of the target class.
         """
        id = self.keys[index]
        img, target = self.data[id], self.labels[id]

        # print("GET ITEM")
        # print(self.keys[index])
        # print(id)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # print(img.shape)
        # if img.shape[0] != 1:
        #     # print(img)
        #     img = Image.fromarray(np.uint8(np.asarray(img.transpose((1, 2, 0)))))

        # elif img.shape[0] == 1:
        #     im = np.uint8(np.asarray(img))
        #     # print(np.vstack([im,im,im]).shape)
        #     im = np.vstack([im, im, im]).transpose((1, 2, 0))
        #     img = Image.fromarray(im)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        # if self.transform is not None:
        #     img = self.transform(img)
        #     # return img, target
        return img, target, id

    def __len__(self):
        return len(self.data)

# class ConditionalDataset0(data.Dataset):
#     """Args:
#         transform (callable, optional): A function/transform that  takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.RandomCrop``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#     """
#
#     def __init__(self, data, label,
#                  transform=None, target_transform=None):
#         self.transform = transform
#         self.target_transform = target_transform
#         self.data = data
#         self.labels = label
#         self.keys = list(data.keys())
#
#     def __getitem__(self, index, my_label):
#         """
#          Args:
#              index (int): Index
#          Returns:
#              tuple: (image, target) where target is index of the target class.
#          """
#         id = self.keys[index]
#
#         if not self.labels[id] == my_label:
#             ids = [k for k, v in self.labels.items() if int(v) == my_label]
#             id = str(sample(ids, 1))
#
#         img, target = self.data[id], self.labels[id]
#
#         return img, target, id
#
#     def __len__(self):
#         return len(self.data)


## MODIFIED FROM VisionLearningGroup.github.io
class UnalignedDataLoader:
    def initialize(self, args, source, target, scale=32):

        ## check these transformations
        # transform = transforms.Compose([
        #     transforms.Scale(scale),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])

        batch_size1 = args.batch_size
        batch_size2 = args.batch_size

        ### Datasets
        self.dataset_source = dict()

        for my_key in source.keys():
            self.dataset_source[my_key] = Dataset(source[my_key]['img'], source[my_key]['labels'])
        self.dataset_t = Dataset(target['img'], target['labels'])

        ## Dataloaders
        source_data_loader_dict = dict()
        for my_key in self.dataset_source.keys():
            source_data_loader_dict[my_key] = torch.utils.data.DataLoader(self.dataset_source[my_key],
                                                                          batch_size=batch_size1, shuffle=True,
                                                                          num_workers=1)

        target_data_loader = torch.utils.data.DataLoader(self.dataset_t, batch_size=batch_size2,
                                                         shuffle=True, num_workers=1)

        self.paired_data = PairedData(source_data_loader_dict, target_data_loader,
                                      max_dataset_size=float("inf"), args=args)

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        len_list = []
        for i in range(len(self.dataset_source)):
            len_list[i] = len(self.dataset_source)
        len_list[len(len_list)] = len(self.dataset_t)

        return min(max(len_list), float("inf"))


## MODIFIED FROM VisionLearningGroup.github.io
class PairedData(object):
    def __init__(self, data_loader_s, data_loader_t, max_dataset_size, args):
        """

        Args:
            data_loader_s: dictionary of source data loaders
            data_loader_t: target data loader (not a dictionary)
            max_dataset_size:
        """
        ## data loaders
        self.data_loader_s = data_loader_s
        self.data_loader_t = data_loader_t
        self.args = args

        ## stops
        self.stop_dict = dict()
        for key in self.data_loader_s.keys():
            self.stop_dict[key] = False
        self.stop_t = False

        self.max_dataset_size = max_dataset_size

    def __iter__(self):
        self.stop_s = dict()
        self.data_loader_s_iter = dict()
        ## add arg : if self.args.c_msda: ; label = sample(unique(labels)) ; iter(self.data_loader_s[key],label)

        for key in self.data_loader_s.keys():
            self.data_loader_s_iter[key] = iter(self.data_loader_s[key])
            self.stop_s[key] = False

        self.data_loader_t_iter = iter(self.data_loader_t)
        self.stop_t = False
        self.iter = 0

        return self

    def __next__(self):

        return_dat_s = dict()

        for key in self.data_loader_s.keys():
            # print("Paired data key: " + key)
            # # return_dat_s[key]["img"], return_dat_s[key]["label"] = None, None
            # print("IDs: ")
            return_dat_s[key] = dict()
            try:
                return_dat_s[key]["img"], return_dat_s[key]["label"], return_dat_s[key]["id"] = next(iter(self.data_loader_s[key]))
                # print(return_dat_s[key]['id'])
            except StopIteration:
                if return_dat_s[key]["img"] is None or return_dat_s[key]["label"] is None:
                    self.stop_s[key] = True
                    self.data_loader_s_iter[key] = iter(self.data_loader_s[key])
                    return_dat_s[key]["img"], return_dat_s[key]["label"], return_dat_s[key]["id"] = next(self.data_loader_s_iter[key])

                    # print(return_dat_s[key]['id'])

        t, t_paths, t_ids = None, None, None
        try:
            t, t_paths, t_ids = next(self.data_loader_t_iter)
        except StopIteration:
            if t is None or t_paths is None:
                self.stop_t = True
                self.data_loader_t_iter = iter(self.data_loader_t)
                t, t_paths, t_ids = next(self.data_loader_t_iter)

        if any(self.stop_s.values()) or self.stop_t or self.iter > self.max_dataset_size:
            for key in self.stop_s:
                self.stop_s[key] = False
            self.stop_t = False
            raise StopIteration()
        else:
            self.iter += 1
            out_dict = {'source': return_dat_s, 'target': {'img': t, 'label': t_paths, 'id': t_ids}}

            # for key in out_dict['source'].keys():
            #     print("out_dict key: " + key)
            #     print("out_dict source IDs")
            #     print(out_dict['source'][key]['id'])

            """
            DATALOADER DATA STRUCTURE: 
            output is organized as a dictionary of: 
            "source": the source data indexed by the source 
                         (machine) name and then the data for each iteration is in a dictionary with 
                         "img" for the image input and "label" for the label input

            "target": the target data with "img" for the image input and "label" for the label input
            """
            return out_dict


###
###
###        MODELS
###

##  VIEW LABEL MODELS -- from original training model. Won't use for now but is similar to "Feature" below
class KidneyLab(nn.Module):
    def __init__(self, args):
        super(KidneyLab, self).__init__()

        self.conv0 = nn.Sequential(nn.Conv2d(1, 64, 7, padding=2),
                                   nn.MaxPool2d(2),
                                   nn.ReLU())
        self.conv1 = nn.Sequential(nn.Conv2d(64, 128, 7, padding=2),
                                   nn.MaxPool2d(2),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 7, padding=2),
                                   nn.MaxPool2d(2),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 32, 7, padding=2),
                                   nn.MaxPool2d(2),
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(32, 16, 7, padding=2),
                                   nn.MaxPool2d(2),
                                   nn.ReLU())

        self.linear1 = nn.Sequential(nn.Linear(576, 256, bias=True),
                                     nn.ReLU(),
                                     nn.Dropout(0.5))
        self.linear2 = nn.Sequential(nn.Linear(256, 64, bias=True),
                                     nn.ReLU(),
                                     nn.Dropout(0.5))

        if args.RL:
            self.linear3 = nn.Sequential(nn.Linear(64, 5, bias=True))
        else:
            self.linear3 = nn.Sequential(nn.Linear(64, 3, bias=True))

    def forward(self, x):

        if len(x.squeeze().shape) > 2:
            bs = x.squeeze().shape[0]
        else:
            bs = 1

        dim = x.shape[len(x.shape) - 1]

        x1 = self.conv0(x.view([bs, 1, dim, dim]))
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x5 = self.conv4(x4)

        x5_flat = x5.view([bs, 1, -1])

        x6 = self.linear1(x5_flat)
        x7 = self.linear2(x6)
        x8 = self.linear3(x7)

        return x8


## ORIGINALLY FROM VisionLearningGroup.github.io
class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


## ORIGINALLY FROM VisionLearningGroup.github.io
def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)


## ORIGINALLY FROM VisionLearningGroup.github.io
## becomes your G(.) function -- replace with what you want it to be
## e.g.: CNN feeding into an LSTM? A CNN feeding into the surg model?
        ### Note: made this smaller, will need to re-train/test baseline models
class Feature(nn.Module):
    def __init__(self, args):
        super(Feature, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 16, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 2, kernel_size=5, stride=1, padding=2)
        self.bn4 = nn.BatchNorm2d(2)
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2_fc = nn.BatchNorm1d(1024)

        self.args = args

    def forward(self, x, reverse=False):
        if len(x.shape) == 4:
            B, C, H, W = x.shape
        elif len(x.shape) == 3:
            C, H, W = x.shape
            B = 1
            x = x.unsqueeze(0)
            print(x.shape)

        x = x.type(torch.FloatTensor).to(self.args.device)

        if self.args.batch_size == 1:
            x = F.max_pool2d(F.relu(self.conv1(x)), stride=2, kernel_size=3, padding=1)
            x = F.max_pool2d(F.relu(self.conv3(x)), stride=2, kernel_size=3, padding=1)
            x = F.max_pool2d(F.relu(self.conv4(x)), stride=2, kernel_size=3, padding=1)

        else:
            x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=3, padding=1)
            x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), stride=2, kernel_size=3, padding=1)
            x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), stride=2, kernel_size=3, padding=1)

        # print(x.shape)
        x2 = x.view(B, 2048)
        x2 = F.dropout(x2, training=self.training)
        if reverse:
            x2 = grad_reverse(x2, self.lambd)

        if self.args.batch_size == 1 or B == 1:
            x2_out = F.relu(self.fc2(x2))
        else:
            x2_out = F.relu(self.bn2_fc(self.fc2(x2)))

        return x2_out


## ORIGINALLY FROM VisionLearningGroup.github.io
class Predictor(nn.Module):
    def __init__(self, prob=0.5):
        super(Predictor, self).__init__()
        # self.fc1 = nn.Linear(8192, 3072)
        # self.bn1_fc = nn.BatchNorm1d(3072)
        # self.fc2 = nn.Linear(3072, 2048)
        # self.bn2_fc = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(1024, 2)
        self.bn_fc3 = nn.BatchNorm1d(2)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        # if reverse:
        #     x = grad_reverse(x, self.lambd)
        # x = F.relu(self.bn2_fc(self.fc2(x)))
        x = self.fc3(x)
        return x


###
###
###        MODEL TRAINING FUNCTIONS
###


## ORIGINALLY FROM VisionLearningGroup.github.io
class Solver(object):
    def __init__(self, args, interval=100, save_epoch=10, data_split="Train"):
        self.batch_size = args.batch_size
        self.target = args.target
        self.checkpoint_dir = args.checkpoint_dir
        self.save_epoch = save_epoch
        self.use_abs_diff = args.use_abs_diff
        self.source = args.source
        self.device = args.device

        self.c_msda = args.c_msda
        self.no_discr = args.no_discr
        self.cams_out = args.cams_out
        self.args = args

        ## Load source and target data loaders
        # print('Dataset loading')
        self.datasets, self.dataset_test = dataset_read(args=args)
        # print(self.dataset['S1'].shape)

        # print('Data loading finished!')
        self.lr = args.lr
        self.mom = args.mom
        self.wd = args.weight_decay
        self.softmax = nn.Softmax(0)

        self.G = Feature(args=args)
        self.C1 = Predictor()
        self.C2 = Predictor()

        if args.eval_only:
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, args.resume_epoch))
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (
                    self.checkpoint_dir, self.source, self.target, self.checkpoint_dir, args.resume_epoch))
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, args.resume_epoch))

        self.G.cuda()
        self.C1.cuda()
        self.C2.cuda()
        # print('Model loaded')

        self.interval = interval

        self.set_optimizer(which_opt=args.optimizer)
        print('Initialization complete')

    def set_optimizer(self, which_opt='momentum'):
        if which_opt == 'momentum':
            self.opt_g = optim.SGD(self.G.parameters(),
                                   lr=self.lr, weight_decay=self.wd,
                                   momentum=self.mom)

            self.opt_c1 = optim.SGD(self.C1.parameters(),
                                    lr=self.lr, weight_decay=self.wd,
                                    momentum=self.mom)
            self.opt_c2 = optim.SGD(self.C2.parameters(),
                                    lr=self.lr, weight_decay=self.wd,
                                    momentum=self.mom)

        if which_opt == 'adam':
            self.opt_g = optim.Adam(self.G.parameters(),
                                    lr=self.lr, weight_decay=self.wd)

            self.opt_c1 = optim.Adam(self.C1.parameters(),
                                     lr=self.lr, weight_decay=self.wd)
            self.opt_c2 = optim.Adam(self.C2.parameters(),
                                     lr=self.lr, weight_decay=self.wd)

    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_c1.zero_grad()
        self.opt_c2.zero_grad()

    def ent(self, output):
        return - torch.mean(output * torch.log(output + 1e-6))

    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

    def update_output(self, output_dict, data_dict, in_key=None, source=True, msda=False):

        ### SEEEMS LIKE THE DATA DICTIONARY HAS THE WRONG DATA SOMEHOW?

        # print("OUTPUT DICT: ")
        # print(output_dict)

        if msda:
            if source:
                # print(data_dict[in_key]["C1_loss"])
                # print(data_dict[in_key][""])
                # print(in_key + " loss from data_dictionary (source_dict?)")
                data_dict[in_key]["loss"] = data_dict[in_key]["C1_loss"]
                # print(data_dict[in_key]["loss"])
                data_dict[in_key]["output"] = data_dict[in_key]["C1"]
                # print(in_key + " output from data_dictionary (source_dict?)")
                # print(data_dict[in_key]["output"])
                # print(in_key + " IDs from data_dictionary (source_dict?)")
                # print(data_dict[in_key]["id"])
            else:
                data_dict["output"] = data_dict["C1"]

        if source:
            try:
                output_dict[in_key]["loss"] = output_dict[in_key]["loss"] + data_dict[in_key]["loss"].item()
                if self.batch_size == 1:
                    output_dict[in_key]["label"].extend([data_dict[in_key]["label"].item()])
                    output_dict[in_key]["output"].extend(
                        [self.softmax(data_dict[in_key]["output"].squeeze())[1].item()])

                else:
                    output_dict[in_key]["label"].extend(data_dict[in_key]["label"].tolist())
                    output_dict[in_key]["output"].extend(self.softmax(data_dict[in_key]["output"])[:, 1].tolist())

            except KeyError:
                output_dict[in_key] = {"loss": 0, "output": [], "label": [], "feat": [], "id": []}
                output_dict[in_key]["loss"] = output_dict[in_key]["loss"] + data_dict[in_key]["loss"].item()
                if self.batch_size == 1:
                    output_dict[in_key]["label"].extend([data_dict[in_key]["label"].item()])
                    output_dict[in_key]["output"].extend(
                        [self.softmax(data_dict[in_key]["output"].squeeze())[1].item()])

                else:
                    output_dict[in_key]["label"].extend(data_dict[in_key]["label"].tolist())
                    output_dict[in_key]["output"].extend(self.softmax(data_dict[in_key]["output"])[:, 1].tolist())

            output_dict[in_key]["id"].extend(list(data_dict[in_key]["id"]))
            output_dict[in_key]["feat"].extend(data_dict[in_key]["feat"].tolist())

        else:
            output_dict["loss"] = output_dict["loss"] + data_dict["loss"].item()
            if self.batch_size == 1:
                output_dict["label"].extend([data_dict["label"].item()])
                output_dict["output"].extend([self.softmax(data_dict["output"].squeeze())[1].item()])

            else:
                output_dict["label"].extend(data_dict["label"].tolist())
                output_dict["output"].extend(self.softmax(data_dict["output"])[:, 1].tolist())
            output_dict["id"].extend(list(data_dict["id"]))
            output_dict["feat"].extend(data_dict["feat"].tolist())

        return output_dict

    def train(self, epoch, record_file=None):
        criterion = nn.CrossEntropyLoss().cuda()
        self.G.train()
        self.C1.train()

        torch.cuda.manual_seed(1)

        source_output = {}
        target_output = {"loss": 0, "output": [], "label": [], "feat": [], "id": []}

        for batch_idx, data in enumerate(self.datasets):

            source_data = data["source"]
            target_data = data["target"]

            source_data_too_small = []

            ## Get new data batch from source and target data
            for my_key in data['source'].keys():
                source_data[my_key]['img'] = Variable(data['source'][my_key]['img'].cuda())
                source_data[my_key]['label'] = Variable(data['source'][my_key]['label'].long().cuda())
                source_data_too_small.append(source_data[my_key]["img"].size()[0] < self.batch_size)

            target_data['img'] = Variable(data['target']['img'].cuda())
            target_data['label'] = Variable(data['target']['label'].long().cuda())

            if any(source_data_too_small) or target_data['img'].size()[0] < self.batch_size:
                break

            self.reset_grad()

            loss_sum = 0
            for my_key in source_data.keys():
                source_data[my_key]["feat"] = self.G(source_data[my_key]['img'])
                source_data[my_key]["output"] = self.C1(source_data[my_key]["feat"])
                source_data[my_key]["loss"] = criterion(source_data[my_key]["output"], source_data[my_key]["label"])

                source_output = self.update_output(output_dict=source_output, data_dict=source_data, in_key=my_key)
                loss_sum = loss_sum + source_data[my_key]["loss"]

            # print("performing target forward pass")
            target_data["feat"] = self.G(target_data["img"])
            target_data["output"] = self.C1(target_data["feat"])
            target_data["loss"] = torch.tensor(np.nan)

            target_output = self.update_output(output_dict=target_output, data_dict=target_data, source=False)

            loss_s = loss_sum / len(source_data)

            # print("performing source backward pass")
            loss_s.backward()  ## why only loss_s backward?? A: This is "train" not "train_MSDA"
            # so there's no accounting for discrepancy

            self.opt_c1.step()
            self.opt_g.step()
            self.reset_grad()
            if batch_idx > 500:
                return batch_idx

            # if batch_idx % self.interval == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx, 100,
            #         100. * batch_idx / 70000, loss_s.item()))

        print("source batches trained")
        for my_key in source_output.keys():
            print(my_key)
            compute_performance(loss_in=source_output[my_key]["loss"],
                                pred_in=source_output[my_key]["output"],
                                lab_in=source_output[my_key]["label"],
                                id_in=source_output[my_key]["id"],
                                feat_in=source_output[my_key]["feat"],
                                out_file_name=record_file, epoch=epoch,
                                data_split="SourceTrain_"+my_key)

        compute_performance(loss_in=target_output["loss"],
                            pred_in=target_output["output"],
                            lab_in=target_output["label"],
                            id_in=target_output["id"],
                            feat_in=target_output["feat"],
                            out_file_name=record_file, epoch=epoch,
                            data_split="TargetTrain")

        return batch_idx

    def train_merge_baseline(self, epoch, record_file=None):
        criterion = nn.CrossEntropyLoss().cuda()
        self.G.train()
        self.C1.train()

        torch.cuda.manual_seed(1)

        for batch_idx, data in enumerate(self.datasets):
            # if
            keys_list = list(data["source"].keys())
            img_s = data["source"][keys_list[0]]["img"]
            label_s = data["source"][keys_list[0]]["label"]
            for my_key in data["source"].keys():
                if my_key != [0]:
                    img_s = torch.cat((img_s, data["source"][my_key]["img"]), 0)
                    label_s = torch.cat((label_s, data["source"][my_key]["label"].long()), 0)

            img_t = Variable(data['T']["img"].cuda())

            label_s = Variable(label_s.long().cuda())
            img_s = Variable(img_s.cuda())

            if img_s.size()[0] < self.batch_size or img_t.size()[0] < self.batch_size:
                break

            self.reset_grad()
            feat_s = self.G(img_s)
            output_s = self.C1(feat_s)

            feat_t = self.G(img_t)
            output_t = self.C1(feat_t)

            loss_s = criterion(output_s, label_s)

            loss_s.backward()

            self.opt_c1.step()
            self.opt_g.step()
            self.reset_grad()
            if batch_idx > 500:
                return batch_idx

            if batch_idx % self.interval == 0:
                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t '.format(
                #     epoch, batch_idx, 100,
                #     100. * batch_idx / 70000, loss_s.data[0]))
                if record_file:
                    record = open(record_file, 'a+')
                    record.write('%s\n' % (loss_s[0]))
                    record.close()
        return batch_idx

    def test(self, epoch, record_file=None, save_model=False):
        self.G.eval()
        self.C1.eval()
        test_loss = 0
        correct1 = 0
        size = 0
        feature_all = np.array([])
        label_all = []
        test_pred_vals = []
        pred_vals = []
        pred1 = []
        ids = []
        feats = []
        for batch_idx, data in enumerate(self.dataset_test):
            # print(batch_idx)
            img = data['target']["img"]
            label = data['target']["label"]
            id = data['target']['id']

            img, label = img.cuda(), label.long().cuda()
            img, label = Variable(img), Variable(label)
            feat = self.G(img)
            # print('feature.shape:{}'.format(feat.shape))

            if batch_idx == 0:
                label_all = label.data.cpu().numpy().tolist()
                feature_all = feat.data.cpu().numpy()
            else:
                feature_all = np.ma.row_stack((feature_all, feat.data.cpu().numpy()))
                feature_all = feature_all.data
                label_all = label_all + label.data.cpu().numpy().tolist()

            # print(feature_all.shape)

            output1 = self.C1(feat)

            test_loss += F.nll_loss(output1, label).item()

            if epoch == self.save_epoch:
                if self.cams_out:
                    g_cam = GradCam(G=self.G, C1=self.C1, args=self.args)
                    trans = transforms.ToPILImage()
                    for i in range(len(label)):
                        samp_img = img[i, :, :, :]
                        print(samp_img.shape)
                        samp_lab = label[i]
                        samp_id = id[i]
                        cam = g_cam.generate_cam(samp_img, samp_lab)
                        cam_img_file = save_class_activation_images(trans(samp_img.cpu()), cam, samp_id,
                                                                    file_path="C:/Users/lauren erdman/Desktop/kidney_img/View_Labeling/OutputNov2020/cams_out/Inst_bs16/noM3SDA/")

            if self.batch_size == 1:
                # target_output["label"].extend([target_data["label"].item()])
                pred_vals.extend([self.softmax(output1.squeeze())[1].item()])
            else:
                # target_output["label"].extend(target_data["label"].tolist())
                pred_vals.extend(self.softmax(output1)[:, 1].tolist())
            pred1.extend(np.round(pred_vals))
            ids.extend(id)
            feats.extend(feature_all.tolist())

            # target_output["id"].extend(list(target_data["id"]))

            # print(target_output)

            # pred1 = output1.data.max(1)[1]
            # pred_vals = self.softmax(output1.data)[1]
            k = label.data.size()[0]
            correct1 += np.sum(label.tolist() == pred1)
            size += k
            test_pred_vals += pred_vals
            # ids +=
        np.savez('result_plot_sv_t', feature_all, label_all)

        loss = test_loss / size
        acc = correct1 / size
        pred = test_pred_vals
        labels = label.data
        # print(
        #     '\nTest set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.0f}%)  \n'.format(
        #         test_loss, correct1, size, 100. * correct1 / size))
        if save_model and epoch % self.save_epoch == 0:
            torch.save(self.G,
                       '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
            torch.save(self.C1,
                       '%s/%s_to_%s_model_epoch%s_C1.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
            torch.save(self.C2,
                       '%s/%s_to_%s_model_epoch%s_C2.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
        if record_file:
            compute_performance(loss_in=loss, pred_in=pred_vals, lab_in=label_all, id_in=ids, out_file_name=record_file,
                                feat_in=feats, epoch=epoch, data_split="Target_Test")

    def feat_all_domain(self, source_dict, target_dict):
        for my_key in source_dict.keys():
            source_dict[my_key]["feat"] = self.G(source_dict[my_key]["img"])
        target_dict["feat"] = self.G(target_dict["img"])

        return source_dict, target_dict

    def C1_all_domain(self, source_dict, target_dict):
        for my_key in source_dict.keys():
            source_dict[my_key]["C1"] = self.C1(source_dict[my_key]["feat"])
        target_dict["C1"] = self.C1(target_dict["feat"])
        return source_dict, target_dict

    def C2_all_domain(self, source_dict, target_dict):
        for my_key in source_dict.keys():
            source_dict[my_key]["C2"] = self.C2(source_dict[my_key]["feat"])
        target_dict["C2"] = self.C2(target_dict["feat"])
        return source_dict, target_dict

    def softmax_loss_all_domain(self, source_dict, output_key):
        """

        Args:
            source_dict: dictionary of the source data containing the output from G*C{1,2}
            output_key: C1 or C2 depending on the output being computed

        Returns:
            source_dict: updated source_dict with the given loss computed for each domain and
                added to that domain's key in the source under the sub-key output_key+"_loss"
        """

        criterion = nn.CrossEntropyLoss().cuda()
        for my_key in source_dict.keys():
            source_dict[my_key][output_key + "_loss"] = criterion(source_dict[my_key][output_key],
                                                                  source_dict[my_key]["label"])

        return source_dict

    def loss_all_domain(self, source_dict, target_dict, c_msda, bs):
        ## update dictionaries to include "feat" for each key
        source_dict, target_dict = self.feat_all_domain(source_dict=source_dict, target_dict=target_dict)
        ## update dictionaries to include "C1" for each key
        source_dict, target_dict = self.C1_all_domain(source_dict=source_dict, target_dict=target_dict)
        ## update dictionaries to include "C2" for each key
        source_dict, target_dict = self.C2_all_domain(source_dict=source_dict, target_dict=target_dict)
        ## compute MDSA loss
        loss_msda = 0.0005 * msda_regulizer(source_dict=source_dict, target_dict=target_dict, belta_moment=5, c_msda=c_msda, bs=bs)

        ## update source_dict to include "C1" loss
        source_dict = self.softmax_loss_all_domain(source_dict=source_dict, output_key="C1")
        ## update source_dict to include "C2" loss
        source_dict = self.softmax_loss_all_domain(source_dict=source_dict, output_key="C2")

        ## sum all losses
        loss_s_sum = 0
        loss_dict = {"C1": 0, "C2": 0}
        for classifier in loss_dict.keys():
            for my_key in source_dict.keys():
                loss_dict[classifier] = loss_dict[classifier] + source_dict[my_key][classifier + "_loss"]

        loss = sum(loss_dict.values()) + loss_msda

        return loss, loss_dict, source_dict, target_dict

    def train_MSDA(self, epoch, record_file=None):
        self.G.train()
        self.C1.train()
        self.C2.train()
        torch.cuda.manual_seed(1)

        source_output = {}
        target_output = {"loss": 0, "output": [], "label": [], "id": [], "feat": []}

        for batch_idx, data in enumerate(self.datasets):

            # print("Training batch sampled")
            source_data = data["source"]
            target_data = data["target"]

            # for my_key in source_data.keys():
                # print(my_key + " IDs: ")
                # print(source_data[my_key]['id'])

            dat_size_lt_batch = []

            for my_key in source_data.keys():
                # print(my_key)
                source_data[my_key]["img"] = Variable(source_data[my_key]["img"].cuda())
                source_data[my_key]["label"] = Variable(source_data[my_key]["label"].long().cuda())
                # print(source_data[my_key]["id"])
                dat_size_lt_batch.append(source_data[my_key]["img"].size()[0] < self.batch_size)

            target_data["img"] = Variable(target_data["img"].cuda())
            target_data["label"] = Variable(target_data["label"].long().cuda())

            if any(dat_size_lt_batch) or target_data["img"].size()[0] < self.batch_size:
                print("One of the drawn batches is smaller than the batch size.")
                break

            self.reset_grad()

            ## compute loss in source domains
            loss, _, source_data, target_data = self.loss_all_domain(source_dict=source_data, target_dict=target_data,
                                                                     c_msda=self.c_msda, bs=self.batch_size)
            # print("Loss computed")

            loss.backward()

            ## take a gradient step after classifying in source domains
            self.opt_g.step()
            self.opt_c1.step()
            self.opt_c2.step()
            self.reset_grad()

            ## recompute loss in source domains
            loss_s, loss_s_dict, source_data, target_data = self.loss_all_domain(source_dict=source_data,
                                                                                 target_dict=target_data,
                                                                                 c_msda=self.c_msda, bs=self.batch_size)

            for my_key in source_data.keys():
                source_output = self.update_output(output_dict=source_output, data_dict=source_data, in_key=my_key,
                                                   source=True, msda=True)
                # print(source_data[my_key]['id'])
                # print("source output")
                # print(source_output)

            ## compute features and classifications for target
            target_data["feat"] = self.G(target_data["img"])
            target_data["C1"] = self.C1(target_data["feat"])
            target_data["C2"] = self.C2(target_data["feat"])

            if self.no_discr:
                loss_dis = torch.tensor(0).to(self.device)
            else:
                loss_dis = self.discrepancy(target_data["C1"], target_data["C2"])
            # print("Discrepancy computed")

            loss = loss_s - loss_dis  ## minimizing discrepancy between task-specific classifiers for fixed G(*)
            loss.backward()
            self.opt_c1.step()
            self.opt_c2.step()
            self.reset_grad()

            if not self.no_discr:
                for i in range(4):  ## taking 4 steps maximizing discrepancy between C1 and C2
                                            ### is this number (4) a function of the number of sources?
                    target_data["feat"] = self.G(target_data["img"])
                    target_data["C1"] = self.C1(target_data["feat"])
                    target_data["C2"] = self.C2(target_data["feat"])
                    loss_dis = self.discrepancy(target_data["C1"], target_data["C2"])
                    loss_dis.backward()
                    self.opt_g.step()
                    self.reset_grad()

            if batch_idx > 500:
                return batch_idx

            target_data["loss"] = loss_dis
            target_output = self.update_output(output_dict=target_output, data_dict=target_data,
                                               source=False, msda=True)
        # print("source data keys: " + " ".join(list(source_data.keys())))
        for my_key in source_data.keys():
            print("computing performance")
            compute_performance(loss_in=source_output[my_key]["loss"],
                                pred_in=source_output[my_key]["output"],
                                lab_in=source_output[my_key]["label"],
                                feat_in=source_output[my_key]["feat"],
                                id_in=source_output[my_key]["id"],
                                out_file_name=record_file, epoch=epoch,
                                data_split="SourceTrain_" + my_key)

        compute_performance(loss_in=target_output["loss"],
                            pred_in=target_output["output"],
                            lab_in=target_output["label"],
                            feat_in=target_output["feat"],
                            id_in=target_output["id"],
                            out_file_name=record_file, epoch=epoch,
                            data_split="TargetTrain")

        return batch_idx

    def train_MSDA_source_classifiers(self, epoch, record_file=None):

        ## create modified "test" function to ensemble predictions from all classifiers

        self.G.train()
        self.C1.train()
        self.C2.train()
        torch.cuda.manual_seed(1)

        for batch_idx, data in enumerate(self.datasets):

            # print("Training batch sampled")
            source_data = data["source"]
            target_data = data["target"]

            dat_size_lt_batch = []
            for my_key in source_data.keys():
                # print(my_key)
                source_data[my_key]["img"] = Variable(source_data[my_key]["img"].cuda())
                source_data[my_key]["label"] = Variable(source_data[my_key]["label"].long().cuda())
                dat_size_lt_batch.append(source_data[my_key]["img"].size()[0] < self.batch_size)
            target_data["img"] = Variable(target_data["img"].cuda())
            target_data["label"] = Variable(target_data["label"].long().cuda())

            if any(dat_size_lt_batch) or target_data["img"].size()[0] < self.batch_size:
                print("One of the drawn batches is smaller than the batch size.")
                break

            self.reset_grad()

            ## compute loss in source domains
            loss, _ = self.loss_all_domain(source_dict=source_data, target_dict=target_data,
                                           record_file="EpochStart_" + record_file, epoch=epoch)
            # print("Loss computed")

            loss.backward()

            ## take a gradient step after classifying in source domains
            self.opt_g.step()
            self.opt_c1.step()
            self.opt_c2.step()
            self.reset_grad()

            ## recompute loss in source domains
            loss_s, loss_s_dict = self.loss_all_domain(source_dict=source_data,
                                                       target_dict=target_data, record_file=record_file,
                                                       epoch="SourceStep_"+epoch)

            ## compute features and classifications for target
            target_data["feat"] = self.G(target_data["img"])
            target_data["C1"] = self.C1(target_data["feat"])
            target_data["C2"] = self.C2(target_data["feat"])

            loss_dis = self.discrepancy(target_data["C1"], target_data["C2"])
            # print("Discrepancy computed")

            loss = loss_s - loss_dis  ## minimizing discrepancy between task-specific classifiers for fixed G(*)
            loss.backward()
            self.opt_c1.step()
            self.opt_c2.step()
            self.reset_grad()

            for i in range(4):  ## taking 4 steps maximizing discrepancy between C1 and C2
                target_data["feat"] = self.G(target_data["img"])
                target_data["C1"] = self.C1(target_data["feat"])
                target_data["C2"] = self.C2(target_data["feat"])
                loss_dis = self.discrepancy(target_data["C1"], target_data["C2"])
                loss_dis.backward()
                self.opt_g.step()
                self.reset_grad()
            if batch_idx > 500:
                return batch_idx

            ## recompute loss in source and target domains and write to file
            _, _ = self.loss_all_domain(source_dict=source_data, target_dict=target_data,
                                        record_file=record_file, epoch="TargetStep_"+epoch)

            self.reset_grad()
            ### SORT THIS OUT TO RUN WITH VARIABLE NUMBER OF SOURCES
            ## add each source, make this write to "LOSS_"+record_file only
            # if batch_idx % self.interval == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t  Discrepancy: {:.6f}'.format(
            #         epoch, batch_idx, 100,
            #         100. * batch_idx / 70000, loss_s_dict["C1"].item(), loss_s_dict["C2"].item(), loss_dis.item()))
            #     if record_file:
            #         record = open(record_file, 'a')
            #         record.write('%s %s %s\n' % (loss_dis.item(), loss_s_dict["C1"].item(), loss_s_dict["C2"].item()))
            #         record.close()

        return batch_idx


###
###
###        MAIN FUNCTION
###

def main():
    # add args:

    parser = argparse.ArgumentParser()
    parser.add_argument('-source', default="SickKids_Stanford3_samples",
                        help="US machine (data source) held out of training")
    parser.add_argument('-target', default="Stanford_S3000", help="US machine (data source) held out of training")
    parser.add_argument('-lab_col', default="kidney", help="Name of label column in datasheet")
    parser.add_argument('-lab_us_dir',
                        default='/Users/lauren erdman/Desktop/kidney_img/View_Labeling/images/all_lab_img/',
                        help="Directory of ultrasound images")

    # parser.add_argument("-dichot", action='store_true', default=False, help="Use dichotomous (vs continuous) outcome")
    # parser.add_argument("-RL", action='store_true', default=False, help="Include r/l labels or only build model on 4 labels")

    parser.add_argument("-run_lab", default="MSE_Model-noOther", help="String to add to output files")

        #### MAKE SMALLER FILE TO TEST THIS WITH -- WILL IT FIX THE PAGING FILE ERROR?
    parser.add_argument('-datasheet',
                        default='C:/Users/lauren erdman/Desktop/kidney_img/View_Labeling/datasheets/kidney_view_n500_sda_sksub_stsub_inst_spec_20210204.csv',
                        help="directory of DMSA images")  ## one datasheet: training and test specified a column of this datasheet

    parser.add_argument('-csv_outdir',
                        default='C:/Users/lauren erdman/Desktop/kidney_img/View_Labeling/OutputNov2020/csv/',
                        help="directory of DMSA images")
    parser.add_argument('-checkpoint_dir',
                        default='C:/Users/lauren erdman/Desktop/kidney_img/View_Labeling/OutputNov2020/checkpoint/',
                        help="directory of DMSA images")

    # parser.add_argument('-out_csv', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/data/load_training_test_sets/test_out.csv',
    #                     help="directory of DMSA images")

    # parser.add_argument("-include_val", action='store_true', default=True,
    #                     help="Use dichotomous (vs continuous) outcome")
    # parser.add_argument("-include_test", action='store_true', default=True,
    #                     help="Use dichotomous (vs continuous) outcome")
    # parser.add_argument("-dmsa_out", action='store_true', default=False,
    #                     help="Save NN?")
    parser.add_argument('--save_epoch', type=int, default=12, metavar='N',
                        help='when to restore the model')
    parser.add_argument('--use_abs_diff', action='store_true', default=False,
                        help='use absolute difference value as a measurement')
    parser.add_argument('--eval_only', action='store_true', default=False,
                        help='evaluation only option')
    parser.add_argument('--no_msda', action='store_true', default=False,
                        help='No MSDA alignment')
    parser.add_argument('--no_discr', action='store_true', default=False,
                        help='No MSDA alignment')
    parser.add_argument('--c_msda', action='store_true', default=False,
                        help='Conditional MDSA alignment')
    parser.add_argument('--cams_out', action='store_true', default=False,
                        help='Create grad_cams from test set')
    parser.add_argument('--max_epoch', type=int, default=15, metavar='N',
                        help='how many epochs')
    parser.add_argument('--one_step', action='store_true', default=False,
                        help='one step training with gradient reversal layer')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='save_model or not')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')

    # parser.add_argument("-save_pred", action='store_true', default=True,
    #                     help="Save NN?")
    # parser.add_argument("-get_kid_labels", action='store_true', default=False,
    #                     help="Save NN?")

    # parser.add_argument("-net_file", default='my_net.pth',
    #                     help="File to save NN to")

    parser.add_argument("-dim", default=256, help="Image dimensions")
    parser.add_argument('-device', default='cuda', help="device to run NN on")

    parser.add_argument("-lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("-mom", type=float, default=0.9, help="Momentum")
    parser.add_argument("-weight_decay", default=0.0005, help="Weight decay")
    parser.add_argument("-optimizer", default='adam', help="momentum or adam")

    opt = parser.parse_args()  ## comment for debug
    print(opt)

    ## from VisionLearningGroup.github.io M3SDA
    solver = Solver(args=opt)
    record_num = 0

    if opt.no_msda:
        descriptor = "noMSDA"
    elif opt.c_msda:
        descriptor = "cMSDA"
    else:
        descriptor = "MSDA"

    record_train = '%s/%s_%s_%s' % (
        opt.csv_outdir, descriptor, opt.target, record_num)
    record_test = '%s/%s_%s_%s_test' % (
        opt.csv_outdir, descriptor, opt.target, record_num)

    while os.path.exists(record_train+"_Detailed.json"):
        record_num += 1
        record_train = '%s/%s_%s_%s' % (
            opt.csv_outdir, descriptor, opt.target, record_num)
        record_test = '%s/%s_%s_%s_test' % (
            opt.csv_outdir, descriptor, opt.target, record_num)

    print("Record train: " + record_train)
    print("Record test: " + record_test)
    if not os.path.exists(opt.checkpoint_dir):
        os.mkdir(opt.checkpoint_dir)
    if not os.path.exists(opt.csv_outdir):
        os.mkdir(opt.csv_outdir)
    if opt.eval_only:
        solver.test(0)
    else:
        count = 0
        for t in range(opt.max_epoch):
            print('epoch: ' + str(t))
            if not opt.one_step:
                print("Training")
                if opt.no_msda:
                    num = solver.train(t, record_file=record_train)
                else:
                    print('Training MSDA')
                    num = solver.train_MSDA(t, record_file=record_train)
            else:
                print("Training one step")
                num = solver.train_onestep(t, record_file=record_train)
            count += num
            if t % 1 == 0:
                print("Testing")
                # print('epoch: ' + str(t))
                solver.test(t, record_file=record_test, save_model=opt.save_model)
            if count >= 20000:
                break


if __name__ == "__main__":
    main()
