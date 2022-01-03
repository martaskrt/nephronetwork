from sklearn.utils import shuffle
# from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
# import codecs
# import errno
# import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
# import random
import torch
from torch import nn
#from torch import optim
import torch.nn.functional as F
import torchvision.datasets.mnist
from torchvision import transforms
# from tqdm import tqdm
import importlib.machinery
from torch.utils.data import Dataset, DataLoader
# from torchsummary import summary
import argparse
from torch.autograd import Variable
from sklearn.utils import class_weight
import json
import math
import collections
from skimage.transform import resize
import random
import cv2

    ##
    ##  UTILITY FUNCTIONS
    ##

def flatten(lst):
    ## from https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
    return eval('[' + str(lst).replace('[', '').replace(']', '') + ']')

def load_dataset(json_infile, test_prop, ordered_split=False):
    with open(json_infile, 'r') as fp:
        in_dict = json.load(fp)

    if 'SU2bae8dc' in list(in_dict.keys()):
        in_dict.pop('SU2bae8dc', None)

    pt_ids = list(in_dict.keys())
    BL_dates = []
    for my_key in pt_ids:
        if 'BL_date' in in_dict[my_key].keys():
            BL_dates.extend([in_dict[my_key]['BL_date']])
        else:
            BL_dates.extend(['2021-01-01'])

    if ordered_split:
        sorted_pt_BL_dates = sorted(zip(pt_ids, BL_dates))
        test_n = round(len(pt_ids) * test_prop)
        train_out = dict()
        test_out = dict()

        for i in range(len(pt_ids) - test_n):
            study_id, _ = sorted_pt_BL_dates[i]
            train_out[study_id] = in_dict[study_id]

        for i in range(len(pt_ids) - test_n + 1, len(pt_ids)):
            study_id, _ = sorted_pt_BL_dates[i]
            test_out[study_id] = in_dict[study_id]

    else:
        shuf_pt_id, shuf_BL_dates = shuffle(list(pt_ids), BL_dates, random_state=42)
        test_n = round(len(shuf_pt_id)*test_prop)
        train_out = dict()
        test_out = dict()

        for i in range(test_n):
            study_id = shuf_pt_id[i]
            test_out[study_id] = in_dict[study_id]

        for i in range(test_n+1, len(shuf_pt_id)):
            study_id = shuf_pt_id[i]
            train_out[study_id] = in_dict[study_id]

    return train_out, test_out

process_results = importlib.machinery.SourceFileLoader('process_results','../../2.Results/process_results.py').load_module()

SEED = 42

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

    ##
    ## NN
    ##

class SiamNet(nn.Module):
    def __init__(self, classes=2, num_inputs=2, dropout_rate=0.5, output_dim=128, cov_layers=False):
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

    def forward(self, x):

        if self.cov_layers:
            in_dict = x
            x = in_dict['img']

        # x = x.unsqueeze(0)
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
            curr_x = curr_x.expand(-1, 3, -1, -1)
            if torch.cuda.is_available():
                input = torch.cuda.FloatTensor(curr_x.to(device))
            else:
                input = torch.FloatTensor(curr_x.to(device))
            z = self.conv(input)
            z = self.fc6(z)
            z = self.fc6b(z)
            z = z.view([B, 1, -1])
            z = self.fc6c(z)
            z = z.view([B, 1, -1])
            x_list.append(z)

        x = torch.cat(x_list, 1)
        # x = torch.sum(x, 1)
        x = self.fc7_new(x.view(B, -1))
        pred = self.classifier_new(x)

        if self.cov_layers:
            age = in_dict['Age_wks'].type(torch.FloatTensor).to(device).view(B, 1)
            # print("Age: ")
            # print(age)
            side = in_dict['Side_L'].type(torch.FloatTensor).to(device).view(B, 1)
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
softmax = torch.nn.Softmax(dim=1)

    ##
    ## DATA LOADER
    ##


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


def process_input_image(img_file):
    """
    Processes image: crop, convert to greyscale and resize

    :param image: image
    :return: formatted image
    """
    # print(img_file)
    image = np.array(Image.open(img_file).convert('L'))

    fit_img = pad_img_le(np.array(image))

    # cv2.imwrite(img_file, fit_img*255)
    # out_img.save(img_file)
    # fit_img = my_img

    return fit_img


def get_images(in_dict):
    img_dict = dict()
    label_dict = dict()
    cov_dict = dict()

    for study_id in in_dict.keys():

        try:
            sides = np.setdiff1d(list(in_dict[study_id].keys()), ['BL_date', 'Sex'])
            for side in sides:
                # print(study_id)
                # print(in_dict[study_id][side])

                if type(in_dict[study_id][side]) == dict:
                    if 'surgery' in list(in_dict[study_id][side].keys()):
                        if type(in_dict[study_id][side]['surgery']) == int:
                            surgery = in_dict[study_id][side]['surgery']
                        elif type(in_dict[study_id][side]['surgery']) == list:
                            s1 = [i for i in in_dict[study_id][side]['surgery'] if type(i) == int]
                            if(len(s1) > 0):
                                surgery = s1[0]
                            else:
                                surgery = None
                        elif type(in_dict[study_id][side]['surgery']) == str:
                            surgery = None
                    else:
                        surgery = None
                else:
                    surgery = None

                us_nums = [my_key for my_key in in_dict[study_id][side].keys() if my_key != 'surgery']
                if len(us_nums) > 0 and surgery is not None:
                    for us_num in us_nums:
                        if in_dict[study_id][side][us_num]['Age_wks'] != "NA" and \
                                set(['sag', 'trv']).issubset(in_dict[study_id][side][us_num].keys()):
                            img_dict[study_id+"_"+side+"_"+us_num] = dict()

                            img_dict[study_id + "_" + side + "_" + us_num]['sag'] = process_input_image(in_dict[study_id][side][us_num]['sag'])
                            img_dict[study_id + "_" + side + "_" + us_num]['trv'] = process_input_image(in_dict[study_id][side][us_num]['trv'])
                            # img_dict[study_id + "_" + side + "_" + us_num]['sag'] = np.array(Image.open(in_dict[study_id][side][us_num]['sag']).convert('L'))
                            # img_dict[study_id + "_" + side + "_" + us_num]['trv'] = np.array(Image.open(in_dict[study_id][side][us_num]['trv']).convert('L'))

                            label_dict[study_id+"_"+side+"_"+us_num] = surgery

                            cov_dict[study_id + "_" + side + "_" + us_num] = dict()
                            cov_dict[study_id+"_"+side+"_"+us_num]['US_machine'] = in_dict[study_id][side][us_num]['US_machine']
                            cov_dict[study_id+"_"+side+"_"+us_num]['Sex'] = in_dict[study_id]['Sex']
                            cov_dict[study_id+"_"+side+"_"+us_num]['Age_wks'] = in_dict[study_id][side][us_num]['Age_wks']
                            # print(label_dict)
                            # print(img_dict[study_id+"_"+side+"_"+us_num])
                            # print(cov_dict[study_id+"_"+side+"_"+us_num])
                            assert(type(label_dict[study_id+"_"+side+"_"+us_num]) is not None)
                            assert(type(img_dict[study_id+"_"+side+"_"+us_num]) is not None)
                            assert(type(cov_dict[study_id+"_"+side+"_"+us_num]['Sex']) is not None)
                            assert(type(cov_dict[study_id+"_"+side+"_"+us_num]['Age_wks']) is not None)

        except AttributeError:
            continue

    ids = img_dict.keys()

    return img_dict, label_dict, cov_dict, ids


class KidneyDataset(torch.utils.data.Dataset):

    def __init__(self, from_full_dict=True, in_dict=None, image_dict=None, label_dict=None, cov_dict=None, study_ids=None,
                 cov_input=False, rand_crop=False):
        if from_full_dict:
            self.image_dict, self.label_dict, self.cov_dict, self.study_ids = get_images(in_dict)
        else:
            self.image_dict, self.label_dict, self.cov_dict, self.study_ids = image_dict, label_dict, cov_dict, study_ids

        self.cov_input = cov_input
        self.rand_crop = rand_crop


    def __getitem__(self, index):

        # print(index)

        id = list(self.study_ids)[index]

        # print("ID: ")
        # print(id)

        img, target, cov = self.image_dict[id], self.label_dict[id], self.cov_dict[id]

        # print("Target: ")
        # print(target)

        target_out = torch.tensor(int(target)).to(device).type(torch.DoubleTensor)
        # print("Target converted to tensor.")

        sag_img = torch.tensor(img['sag']).to(device)
        trans_img = torch.tensor(img['trv']).to(device)

        if self.rand_crop:
            trans1 = torchvision.transforms.RandomCrop(200)
            sag_img = torchvision.transforms.functional.pad(trans1(sag_img), padding=28)
            trans_img = torchvision.transforms.functional.pad(trans1(trans_img), padding=28)

        img_out = torch.stack((sag_img, trans_img)).type(torch.FloatTensor)

        # print("Image converted to tensor.")

        if self.cov_input:
            id_split = id.split("_")

            dict_out = cov
            dict_out['img'] = img_out
            dict_out['Side_L'] = 1 if id_split[1] == 'Left' else 0

            if dict_out['Age_wks'] is None:
                dict_out['Age_wks'] = 36
            elif math.isnan(torch.tensor(dict_out['Age_wks']).type(torch.FloatTensor)):
                dict_out['Age_wks'] = 36

            # print("Side:")
            # print(dict_out['Side_L'])
            # print("Age:")
            # print(dict_out['Age_wks'])
            # print("target:")
            # print(target_out)
            # print("image")
            # print(dict_out["img"].shape)
            return dict_out, target_out, id
        else:
            return img_out, target_out, id

    def __len__(self):
        return len(self.study_ids)

    ##
    ## INITIALIZATION AND TRAINING
    ##


def init_weights(m):
    #if type(m) == nn.Linear:
    print(m.weight)
    torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #torch.nn.init.xavier_uniform(m.weight)
    m.bias.data.fill_(0.01)
    print(m.weight)


def train(args, train_dict, test_dict, max_epochs, cov_in, rand_crop=False):


    out_file_root = args.out_dir + "/SickKids_origST_" + str(max_epochs) + "epochs_bs" + \
                    str(args.batch_size) + "_lr" + str(args.lr) + \
                    "_RC" + str(args.random_crop) + "_cov" + str(cov_in) + "_OS" + str(args.ordered_split)
    out_file_name = out_file_root+".txt"

    outfile = open(out_file_name, 'w+')
    outfile.close()

    hyperparams = {'lr': args.lr,
                   'momentum': args.momentum,
                   'weight_decay': args.weight_decay,
                   'train/test_split': args.split 
                  }
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.num_workers}

    test_set = KidneyDataset(in_dict=test_dict, cov_input=cov_in)
    test_generator = DataLoader(test_set, num_workers=0, batch_size=16)

    fold = 1
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # train_y = np.array(train_y)
    # train_cov = np.array(train_cov)
    train_img_dict, train_label_dict, train_cov_dict, train_study_ids = get_images(train_dict)

    train_study_ids = shuffle(list(train_study_ids), random_state=42)

    model_output = {"train": {str(j+1): {str(k): dict() for k in range(max_epochs)} for j in range(n_splits)},
                    "val": {str(j+1): {str(k): dict() for k in range(max_epochs)} for j in range(n_splits)},
                    "test": {str(j+1): {str(k): dict() for k in range(max_epochs)} for j in range(n_splits)}}
    split = 0
    for train_idx, val_idx in skf.split(train_study_ids, list(train_label_dict.values())):
        split += 1

        if args.view != "siamese":
            net = SiamNet(num_inputs=1, output_dim=args.output_dim, cov_layers=cov_in).to(device)
        else:
            net = SiamNet(output_dim=args.output_dim, cov_layers=cov_in).to(device)
        net.zero_grad()
        #net.apply(init_weights)

        ## come back to this
        if args.checkpoint != "":
            #pretrained_dict = torch.load(args.checkpoint)
            pretrained_dict = torch.load(args.checkpoint)['model_state_dict']
            model_dict = net.state_dict()
            print("loading checkpoint..............")
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k not in ["fc6c.fc7.weight","fc6c.fc7.bias", "fc7_new.fc7.weight", "fc7_new.fc7.bias", "classifier_new.fc8.weight", "classifier_new.fc8.bias"]}

            pretrained_dict['conv1.conv1_s1.weight'] = pretrained_dict['conv1.conv1_s1.weight'].mean(1).unsqueeze(1)
            #pretrained_dict['fc6.fc6_s1.weight'] = pretrained_dict['fc6.fc6_s1.weight'].view(1024, 256, 2, 2)
            #for k, v in model_dict.items():
             #   if k not in pretrained_dict:
              #      pretrained_dict[k] = model_dict[k]
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            net.load_state_dict(model_dict)
            print("Checkpoint loaded...............")

        if args.adam:
            optimizer = torch.optim.Adam(net.parameters(), lr=hyperparams['lr'],
                                         weight_decay=hyperparams['weight_decay'])
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=hyperparams['lr'], momentum=hyperparams['momentum'],
                                        weight_decay=hyperparams['weight_decay'])

        print("Splitting Training/Validation data")

        train_ids = [train_study_ids[k] for k in train_idx]
        val_ids = [train_study_ids[k] for k in val_idx]
        _, _, _, test_ids = get_images(test_dict)

        print(test_ids)

        train_imgs_d = {train_id: train_img_dict[train_id] for train_id in train_ids}
        val_imgs_d = {val_id: train_img_dict[val_id] for val_id in val_ids}

        train_lab_d = {train_id: train_label_dict[train_id] for train_id in train_ids}
        val_lab_d = {val_id: train_label_dict[val_id] for val_id in val_ids}

        train_cov_d = {train_id: train_cov_dict[train_id] for train_id in train_ids}
        val_cov_d = {val_id: train_cov_dict[val_id] for val_id in val_ids}

        training_set = KidneyDataset(from_full_dict=False, image_dict=train_imgs_d, label_dict=train_lab_d, cov_dict=train_cov_d, study_ids=train_ids, rand_crop=rand_crop, cov_input=cov_in)
        training_generator = DataLoader(training_set, shuffle=True, num_workers=0, batch_size=16)

        ### DEBUGGING DATA LOADER FEB 22, 2022
        # for my_key in train_imgs_d.keys():
        #     if train_imgs_d[my_key]['trv'].shape != (256, 256):
        #         print(train_imgs_d[my_key]['trv'].shape)
        #
        # for my_key in train_lab_d.keys():
        #     if train_lab_d[my_key] != 1 & train_lab_d[my_key] != 0:
        #         print(train_lab_d[my_key])
        #
        # for my_key in train_cov_d.keys():
        #     print(train_cov_d[my_key]['Sex'])
        #     print(train_cov_d[my_key]['Age_wks'])
        #
        #
        # set(train_imgs_d.keys()) == set(train_lab_d.keys())
        # set(train_imgs_d.keys()) == set(train_cov_d.keys())
        #
        # set(training_set.image_dict.keys()) == set(training_set.label_dict.keys())
        # training_set.cov_dict.keys()


        print("Training set ready.")

        validation_set = KidneyDataset(from_full_dict=False, image_dict=val_imgs_d, label_dict=val_lab_d, cov_dict=val_cov_d, study_ids=val_ids, rand_crop=False, cov_input=cov_in)
        validation_generator = DataLoader(validation_set, num_workers=0, batch_size=16)

        print("Validation set ready.")

    ## run all epochs on given split of the data
        for epoch in range(max_epochs):
            print("Epoch: " + str(epoch) + " running.")
            accurate_labels_train = 0
            accurate_labels_val = 0
            accurate_labels_test = 0

            loss_accum_train = 0
            loss_accum_val = 0
            loss_accum_test = 0

            all_targets_train = []
            all_pred_prob_train = []
            all_pred_label_train = []

            all_targets_val = []
            all_pred_prob_val = []
            all_pred_label_val = []

            all_targets_test = []
            all_pred_prob_test = []
            all_pred_label_test = []

            all_train_ids = []
            all_val_ids = []
            all_test_ids = []

            patient_ID_train = []
            patient_ID_val = []
            patient_ID_test = []

            counter_train = 0
            counter_val = 0
            counter_test = 0
            net.train()

            ## Run training set
            for batch_idx, (data, target, id) in enumerate(training_generator):
                # print("Training batch drawn.")
                optimizer.zero_grad()
                #net.train() # 20190619
                # print("Running network.")
                output = net(data)
                # print("Target prepared.")
                target = Variable(target.type(torch.LongTensor), requires_grad=False).to(device)
                #print(output.shape, target.shape)

                # print("Output: ")
                # print(output)
                # print("Target: ")
                # print(target)

                if len(output.shape) == 1:
                    output = output.unsqueeze(0)
                loss = F.cross_entropy(output, target)
                #loss = cross_entropy(output, target)
                # print("Loss: ")
                # print(loss)
                loss_accum_train += loss.item() * len(target)

                loss.backward()

                accurate_labels_train += torch.sum(torch.argmax(output, dim=1) == target).cpu()
                optimizer.step()
                counter_train += len(target)

                output_softmax = softmax(output)
                #output_softmax = output
                pred_prob = output_softmax[:, 1]
                pred_label = torch.argmax(output_softmax, dim=1)

                # print("Training")
                # print(target)
                # print(pred_prob)
                # print(len(target))
                # print(len(pred_prob))

                # print(pred_label)
                assert len(pred_prob) == len(target)
                assert len(pred_label) == len(target)


                all_pred_prob_train.append(pred_prob)
                all_targets_train.append(target)
                all_pred_label_train.append(pred_label)
                patient_ID_train.append(list(id))

            net.eval()

            ## Run val set
            with torch.no_grad():
            #with torch.set_grad_enabled(False):
                for batch_idx, (data, target, id) in enumerate(validation_generator):
                    # print("Validation batch drawn.")
                    net.zero_grad()
                    #net.eval() # 20190619
                    optimizer.zero_grad()
                    output = net(data)
                    target = target.type(torch.LongTensor).to(device)

                    loss = F.cross_entropy(output, target)
                    #loss = cross_entropy(output, target)
                    loss_accum_val += loss.item() * len(target)
                    counter_val += len(target)
                    #output_softmax = output
                    output_softmax = softmax(output)

                    accurate_labels_val += torch.sum(torch.argmax(output, dim=1) == target).cpu()

                    pred_prob = output_softmax[:, 1]
                    pred_prob = pred_prob.squeeze()
                    pred_label = torch.argmax(output, dim=1)

                    # print("Validating")
                    # print(target)
                    # print(pred_prob)
                    # print(len(target))
                    # print(len(pred_prob))

                    assert len(pred_prob) == len(target)
                    assert len(pred_label) == len(target)

                    all_pred_prob_val.append(pred_prob)
                    all_targets_val.append(target)
                    all_pred_label_val.append(pred_label)
                    patient_ID_val.append(list(id))
            net.eval()

            ## Run test set
            with torch.no_grad():
            #with torch.set_grad_enabled(False):
                for batch_idx, (data, target, id) in enumerate(test_generator):
                    # print("Test batch drawn.")
                    net.zero_grad()
                    net.eval() # 20190619
                    optimizer.zero_grad()
                    output = net(data)
                    target = target.type(torch.LongTensor).to(device)

                    loss = F.cross_entropy(output, target)
                    #loss = cross_entropy(output, target)
                    loss_accum_test += loss.item() * len(target)
                    counter_test += len(target)
                    output_softmax = softmax(output)
                    #output_softmax = output
                    accurate_labels_test += torch.sum(torch.argmax(output, dim=1) == target).cpu()

                    pred_prob = output_softmax[:, 1]
                    pred_prob = pred_prob.squeeze()
                    pred_label = torch.argmax(output, dim=1)

                    # print("Testing")
                    # print(target)
                    # print(pred_prob)
                    # print(len(target))
                    # print(len(pred_prob))

                    if pred_prob.shape == torch.Size([]):
                        pred_prob = pred_prob.unsqueeze(0)

                    assert len(pred_prob) == len(target)
                    assert len(pred_label) == len(target)

                    all_pred_prob_test.append(pred_prob)
                    all_targets_test.append(target)
                    all_pred_label_test.append(pred_label)

                    patient_ID_test.append(list(id))

            all_pred_prob_train = torch.cat(all_pred_prob_train)
            all_targets_train = torch.cat(all_targets_train)
            all_pred_label_train = torch.cat(all_pred_label_train)
            all_train_ids = flatten(patient_ID_train)

            model_output['train'][str(split)][str(epoch)]['id'] = all_train_ids
            model_output['train'][str(split)][str(epoch)]['pred'] = all_pred_prob_train.tolist()
            model_output['train'][str(split)][str(epoch)]['target'] = all_targets_train.tolist()

           # patient_ID_train = torch.cat(patient_ID_train)

            assert len(all_targets_train) == len(training_set)
            assert len(all_pred_prob_train) == len(training_set)
            assert len(all_pred_label_train) == len(training_set)
            # assert len(patient_ID_train) == len(training_set)

            results_train = process_results.get_metrics(y_score=all_pred_prob_train.cpu().detach().numpy(),
                                                  y_true=all_targets_train.cpu().detach().numpy(),
                                                  y_pred=all_pred_label_train.cpu().detach().numpy())

            print('Fold\t{}\tTrainEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch, int(accurate_labels_train)/counter_train,
                                                                        loss_accum_train/counter_train, results_train['auc'],
                                                                        results_train['auprc'], results_train['tn'],
                                                                        results_train['fp'], results_train['fn'],
                                                                        results_train['tp']))
            outfile = open(out_file_name, 'a')
            outfile.write('\nFold\t{}\tTrainEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch, int(accurate_labels_train)/counter_train,
                                                                        loss_accum_train/counter_train, results_train['auc'],
                                                                        results_train['auprc'], results_train['tn'],
                                                                        results_train['fp'], results_train['fn'],
                                                                        results_train['tp']))
            outfile.close()

            all_pred_prob_val = torch.cat(all_pred_prob_val)
            all_targets_val = torch.cat(all_targets_val)
            all_pred_label_val = torch.cat(all_pred_label_val)
            all_val_ids = flatten(patient_ID_val)

            model_output['val'][str(split)][str(epoch)]['id'] = all_val_ids
            model_output['val'][str(split)][str(epoch)]['pred'] = all_pred_prob_val.tolist()
            model_output['val'][str(split)][str(epoch)]['target'] = all_targets_val.tolist()

            #patient_ID_val = torch.cat(patient_ID_val)

            assert len(all_targets_val) == len(validation_set)
            assert len(all_pred_prob_val) == len(validation_set)
            assert len(all_pred_label_val) == len(validation_set)
            # assert len(patient_ID_val) == len(validation_set)

            results_val = process_results.get_metrics(y_score=all_pred_prob_val.cpu().detach().numpy(),
                                                      y_true=all_targets_val.cpu().detach().numpy(),
                                                      y_pred=all_pred_label_val.cpu().detach().numpy())
            print('Fold\t{}\tValEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch, int(accurate_labels_val) / counter_val,
                                                                         loss_accum_val / counter_val, results_val['auc'],
                                                                         results_val['auprc'], results_val['tn'],
                                                                         results_val['fp'], results_val['fn'],
                                                                         results_val['tp']))
            outfile = open(out_file_name, 'a')
            outfile.write('\nFold\t{}\tValEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch, int(accurate_labels_val) / counter_val,
                                                                         loss_accum_val / counter_val, results_val['auc'],
                                                                         results_val['auprc'], results_val['tn'],
                                                                         results_val['fp'], results_val['fn'],
                                                                         results_val['tp']))
            outfile.close()

            all_pred_prob_test = torch.cat(all_pred_prob_test)
            all_targets_test = torch.cat(all_targets_test)
            all_pred_label_test = torch.cat(all_pred_label_test)
            all_test_ids = flatten(patient_ID_test)

            model_output['test'][str(split)][str(epoch)]['id'] = all_test_ids
            model_output['test'][str(split)][str(epoch)]['pred'] = all_pred_prob_test.tolist()
            model_output['test'][str(split)][str(epoch)]['target'] = all_targets_test.tolist()

            # patient_ID_test = torch.cat(patient_ID_test)

            assert len(all_targets_test) == len(test_set)
            assert len(all_pred_label_test) == len(test_set)
            assert len(all_pred_prob_test) == len(test_set)
            # assert len(patient_ID_test) == len(test_set)

            results_test = process_results.get_metrics(y_score=all_pred_prob_test.cpu().detach().numpy(),
                                                      y_true=all_targets_test.cpu().detach().numpy(),
                                                      y_pred=all_pred_label_test.cpu().detach().numpy())
            print('Fold\t{}\tTestEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch, int(accurate_labels_test) / counter_test,
                                                                         loss_accum_test / counter_test, results_test['auc'],
                                                                         results_test['auprc'], results_test['tn'],
                                                                         results_test['fp'], results_test['fn'],
                                                                         results_test['tp']))
            outfile = open(out_file_name, 'a')
            outfile.write('\nFold\t{}\tTestEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                          'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch, int(accurate_labels_test) / counter_test,
                                                                         loss_accum_test / counter_test, results_test['auc'],
                                                                         results_test['auprc'], results_test['tn'],
                                                                         results_test['fp'], results_test['fn'],
                                                                         results_test['tp']))
            outfile.close()

           
            # if ((epoch+1) % 5) == 0 and epoch > 0:
            checkpoint = {'epoch': epoch,
                          'loss': loss,
                          'hyperparams': hyperparams,
                          'args': args,
                          'model_state_dict': net.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'loss_train': loss_accum_train / counter_train,
                          'loss_val': loss_accum_val / counter_val,
                          'loss_test': loss_accum_test / counter_test,
                          'accuracy_train': int(accurate_labels_train) / counter_train,
                          'accuracy_val': int(accurate_labels_val) / counter_val,
                          'accuracy_test': int(accurate_labels_test) / counter_test,
                          'results_train': results_train,
                          'results_val': results_val,
                          'results_test': results_test,
                          'all_pred_prob_train': all_pred_prob_train,
                          'all_pred_prob_val': all_pred_prob_val,
                          'all_pred_prob_test': all_pred_prob_test,
                          'all_targets_train': all_targets_train,
                          'all_targets_val': all_targets_val,
                          'all_targets_test': all_targets_test,
                          'patient_ID_train': patient_ID_train,
                          'patient_ID_val': patient_ID_val,
                          'patient_ID_test': patient_ID_test}

            if not os.path.isdir(args.dir):
                os.makedirs(args.dir)
            #if not os.path.isdir(args.dir + "/" + str(fold)):
                #os.makedirs(args.dir + "/" + str(fold))

            ## TO SAVE THE CHECKPOINT
            if epoch == args.epoch_save:
                path_to_checkpoint = out_file_root + '.pth'
                torch.save(checkpoint, path_to_checkpoint)
            
        fold += 1

    with open(out_file_root + ".json", "w") as fp:
        json.dump(model_output, fp, indent=4)

    ##
    ## MAIN FUNCTION
    ##

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=47, type=int, help="Number of epochs")
    parser.add_argument('--epoch_save', default=None, type=int, help="Number of epochs at which to save the model")
    parser.add_argument('--batch_size', default=16, type=int, help="Batch size")
    parser.add_argument('--lr', default=0.001, type=float, help="Learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, help="Momentum")
    parser.add_argument('--adam', action="store_true", help="Use Adam optimizer instead of SGD")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay")
    parser.add_argument("--num_workers", default=1, type=int, help="Number of CPU workers")
    parser.add_argument("--dir", default="./", help="Directory to save model checkpoints to")
    parser.add_argument("--contrast", default=1, type=int, help="Image contrast to train on")
    parser.add_argument("--view", default="siamese", help="siamese, sag, trans")
    parser.add_argument("--checkpoint", default="", help="Path to load pretrained model checkpoint from")
    parser.add_argument("--split", default=0.9, type=float, help="proportion of dataset to use as training")
    parser.add_argument('--ordered_split', action="store_true", default=False, help="Use Adam optimizer instead of SGD")
    parser.add_argument('--random_crop', action="store_true", default=False, help="Use Adam optimizer instead of SGD")
    parser.add_argument("--bottom_cut", default=0.0, type=float, help="proportion of dataset to cut from bottom")
    parser.add_argument("--etiology", default="B", help="O (obstruction), R (reflux), B (both)")
    parser.add_argument('--unet', action="store_true", help="UNet architecthure")
    parser.add_argument("--crop", default=0, type=int, help="Crop setting (0=big, 1=tight)")
    parser.add_argument("--sc", default=5, type=int, help="number of skip connections for unet (0, 1, 2, 3, 4, or 5)")
    parser.add_argument("--init", default="none")
    parser.add_argument("--hydro_only", action="store_true")
    parser.add_argument("--output_dim", default=256, type=int, help="output dim for last linear layer")
    parser.add_argument("--gender", default=None, type=str, help="choose from 'male' and 'female'")
    parser.add_argument("--json_infile", default="C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_SickKidswST_filenames_20210411.json",
                        help="Json file of all our data")
    # parser.add_argument("--json_infile", default="C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_SK_ST_filenames_20210217.json",
    #                     help="Json file of all our data")
    parser.add_argument('--cov_in', action="store_true", default=False, help="Use age at ultrasound and kidney side (R/L) as covariates in the model")
    parser.add_argument("--out_dir", default="C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/orig_st_results/", help="Directory to save model checkpoints to")

    args = parser.parse_args()

    print("ARGS" + '\t' + str(args))

    train_dict, test_dict = load_dataset(args.json_infile, test_prop=0.2, ordered_split=args.ordered_split)
    print("Data loaded.")

    train(args, train_dict=train_dict, test_dict=test_dict, max_epochs=args.epochs, rand_crop=args.random_crop, cov_in=args.cov_in)

    # if args.view == "sag" or args.view == "trans":
    #     train_X_single=[]
    #     test_X_single=[]
    #
    #     for item in train_X:
    #         if args.view == "sag":
    #             train_X_single.append(item[0])
    #         elif args.view == "trans":
    #             train_X_single.append(item[1])
    #     for item in test_X:
    #         if args.view == "sag":
    #             test_X_single.append(item[0])
    #         elif args.view == "trans":
    #             test_X_single.append(item[1])
    #
    #     train_X=train_X_single
    #     test_X=test_X_single
    #     train_X=np.array(train_X_single)
    #     test_X=np.array(test_X_single)
    #
    #
    # print(len(train_X), len(train_y), len(train_cov), len(test_X), len(test_y), len(test_cov))
    # train(args, train_X, train_y, train_cov, test_X, test_y, test_cov, max_epochs)


if __name__ == '__main__':
    main()
