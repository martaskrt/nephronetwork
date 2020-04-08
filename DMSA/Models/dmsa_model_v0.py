import numpy as np
import os
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import importlib.machinery
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd


    ###
    ###        HELPER FUNCTIONS
    ###

def read_image_file(file_name):

    """
    Reads in image file, crops it (if needed), converts it to greyscale, and returns image
    :param file_name: path and name of file

    :return: Torch tensor of image
    """
    return torch.from_numpy(np.asarray(Image.open(file_name).convert('L')))

    ###
    ###        DEFINING DATASETS
    ###

    ###
    ###        DATASET
    ###

class DMSADataset(Dataset):
    """ Data loader for DMSA data """

    def __init__(self, data_sheet, args, dim=256):
        """
        data_sheet: csv spreadsheet with 1 column for each US view image file ("SR","SL","TR","TL","B"),
            1 column for each DMSA view image file (A/P) ("DMSA_A" and "DMSA_P"),
            2 label columns: 1 dichotomous and 1 continuous ("FUNC_DICH" and "FUNC_CONT"), and
            1 column for the instance id "ID"

        args: argument dictionary
            args.dichot: if set to True, a binary 0/1 value will be used as the label
            else if set to False, a continuous value for left function will be used as the label
        """
        self.dim = dim

        self.rootdir = args.rootdir

        self.id = data_sheet['ID']

        self.sr_file = data_sheet['SR']
        self.sl_file = data_sheet['SL']
        self.tr_file = data_sheet['TR']
        self.tl_file = data_sheet['TL']
        self.b_file = data_sheet['B']

        self.dmsa_a_file = data_sheet['DMSA_A']
        self.dmsa_p_file = data_sheet['DMSA_P']

        if args.dichot:
            self.label = data_sheet['FUNC_DICH']
        else:
            self.label = data_sheet['FUNC_CONT']

    def __len__(self):
        return len(self.id)

    def __getitem__(self, index):
            ## Maybe add file path as argument to init? Maybe "opt"?
        sr = read_image_file(self.rootdir + self.sr_file[index]).view(1, self.dim, self.dim)
        sl = read_image_file(self.rootdir + self.sl_file[index]).view(1, self.dim, self.dim)
        tr = read_image_file(self.rootdir + self.tr_file[index]).view(1, self.dim, self.dim)
        tl = read_image_file(self.rootdir + self.tl_file[index]).view(1, self.dim, self.dim)
        b = read_image_file(self.rootdir + self.b_file[index]).view(1, self.dim, self.dim)
            ## make 5x256x256 tensor
        input_tensor = torch.cat((sr, sl, tr, tl, b), 0)

        dmsa_a = read_image_file(self.rootdir + self.dmsa_a_file[index]).view(1, self.dim, self.dim)
        dmsa_p = read_image_file(self.rootdir + self.dmsa_p_file[index]).view(1, self.dim, self.dim)
            ## make 2x256x256 tensor
        output_tensor = torch.cat((dmsa_a, dmsa_p),0)

        out_label = torch.tensor(self.label[index])

        return input_tensor, output_tensor, out_label


## to use dataset:

## args for debugging :)
class make_opt():
    def __init__(self):
        self.rootdir = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Image organization Nov 2019/'
        # self.rootdir = '/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/data/load_training_test_sets/'
        self.dichot = False # for a 0/1 outcome, set to True

opt = make_opt()
my_datasheet=pd.read_csv(opt.rootdir + 'DMSA-datasheet-top3view.csv')
my_data = DMSADataset(data_sheet=my_datasheet, args=opt)
my_dataloader = DataLoader(my_data, shuffle=True)

