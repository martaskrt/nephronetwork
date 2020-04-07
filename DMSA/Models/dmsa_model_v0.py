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


###
###        DEFINING DATASETS
###

class DMSADataset(Dataset):
    """ Data loader for DMSA data """

    def __init__(self,data_sheet,dichot = False):
        """
        data_sheet: csv spreadsheet with 1 column for each US view image file ("SR","SL","TR","TL","B"),
            1 column for each DMSA view image file (A/P) ("DMSA_A" and "DMSA_P"),
            2 label columns: 1 dichotomous and 1 continuous ("FUNC_DICH" and "FUNC_CONT"), and
            1 column for the instance id "ID"

        dichot: if set to True, a binary 0/1 value will be used as the label
            else if set to False, a continuous value for left function will be used as the label
        """

        self.id = data_sheet['ID']

        self.sr_file = data_sheet['SR']
        self.sl_file = data_sheet['SL']
        self.tr_file = data_sheet['TR']
        self.tl_file = data_sheet['TL']
        self.b_file = data_sheet['B']

        self.dmsa_a_file = data_sheet['DMSA_A']
        self.dmsa_p_file = data_sheet['DMSA_P']

        if dichot:
            self.label = data_sheet['FUNC_DICH']
        else:
            self.label = data_sheet['FUNC_CONT']

    def __len__(self):
        return len(self.id)

    def __getitem__(self, index):
            ## Maybe add file path as argument to init? Maybe "opt"?
        self.sr = Image.open(self.sr_file[index]).convert('L')
        self.sl = Image.open(self.sl_file[index]).convert('L')
        self.tr = Image.open(self.tr_file[index]).convert('L')
        self.tl = Image.open(self.tl_file[index]).convert('L')
        self.b = Image.open(self.b_file[index]).convert('L')
            ## make 5x256x256 tensor
        input_tensor = torch.tensor()

        self.dmsa_a = Image.open(self.dmsa_a_file[index]).convert('L')
        self.dmsa_p = Image.open(self.dmsa_p_file[index]).convert('L')
            ## make 2x256x256 tensor
        output_tensor = torch.tensor()

        out_label = torch.tensor(self.label[index])


        return input_tensor, output_tensor, out_label

