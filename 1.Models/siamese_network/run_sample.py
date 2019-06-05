import codecs
import errno
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
from PIL import Image
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import importlib.machinery
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import argparse
from torch.autograd import Variable
from SiameseNetworkUNet import SiamNet

# from FraternalSiameseNetwork import SiamNet
prep_single_sample = importlib.machinery.SourceFileLoader('prep_single_sample', '../../0.Preprocess/prep_single_sample.py').load_module()

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
softmax = torch.nn.Softmax(dim=1)

class KidneyDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        # self.X = torch.from_numpy(X).float().unsqueeze(0)
        self.X = torch.from_numpy(X).float()

    def __getitem__(self, index):
        imgs = self.X[index]
        return imgs

    def __len__(self):
        return len(self.X)

def run_sample(args):
    params = {'batch_size': 2,
              'shuffle': True,
              'num_workers': 1}

    prepped_image = prep_single_sample.load_image(args.sag_path, args.trans_path)
    prepped_image2 = prep_single_sample.load_image(args.sag_path, args.trans_path)
    X = [prepped_image, prepped_image2]

    prepped_image = np.array(X)
    # counter = 0
    # for view in prepped_image:
    #     scipy.misc.imsave("./{}.jpg".format(counter), view)
    #     counter += 1

    print(prepped_image.shape)
    image = KidneyDataset(prepped_image)


    data_generator = DataLoader(image, **params)

    net = SiamNet(output_dim=256).to(device)

    # pretrained_dict = torch.load(args.checkpoint, map_location='cpu')
    pretrained_dict = torch.load(args.checkpoint, map_location='cpu')['model_state_dict']
    model_dict = net.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    #pretrained_dict['fc6.fc6_s1.weight'] = pretrained_dict['fc6.fc6_s1.weight'].view(1024, 256, 2, 2)
    # for k, v in model_dict.items():
    #     if k not in pretrained_dict:
    #         pretrained_dict[k] = model_dict[k]
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    net.load_state_dict(pretrained_dict)

    with torch.set_grad_enabled(False):
        for batch_idx, (data) in enumerate(data_generator):
            net.zero_grad()
            # if len(data.shape) == 3:
            #     data = data.unsqueeze(0)
            output = net(data)
            print(output)
            output_softmax = softmax(output)
            pred_prob = output_softmax[:, 1]
            print("pred_prob: {}".format(pred_prob[0]))


def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--contrast", default=1, type=int, help="Image contrast to train on")
    #parser.add_argument("--view", default="siamese", help="siamese, sag, trans")
    parser.add_argument("--checkpoint", required=True, help="Path to load pretrained model checkpoint from")
    parser.add_argument("--sag_path", required=True, help="path to sag image")
    parser.add_argument("--trans_path", required=True, help="path to trans image")
    args = parser.parse_args()

    print("ARGS" + '\t' + str(args))

    run_sample(args)

if __name__ == '__main__':
    main()