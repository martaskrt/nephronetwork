import numpy as np
import os
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
# from torchvision import transforms
# import importlib.machinery
import torchvision.models as models
# from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from PIL import Image
import pandas as pd
import argparse
import datetime
from sklearn.metrics import roc_auc_score


    ###
    ###        HELPER FUNCTIONS
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

        ## check if machine is the target domain (this will make it easy to rotate targets)
        if row['machine'] == args.target:
                ## split machine-dictionary into train and test
            if row['test'] == 0:
                        ## read_image_file returns 256x256 (args.dim) torch tensor
                target_dict["img"][row['IMG_ID']] = read_image_file(args.lab_us_dir + "/" + row['IMG_FILE']).view(1, args.dim, args.dim)
                target_dict["labels"][row['IMG_ID']] = row[args.lab_col]
            else:
                target_test_dict["img"][row['IMG_ID']] = read_image_file(args.lab_us_dir + "/" + row['IMG_FILE']).view(1, args.dim, args.dim)
                target_test_dict["labels"][row['IMG_ID']] = row[args.lab_col]

        else:
            if row['test'] == 0:
                source_dict[row['machine']]["img"][row['IMG_ID']] = read_image_file(
                    args.lab_us_dir + "/" + row['IMG_FILE']).view(1, args.dim, args.dim)
                source_dict[row['machine']]["labels"][row['IMG_ID']] = row[args.lab_col]

                # source_dict[row['machine']]["img"][row['IMG_ID']], source_dict[row['machine']]["labels"][row['IMG_ID']] = \
                #     read_image_file(args.lab_us_dir + "/" + row['IMG_FILE']).view(1, args.dim, args.dim), row[args.lab_col]

            else:
                source_test_dict[row['machine']]["img"][row['IMG_ID']] = read_image_file(
                    args.lab_us_dir + "/" + row['IMG_FILE']).view(1, args.dim, args.dim)
                source_test_dict[row['machine']]["labels"][row['IMG_ID']] = row[args.lab_col]

    print("Data dictionaries created.")

    return source_dict, source_test_dict, target_dict, target_test_dict


## ORIGINALLY FROM VisionLearningGroup.github.io
def dataset_read(args):

    ### TRAINING AND TEST DATA
    S, S_test, T, T_test = return_dataset(args=args)

    scale = 32

    train_loader = UnalignedDataLoader()
    train_loader.initialize(S, T, args.batch_size, args.batch_size, scale=scale)
    dataset = train_loader.load_data()

    test_loader = UnalignedDataLoader()
    test_loader.initialize(S_test, T_test, args.batch_size, args.batch_size, scale=scale)

    dataset_test = test_loader.load_data()

    return dataset, dataset_test


## ORIGINALLY FROM VisionLearningGroup.github.io
def euclidean(x1, x2, ep=1e-7):
    euc = ((x1 - x2) ** 2).sum() + torch.tensor(ep)
    return euc.sqrt()


## ORIGINALLY FROM VisionLearningGroup.github.io
def k_moment(source_dict, target_dict, k):

    eucl_sum = 0
    target_k_moment_mean = (target_dict["centered_feat"] ** k).mean(0)
    for my_key in source_dict.keys():
        source_dict[my_key]["k_moment_mean"] = (source_dict[my_key]["centered_feat"]**k).mean(0)
        eucl_sum = eucl_sum + euclidean(target_k_moment_mean, source_dict[my_key]["k_moment_mean"])

    key_list = list(source_dict.keys())
    for i in range(len(source_dict)):
        for j in range(i):
            eucl_sum = eucl_sum + euclidean(source_dict[key_list[i]]["k_moment_mean"],
                                            source_dict[key_list[j]]["k_moment_mean"])

    return eucl_sum


## ORIGINALLY FROM VisionLearningGroup.github.io
def msda_regulizer(source_dict, target_dict, belta_moment): ## CHECK BELTA_MOMENT -- DOES IT CORRESPOND WITH THE NUMBER OF SOURCES?
    # print('s1:{}, s2:{}, s3:{}, s4:{}'.format(output_s1.shape, output_s2.shape, output_s3.shape, output_t.shape))

    eucl_sum = 0

    target_dict["centered_feat"] = target_dict["feat"] - target_dict["feat"].mean(0)
    for my_key in source_dict.keys():
        source_dict[my_key]["centered_feat"] = source_dict[my_key]["feat"] - source_dict[my_key]["feat"].mean(0)
        eucl_sum = eucl_sum + euclidean(source_dict[my_key]["centered_feat"], target_dict["centered_feat"])

    ## double check this loop with a toy model
    key_list = list(source_dict.keys())
    for i in range(len(source_dict)):
        for j in range(i):
            eucl_sum = eucl_sum + euclidean(source_dict[key_list[i]]["centered_feat"],
                                            source_dict[key_list[j]]["centered_feat"])

    reg_info = eucl_sum
    # print(reg_info)
    for i in range(belta_moment - 1):
        reg_info += k_moment(source_dict=source_dict, target_dict=target_dict, k=i+2)

    return reg_info / 6




# return euclidean(output_s1, output_t)


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

        img, target = self.data[self.keys[index]], self.labels[self.keys[index]]
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
        return img, target

    def __len__(self):
        return len(self.data)


## MODIFIED FROM VisionLearningGroup.github.io
class UnalignedDataLoader:
    def initialize(self, source, target, batch_size1, batch_size2, scale=32):

        ## check these transformations
        # transform = transforms.Compose([
        #     transforms.Scale(scale),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])

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

        self.paired_data = PairedData(source_data_loader_dict, target_data_loader, float("inf"))

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
    def __init__(self, data_loader_s, data_loader_t, max_dataset_size):
        """

        Args:
            data_loader_s: dictionary of source data loaders
            data_loader_t: target data loader (not a dictionary)
            max_dataset_size:
        """
        ## data loaders
        self.data_loader_s = data_loader_s
        self.data_loader_t = data_loader_t

        ## stops
        self.stop_dict = dict()
        for key in self.data_loader_s.keys():
            self.stop_dict[key] = False
        self.stop_t = False

        self.max_dataset_size = max_dataset_size

    def __iter__(self):
        self.stop_s = dict()
        self.data_loader_s_iter = dict()
        for key in self.data_loader_s.keys():
            self.data_loader_s_iter[key] = iter(self.data_loader_s[key])
            self.stop_s[key] = False

        self.data_loader_t_iter = iter(self.data_loader_t)
        self.stop_t = False
        self.iter = 0

        return self

    def __next__(self):

        return_dat_s = dict.fromkeys(self.data_loader_s.keys(), dict.fromkeys(['img', 'label'], None))
        for key in self.data_loader_s.keys():
            # return_dat_s[key]["img"], return_dat_s[key]["label"] = None, None
            try:
                return_dat_s[key]["img"], return_dat_s[key]["label"] = next(iter(self.data_loader_s[key]))
            except StopIteration:
                if return_dat_s[key]["img"] is None or return_dat_s[key]["label"] is None:
                    self.stop_s[key] = True
                    self.data_loader_s_iter[key] = iter(self.data_loader_s[key])
                    return_dat_s[key]["img"], return_dat_s[key]["label"] = next(self.data_loader_s_iter[key])

        t, t_paths = None, None
        try:
            t, t_paths = next(self.data_loader_t_iter)
        except StopIteration:
            if t is None or t_paths is None:
                self.stop_t = True
                self.data_loader_t_iter = iter(self.data_loader_t)
                t, t_paths = next(self.data_loader_t_iter)

        if any(self.stop_s.values()) or self.stop_t or self.iter > self.max_dataset_size:
            for key in self.stop_s:
                self.stop_s[key] = False
            self.stop_t = False
            raise StopIteration()
        else:
            self.iter += 1
            out_dict = {'source': return_dat_s, 'target': {'img': t, 'label': t_paths}}

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

        dim = x.shape[len(x.shape)-1]

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
class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 8, kernel_size=5, stride=1, padding=2)
        self.bn4 = nn.BatchNorm2d(8)
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048)

    def forward(self, x, reverse=False):
        x = x.to(torch.float)
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=3, padding=1)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=3, padding=1)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), stride=2, kernel_size=3, padding=1)
        x = F.relu(self.bn4(self.conv4(x)))

        # print(x.shape)

        x = x.view(x.size(0), 8192)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        return x

## ORIGINALLY FROM VisionLearningGroup.github.io
class Predictor(nn.Module):
    def __init__(self, prob=0.5):
        super(Predictor, self).__init__()
        # self.fc1 = nn.Linear(8192, 3072)
        # self.bn1_fc = nn.BatchNorm1d(3072)
        # self.fc2 = nn.Linear(3072, 2048)
        # self.bn2_fc = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 2)
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
    def __init__(self, args, interval=100, save_epoch=10):
        self.batch_size = args.batch_size
        self.target = args.target
        self.checkpoint_dir = args.checkpoint_dir
        self.save_epoch = save_epoch
        self.use_abs_diff = args.use_abs_diff
        self.source = args.source

        ## Load source and target data loaders
        print('Dataset loading')
        self.datasets, self.dataset_test = dataset_read(args=args)
        # print(self.dataset['S1'].shape)

        print('Data loading finished!')
        self.lr = args.lr
        self.mom = args.mom
        self.wd = args.weight_decay

        self.G = Feature()
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
        print('Model loaded')

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

    def train(self, epoch, record_file=None):
        criterion = nn.CrossEntropyLoss().cuda()
        self.G.train()
        self.C1.train()

        torch.cuda.manual_seed(1)

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
                loss_sum = loss_sum + source_data[my_key]["loss"]

            target_data["feat"] = self.G(target_data["img"])
            target_data["output"] = self.C1(target_data["feat"])

            loss_s = loss_sum / len(source_data)

            loss_s.backward() ## why only loss_s backward?? A: This is "train" not "train_MSDA"
                                    # so there's no accounting for discrepancy

            self.opt_c1.step()
            self.opt_g.step()
            self.reset_grad()
            if batch_idx > 500:
                return batch_idx

            if batch_idx % self.interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx, 100,
                    100. * batch_idx / 70000, loss_s.item()))
                if record_file:
                    record = open(record_file, 'a')
                    record.write('%s\n' % (loss_s.item()))
                    record.close()
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
                    img_s = torch.cat((img_s,data["source"][my_key]["img"]),0)
                    label_s = torch.cat((label_s,data["source"][my_key]["label"].long()),0)

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
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t '.format(
                    epoch, batch_idx, 100,
                    100. * batch_idx / 70000, loss_s.data[0]))
                if record_file:
                    record = open(record_file, 'a')
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
        for batch_idx, data in enumerate(self.dataset_test):
            # print(batch_idx)
            img = data['target']["img"]
            label = data['target']["label"]

            img, label = img.cuda(), label.long().cuda()
            img, label = Variable(img, volatile=True), Variable(label)
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
            pred1 = output1.data.max(1)[1]
            k = label.data.size()[0]
            correct1 += pred1.eq(label.data).cpu().sum()
            size += k
        np.savez('result_plot_sv_t', feature_all, label_all)
        test_loss = test_loss / size
        print(
            '\nTest set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.0f}%)  \n'.format(
                test_loss, correct1, size, 100. * correct1 / size))
        if save_model and epoch % self.save_epoch == 0:
            torch.save(self.G,
                       '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
            torch.save(self.C1,
                       '%s/%s_to_%s_model_epoch%s_C1.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
            torch.save(self.C2,
                       '%s/%s_to_%s_model_epoch%s_C2.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
        if record_file:
            record = open(record_file, 'a')
            print('recording %s', record_file)
            record.write('%s\n' % (float(correct1) / size))
            record.close()

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
            source_dict[my_key][output_key+"_loss"] = criterion(source_dict[my_key][output_key], source_dict[my_key]["label"])

        return source_dict

    def loss_all_domain(self, source_dict, target_dict):
        ## update dictionaries to include "feat" for each key
        source_dict, target_dict = self.feat_all_domain(source_dict=source_dict, target_dict=target_dict)
        ## update dictionaries to include "C1" for each key
        source_dict, target_dict = self.C1_all_domain(source_dict=source_dict, target_dict=target_dict)
        ## update dictionaries to include "C2" for each key
        source_dict, target_dict = self.C2_all_domain(source_dict=source_dict, target_dict=target_dict)
        ## compute MDSA loss
        loss_msda = 0.0005 * msda_regulizer(source_dict=source_dict, target_dict=target_dict, belta_moment=5)

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

        return loss, loss_dict

    def train_MSDA(self, epoch, record_file=None):
        self.G.train()
        self.C1.train()
        self.C2.train()
        torch.cuda.manual_seed(1)

        for batch_idx, data in enumerate(self.datasets):

            print("Training batch sampled")
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
            loss, _ = self.loss_all_domain(source_dict=source_data, target_dict=target_data)
            # print("Loss computed")

            loss.backward()

            ## take a gradient step after classifying in source domains
            self.opt_g.step()
            self.opt_c1.step()
            self.opt_c2.step()
            self.reset_grad()

            ## recompute loss in source domains
            loss_s, loss_s_dict = self.loss_all_domain(source_dict=source_data, target_dict=target_data)

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

            for i in range(4): ## taking 4 steps maximizing discrepancy between C1 and C2
                target_data["feat"] = self.G(target_data["img"])
                target_data["C1"] = self.C1(target_data["feat"])
                target_data["C2"] = self.C2(target_data["feat"])
                loss_dis = self.discrepancy(target_data["C1"], target_data["C2"])
                loss_dis.backward()
                self.opt_g.step()
                self.reset_grad()
            if batch_idx > 500:
                return batch_idx

        ### SORT THIS OUT TO RUN WITH VARIABLE NUMBER OF SOURCES
            if batch_idx % self.interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t  Discrepancy: {:.6f}'.format(
                    epoch, batch_idx, 100,
                    100. * batch_idx / 70000, loss_s_dict["C1"].item(), loss_s_dict["C2"].item(), loss_dis.item()))
                if record_file:
                    record = open(record_file, 'a')
                    record.write('%s %s %s\n' % (loss_dis.item(), loss_s_dict["C1"].item(), loss_s_dict["C2"].item()))
                    record.close()
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
            loss, _ = self.loss_all_domain(source_dict=source_data, target_dict=target_data)
            # print("Loss computed")

            loss.backward()

            ## take a gradient step after classifying in source domains
            self.opt_g.step()
            self.opt_c1.step()
            self.opt_c2.step()
            self.reset_grad()

            ## recompute loss in source domains
            loss_s, loss_s_dict = self.loss_all_domain(source_dict=source_data, target_dict=target_data)

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

            for i in range(4): ## taking 4 steps maximizing discrepancy between C1 and C2
                target_data["feat"] = self.G(target_data["img"])
                target_data["C1"] = self.C1(target_data["feat"])
                target_data["C2"] = self.C2(target_data["feat"])
                loss_dis = self.discrepancy(target_data["C1"], target_data["C2"])
                loss_dis.backward()
                self.opt_g.step()
                self.reset_grad()
            if batch_idx > 500:
                return batch_idx

        ### SORT THIS OUT TO RUN WITH VARIABLE NUMBER OF SOURCES
            if batch_idx % self.interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t  Discrepancy: {:.6f}'.format(
                    epoch, batch_idx, 100,
                    100. * batch_idx / 70000, loss_s_dict["C1"].item(), loss_s_dict["C2"].item(), loss_dis.item()))
                if record_file:
                    record = open(record_file, 'a')
                    record.write('%s %s %s\n' % (loss_dis.item(), loss_s_dict["C1"].item(), loss_s_dict["C2"].item()))
                    record.close()

        return batch_idx


###
    ###
    ###        MAIN FUNCTION
    ###

def main():
    # add args:

    parser = argparse.ArgumentParser()
    parser.add_argument('-source', default="SickKids_Stanford3_samples", help="US machine (data source) held out of training")
    parser.add_argument('-target', default="SIEMENS_S3000", help="US machine (data source) held out of training")
    parser.add_argument('-lab_col', default="kidney", help="Name of label column in datasheet")
    parser.add_argument('-lab_us_dir', default='/Users/lauren erdman/Desktop/kidney_img/View_Labeling/images/all_lab_img/',
                        help="Directory of ultrasound images")

    # parser.add_argument("-dichot", action='store_true', default=False, help="Use dichotomous (vs continuous) outcome")
    # parser.add_argument("-RL", action='store_true', default=False, help="Include r/l labels or only build model on 4 labels")

    parser.add_argument("-run_lab", default="MSE_Model-noOther", help="String to add to output files")

    parser.add_argument('-datasheet',
                        default='C:/Users/lauren erdman/Desktop/kidney_img/View_Labeling/datasheets/kidney_view_n100_sda_sksub_stsub_machine_spec_20210102.csv',
                        help="directory of DMSA images")  ## one datasheet: training and test specified a column of this datasheet
    # parser.add_argument('-datasheet', default='C:/Users/lauren erdman/Desktop/kidney_img/View_Labeling/datasheets/kidney_view_sda_sksub_stsub_machine_spec_3sampAcGE_20201203.csv',
    #                     help="directory of DMSA images") ## one datasheet: training and test specified a column of this datasheet

    ## try with: kidney_view_n100_sda_sksub_stsub_machine_spec_20210102.csv

    parser.add_argument('-csv_outdir', default='C:/Users/lauren erdman/Desktop/kidney_img/View_Labeling/OutputNov2020/csv/',
                        help="directory of DMSA images")
    parser.add_argument('-checkpoint_dir', default='C:/Users/lauren erdman/Desktop/kidney_img/View_Labeling/OutputNov2020/checkpoint/',
                        help="directory of DMSA images")

    # parser.add_argument('-out_csv', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/data/load_training_test_sets/test_out.csv',
    #                     help="directory of DMSA images")

    # parser.add_argument("-include_val", action='store_true', default=True,
    #                     help="Use dichotomous (vs continuous) outcome")
    # parser.add_argument("-include_test", action='store_true', default=True,
    #                     help="Use dichotomous (vs continuous) outcome")
    # parser.add_argument("-dmsa_out", action='store_true', default=False,
    #                     help="Save NN?")
    parser.add_argument('--save_epoch', type=int, default=6, metavar='N',
                        help='when to restore the model')
    parser.add_argument('--use_abs_diff', action='store_true', default=False,
                        help='use absolute difference value as a measurement')
    parser.add_argument('--eval_only', action='store_true', default=False,
                        help='evaluation only option')
    parser.add_argument('--no_msda', action='store_true', default=False,
                        help='evaluation only option')
    parser.add_argument('--max_epoch', type=int, default=8, metavar='N',
                        help='how many epochs')
    parser.add_argument('--one_step', action='store_true', default=False,
                        help='one step training with gradient reversal layer')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='save_model or not')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
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

    opt = parser.parse_args() ## comment for debug
    print(opt)

    ## from VisionLearningGroup.github.io M3SDA
    solver = Solver(args=opt)
    record_num = 0

    if opt.no_msda:
        record_train = '%s/NoMSDA_%s_%s.txt' % (
            opt.csv_outdir, opt.target, record_num)
        record_test = '%s/NoMSDA_%s_%s_test.txt' % (
            opt.csv_outdir, opt.target, record_num)
    else:
        record_train = '%s/%s_%s.txt' % (
            opt.csv_outdir, opt.target, record_num)
        record_test = '%s/%s_%s_test.txt' % (
            opt.csv_outdir, opt.target, record_num)

    while os.path.exists(record_train):
        if opt.no_msda:
            record_num += 1
            record_train = '%s/NoMSDA_%s_%s.txt' % (
                opt.csv_outdir, opt.target, record_num)
            record_test = '%s/NoMSDA_%s_%s_test.txt' % (
                opt.csv_outdir, opt.target, record_num)
        else:
            record_num += 1
            record_train = '%s/%s_%s.txt' % (
                opt.csv_outdir, opt.target, record_num)
            record_test = '%s/%s_%s_test.txt' % (
                opt.csv_outdir, opt.target, record_num)

    if not os.path.exists(opt.checkpoint_dir):
        os.mkdir(opt.checkpoint_dir)
    if not os.path.exists(opt.csv_outdir):
        os.mkdir(opt.csv_outdir)
    if opt.eval_only:
        solver.test(0)
    else:
        count = 0
        for t in range(opt.max_epoch):
            print('epoch: '+str(t))
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
