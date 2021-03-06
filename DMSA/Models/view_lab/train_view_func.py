import numpy as np
# import os
import torch
from torch import nn
import torch.optim as optim
# import torch.nn.functional as F
# from torchvision import transforms
# import importlib.machinery
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
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
    return torch.from_numpy(np.asarray(Image.open(file_name).convert('L')))


def flatten_list(in_list):
    flat_list = []
    for sublist in in_list:
        for item in sublist:
            flat_list.append(item)

    return flat_list


def make_img_dict(path, file_list, dim):

    dict_out = dict()
    for img_file in file_list:
        dict_out[img_file] = read_image_file(path + "/" + img_file).view(1, dim, dim)

    return dict_out


def make_seq_dict(datasheet, opt):

    lab_out = dict()
    seq_out = dict()
    datasheet.set_index('SEQ_ID')
    my_seqs = list(set(list(datasheet['SEQ_ID'])))

    for seq in my_seqs:
        seq_list = list(datasheet.loc[datasheet['SEQ_ID'] == seq, 'US_FILE'])
        seq_out[seq] = make_img_dict(path=opt.us_dir, file_list=seq_list, dim=opt.dim)

        if opt.dichot:
            lab_out[seq] = list(set(list(datasheet.loc[datasheet['SEQ_ID'] == seq, 'FUNC_DICH'])))[0]
        else:
            lab_out[seq] = list(set(list(datasheet.loc[datasheet['SEQ_ID'] == seq, 'FUNC_CONT'])))[0]

    return seq_out, lab_out

###

    ###
    ###        DATASET
    ###


class USFuncDataset(Dataset):
    """ Data loader for DMSA data """

    def __init__(self, data_sheet, args):
        """
        data_sheet: csv spreadsheet with 1 column US files ("US_FILE")
            1 column for the sequence ID ("SEQ_ID")
            2 label columns: 1 dichotomous and 1 continuous ("FUNC_DICH" and "FUNC_CONT"), and

        args: argument dictionary
            args.dichot: if set to True, a binary 0/1 value will be used as the label
            else if set to False, a continuous value for left function will be used as the label
        """

        self.dim = args.dim

        self.id = list(set(list(data_sheet['SEQ_ID'])))

        self.us_seq_dict, self.us_lab_dict = make_seq_dict(datasheet=data_sheet, opt=args)

    def __len__(self):
        return len(self.id)

    def __getitem__(self, index):
            ## Maybe add file path as argument to init? Maybe "opt"?
        seq_id = self.id[index]
        for i, file_name in enumerate(self.us_seq_dict[seq_id].keys()):
            if i == 0:
                seq_imgs = self.us_seq_dict[seq_id][file_name].unsqueeze(0)
            else:
                seq_imgs = torch.cat((seq_imgs, self.us_seq_dict[seq_id][file_name].unsqueeze(0)), 0)

        bs = len(self.us_seq_dict[seq_id].keys())

        label = self.us_lab_dict[seq_id]

        return seq_imgs.view([bs, 1, self.dim, self.dim]), label


class KidLabDataset(Dataset):
    """ Data loader for DMSA data """

    def __init__(self, data_sheet, args):
        """
        data_sheet: csv spreadsheet with 1 column for each US view image file ("SR","SL","TR","TL","B"),
            1 column for each DMSA view image file (A/P) ("DMSA_A" and "DMSA_P"),
            2 label columns: 1 dichotomous and 1 continuous ("FUNC_DICH" and "FUNC_CONT"), and
            1 column for the instance id "ID"

        args: argument dictionary
            args.dichot: if set to True, a binary 0/1 value will be used as the label
            else if set to False, a continuous value for left function will be used as the label
        """

        self.dim = args.dim

        self.id = data_sheet['img_id']

        self.img_files = data_sheet['IMG_FILE']

        self.us_img_dict = make_img_dict(path=args.lab_us_dir, file_list=self.img_files, dim=args.dim)

        if args.RL:
            self.label = data_sheet['NUM_LAB6']
        else:
            self.label = data_sheet['NUM_LAB4']

    def __len__(self):
        return len(self.id)

    def __getitem__(self, index):
        input_tensor = self.us_img_dict[self.img_files[index]]

        out_label = self.label[index]
        # print("out_label")
        # print(out_label)

        return input_tensor.view([self.dim, self.dim]), out_label


###
    ###
    ###        MODELS
    ###

##  VIEW LABEL MODELS
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


##  VIEW LABEL + FUNCTION MODEL
class LabFuncMod(nn.Module):
    def __init__(self, args):
        super(LabFuncMod, self).__init__()

        self.kid_labs = KidneyLab(args)
        self.get_kid_labs = args.get_kid_labels
        self.RL = args.RL

        conv0 = nn.Sequential(nn.Conv2d(1, 128, 7, padding=2),
                              nn.BatchNorm2d(128),
                              nn.MaxPool2d(2),
                              nn.ReLU())
        conv1 = nn.Sequential(nn.Conv2d(128, 64, 7, padding=2),
                              nn.BatchNorm2d(64),
                              nn.MaxPool2d(2),
                              nn.ReLU())
        conv2 = nn.Sequential(nn.Conv2d(64, 32, 7, padding=2),
                              nn.BatchNorm2d(32),
                              nn.MaxPool2d(2),
                              nn.ReLU())
        conv3 = nn.Sequential(nn.Conv2d(32, 16, 7, padding=2),
                              nn.BatchNorm2d(16),
                              nn.MaxPool2d(2),
                              nn.ReLU())

        self.in_conv = nn.Sequential(conv0, conv1, conv2, conv3)

        if args.RL:
            linear1 = nn.Sequential(nn.Linear(3136*5, 512, bias=True),
                                    nn.ReLU(),
                                    nn.Dropout(0.5))
        else:
            linear1 = nn.Sequential(nn.Linear(3136*3, 512, bias=True),
                                    nn.ReLU(),
                                    nn.Dropout(0.5))

        linear2 = nn.Sequential(nn.Linear(512, 64, bias=True),
                                nn.ReLU(),
                                nn.Dropout(0.5))

        # linear3 = nn.Sequential(nn.Linear(64, 1, bias=True),
        #                         nn.Sigmoid())

        if args.dichot:
            linear3 = nn.Sequential(nn.Linear(64, 2, bias=True))
        else:
            linear3 = nn.Sequential(nn.Linear(64, 1, bias=True),
                                    nn.Sigmoid())

        self.out_fc = nn.Sequential(linear1, linear2, linear3)

    def forward(self, x, lab_out=False):

        if len(x.squeeze().shape) > 2:
            bs = x.squeeze().shape[0]
        else:
            bs = 1

        dim = x.shape[len(x.shape)-1]

        my_kid_labs = self.kid_labs(x.view([bs, 1, dim, dim]))

        if lab_out:
            return my_kid_labs
        else:
            my_kid_convs = self.in_conv(x)
            my_kid_convs_flat = my_kid_convs.view([-1, bs])
            # my_kid_convs_transp = torch.transpose(my_kid_convs_flat, 0, 1)

            softmax = nn.Softmax(1)

            if self.RL:
                # print("kid labs shape: ")
                # print(my_kid_labs.shape)
                kid_labs_in = my_kid_labs.view([bs, 5])#[:, 0:5]
            else:
                kid_labs_in = my_kid_labs.view([bs, 3])#[:, 0:3]

            kid_labs_wts = softmax(kid_labs_in)
            # print("kid_lab_wts shape: ")
            # print(kid_labs_wts)
            #
            # print("my_kid_convs_transp shape: ")
            # print(my_kid_convs_transp.shape)

            if self.RL:
                # weight_embed = torch.matmul(my_kid_convs_flat, kid_labs_wts).view([1, 5, -1])
                weight_embed = torch.matmul(my_kid_convs_flat, kid_labs_wts).view([1, -1])

            else:
                # weight_embed = torch.matmul(my_kid_convs_flat, kid_labs_wts).view([1, 3, -1])
                weight_embed = torch.matmul(my_kid_convs_flat, kid_labs_wts).view([1, -1])

            func_out = self.out_fc(weight_embed)

            if self.get_kid_labs:
                return func_out, my_kid_labs
            else:
                return func_out

###
    ###
    ###        MODEL TRAINING FUNCTIONS
    ###


def initialize_training(args, neural_net):

    func_train_datasheet = pd.read_csv(args.func_train_datasheet)
    func_test_datasheet = pd.read_csv(args.func_test_datasheet)

    lab_train_datasheet = pd.read_csv(args.lab_train_datasheet)
    lab_test_datasheet = pd.read_csv(args.lab_test_datasheet)

    net = neural_net(args).to(args.device)

    func_train_dat = USFuncDataset(func_train_datasheet, args)
    func_test_dat = USFuncDataset(func_test_datasheet, args)

    lab_train_dat = KidLabDataset(lab_train_datasheet, args)
    lab_test_dat = KidLabDataset(lab_test_datasheet, args)

    func_train_loader = DataLoader(func_train_dat, batch_size=1, shuffle=True)
    func_test_loader = DataLoader(func_test_dat, batch_size=1, shuffle=False)

    lab_train_loader = DataLoader(lab_train_dat, batch_size=args.bs, shuffle=True)
    lab_test_loader = DataLoader(lab_test_dat, batch_size=args.bs, shuffle=False)

    if args.include_val:
        func_val_datasheet = pd.read_csv(args.func_val_datasheet)
        func_val_dat = USFuncDataset(func_val_datasheet, args)
        func_val_loader = DataLoader(func_val_dat, batch_size=1, shuffle=False)

        lab_val_datasheet = pd.read_csv(args.lab_val_datasheet)
        lab_val_dat = KidLabDataset(lab_val_datasheet, args)
        lab_val_loader = DataLoader(lab_val_dat, batch_size=args.bs, shuffle=False)

        return net, func_train_loader, func_val_loader, func_test_loader, lab_train_loader, lab_val_loader, lab_test_loader

    else:
        return net, func_train_loader, func_test_loader, lab_train_loader, lab_test_loader


def training_loop(args, network, file_lab):

    if args.include_val:
        net, func_train_dloader, func_val_dloader, func_test_dloader, lab_train_dloader, lab_val_dloader, lab_test_dloader = initialize_training(args, network)
    else:
        net, func_train_dloader, func_test_dloader, lab_train_dloader, lab_test_dloader = initialize_training(args, network)

    # if args.save_net:
    #     torch.save(network)

    lab_criterion = nn.CrossEntropyLoss()

    if args.dichot:
        criterion = nn.CrossEntropyLoss()
    else:
        # criterion = nn.BCELoss()
        criterion = nn.MSELoss()
    # criterion = nn.MSELoss()

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.mom)

    lab_train_mean_loss = []
    lab_val_mean_loss = []
    lab_test_mean_loss = []

    func_train_mean_loss = []
    func_val_mean_loss = []
    func_test_mean_loss = []

    softmax_func = nn.Softmax(1)

    for epoch in range(args.max_epochs):
        ## label output
        lab_train_epoch_loss = []
        lab_val_epoch_loss = []
        lab_test_epoch_loss = []

        lab_epoch_train_lab = []
        lab_epoch_train_pred = []

        lab_epoch_val_lab = []
        lab_epoch_val_pred = []

        lab_epoch_test_lab = []
        lab_epoch_test_pred = []

        ## function output
        func_train_epoch_loss = []
        func_val_epoch_loss = []
        func_test_epoch_loss = []

        func_epoch_train_lab = []
        func_epoch_train_pred = []

        func_epoch_val_lab = []
        func_epoch_val_pred = []

        func_epoch_test_lab = []
        func_epoch_test_pred = []

        if args.dichot:
            func_train_epoch_auc = []
            func_val_epoch_auc = []
            func_test_epoch_auc = []

        split = 0

        ##
        ## LABEL LOOP
        ##

        for idx, (lab_us, view_lab) in enumerate(lab_train_dloader):

            lab_out = net(lab_us.to(args.device).float(), lab_out=True) ## update this to have 2 outcomes function prediction + image prediction
            # print(lab_out.to(device=args.device).squeeze().float())
            # print(view_lab.to(args.device).squeeze().to(device=args.device).long())

            loss = lab_criterion(lab_out.to(device=args.device).squeeze().float(),
                                 torch.tensor(view_lab).to(args.device).squeeze().to(device=args.device).long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lab_train_epoch_loss.append(loss.item())
            split = split + 1

            lab_epoch_train_pred.append(softmax_func(lab_out).to("cpu").tolist())
            lab_epoch_train_lab.append(view_lab.to("cpu").tolist())

        lab_train_mean_loss.append(np.mean(np.array(lab_train_epoch_loss)))
        print('epoch: %d, label train loss: %.3f' %
              (epoch + 1, lab_train_mean_loss[epoch]))

        if args.include_val:
            for idx, (lab_us_val, view_lab_val) in enumerate(lab_val_dloader):

                lab_out_val = net(lab_us_val.to(args.device).float(), lab_out=True)

                loss_val = lab_criterion(lab_out_val.to(device=args.device).squeeze().float(),
                                         torch.tensor(view_lab_val).to(args.device).squeeze().to(device=args.device).long())

                lab_val_epoch_loss.append(loss_val.item())

                lab_epoch_val_pred.append(lab_out_val.to("cpu").tolist())
                lab_epoch_val_lab.append(view_lab_val.to("cpu").tolist())


            lab_val_mean_loss.append(np.mean(np.array(lab_val_epoch_loss)))
            print('epoch: %d, label val loss: %.3f' %
                  (epoch + 1, lab_val_mean_loss[epoch]))

            if args.include_test:
                for idx, (lab_us_test, view_lab_test) in enumerate(lab_test_dloader):

                    lab_out_test = net(lab_us_test.to(args.device).float(), lab_out=True)

                    loss_test = lab_criterion(lab_out_test.to(device=args.device).squeeze().float(),
                                     torch.tensor(view_lab_test).to(args.device).squeeze().to(device=args.device).long())

                    lab_test_epoch_loss.append(loss_test.item())

                    lab_epoch_test_lab.append(view_lab_test.to("cpu").tolist())
                    lab_epoch_test_pred.append(lab_out_test.to("cpu").tolist())

                lab_test_mean_loss.append(np.mean(np.array(lab_test_epoch_loss)))

                print('epoch: %d, label test loss: %.3f' %
                      (epoch + 1, lab_test_mean_loss[epoch]))

        else:
            if args.include_test:
                for idx, (lab_us_test, view_lab_test) in enumerate(lab_test_dloader):

                    lab_out_test = net(lab_us_test.to(args.device).float(), lab_out=True)

                    loss_test = lab_criterion(lab_out_test.to(device=args.device).squeeze().float(),
                                              torch.tensor(view_lab_test).to(args.device).squeeze().to(device=args.device).long())

                    lab_test_epoch_loss.append(loss_test.item())

                    lab_epoch_test_lab.append(view_lab_test.to("cpu").tolist())
                    lab_epoch_test_pred.append(lab_out_test.to("cpu").tolist())

                lab_test_mean_loss.append(np.mean(np.array(lab_test_epoch_loss)))

                print('epoch: %d, label test loss: %.3f' %
                      (epoch + 1, lab_test_mean_loss[epoch]))

        ##
        ## FUNCTION LOOP
        ##

        for idx, (func_us, func_lab) in enumerate(func_train_dloader):
            # print(func_us.to(args.device).float().squeeze().shape)

            if len(func_us.to(args.device).float().squeeze().shape) > 2:
                bs = func_us.to(args.device).float().squeeze().shape[0]
            else:
                bs = 1

            func_out = net(func_us.to(args.device).float().view([bs, 1, args.dim, args.dim]), lab_out=False)

            # loss = criterion(func_out.to(device=args.device).float(),
            #                  torch.tensor(func_lab).to(args.device).to(device=args.device).float())

            if args.dichot:
                # print("Function (dich): ")
                # print(func_out)
                # print("Label: ")
                # print(func_lab)

                loss = criterion(func_out.to(device=args.device).float().view([1, 2]),
                                 torch.tensor(func_lab).to(args.device).to(device=args.device).long())
            else:
                loss = criterion(func_out.to(device=args.device).float(),
                                 torch.tensor(func_lab).to(args.device).to(device=args.device).float())

            func_train_label = func_lab.to("cpu").item()
            func_epoch_train_lab.append(func_train_label)

            if args.dichot:
                pred_probs = np.max(np.array(func_out.to("cpu").tolist()))
                func_epoch_train_pred.append(pred_probs.item())
            else:
                func_epoch_train_pred.append(func_out.to("cpu").item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            func_train_epoch_loss.append(loss.item())
            split = split + 1
            # print('epoch: %d, split: %d, train loss: %.3f' %
            #       (epoch + 1, split, loss.item()))

        # print(np.array(func_epoch_train_lab))
        # print(np.array(func_epoch_train_pred))

        if args.dichot:
            train_auc = roc_auc_score(np.array(func_epoch_train_lab), np.array(func_epoch_train_pred))
            print("Train AUC : " + str(train_auc))
            func_train_epoch_auc.append(train_auc)

        func_train_mean_loss.append(np.mean(np.array(func_train_epoch_loss)))
        print('epoch: %d, function train loss: %.3f' %
              (epoch + 1, func_train_mean_loss[epoch]))

        if args.include_val:
            for idx, (func_us_val, func_lab_val) in enumerate(func_val_dloader):

                if len(func_us_val.to(args.device).float().squeeze().shape) > 2:
                    bs = func_us_val.to(args.device).float().squeeze().shape[0]
                else:
                    bs = 1

                func_out_val = net(func_us_val.to(args.device).float().view([bs, 1, args.dim, args.dim]), lab_out=False)

                # loss_val = criterion(func_out_val.to(device=args.device).float(),
                #                      torch.tensor(func_lab_val).to(args.device).to(device=args.device).float())

                func_val_label = func_lab_val.to("cpu").item()
                func_epoch_val_lab.append(func_val_label)

                if args.dichot:
                    pred_probs_val = np.max(np.array(func_out_val.to("cpu").tolist()), axis=1)
                    func_epoch_val_pred.append(pred_probs_val.item())
                else:
                    func_epoch_val_pred.append(func_out_val.to("cpu").item())

                if args.dichot:
                    loss_val = criterion(func_out_val.to(device=args.device).float().view([1, 2]),
                                         torch.tensor(func_lab_val).to(args.device).to(device=args.device).long())
                else:
                    loss_val = criterion(func_out_val.to(device=args.device).float(),
                                         torch.tensor(func_lab_val).to(args.device).to(device=args.device).float())

                func_val_epoch_loss.append(loss_val.item())

            if args.dichot:
                val_auc = roc_auc_score(np.array(func_epoch_val_lab), np.array(func_epoch_val_pred))
                print("Val AUC : " + str(val_auc))
                func_val_epoch_auc.append(val_auc)

            func_val_mean_loss.append(np.mean(np.array(func_val_epoch_loss)))
            print('epoch: %d, function val loss: %.3f' %
                  (epoch + 1, func_val_mean_loss[epoch]))

            # func_epoch_val_pred.append(func_out_val.to("cpu").tolist())
            # func_epoch_val_lab.append(func_lab_val.to("cpu").tolist())

            if args.include_test:
                for idx, (func_us_test, func_lab_test) in enumerate(func_test_dloader):

                    if len(func_us_test.to(args.device).float().squeeze().shape) > 2:
                        bs = func_us_test.to(args.device).float().squeeze().shape[0]
                    else:
                        bs = 1

                    func_out_test = net(func_us_test.to(args.device).float().view([bs, 1, args.dim, args.dim]), lab_out=False)

                    # loss_test = criterion(func_out_test.to(device=args.device).float(),
                    #                       torch.tensor(func_lab_test).to(args.device).to(device=args.device).float())
                    if args.dichot:
                        loss_test = criterion(func_out_test.to(device=args.device).float().view([1, 2]),
                                              torch.tensor(func_lab_test).to(args.device).to(device=args.device).long())
                    else:
                        loss_test = criterion(func_out_test.to(device=args.device).float(),
                                              torch.tensor(func_lab_test).to(args.device).to(device=args.device).float())

                    func_test_epoch_loss.append(loss_test.item())

                    func_test_label = func_lab_test.to("cpu").item()
                    func_epoch_test_lab.append(func_test_label)

                    if args.dichot:
                        pred_probs_test = np.max(np.array(func_out_test.to("cpu").tolist()), axis=1)
                        func_epoch_test_pred.append(pred_probs_test.item())
                    else:
                        func_epoch_test_pred.append(func_out_test.to("cpu").item())

                    # func_val_labels = flatten_list(func_lab_val.to("cpu").tolist())
                    # func_epoch_val_lab.append(func_val_labels)

                    func_test_label = func_lab_test.to("cpu").item()
                    func_epoch_test_lab.append(func_test_label)

                    if args.dichot:
                        pred_probs_test = np.max(np.array(func_out_test.to("cpu").tolist()), axis=1)
                        func_epoch_test_pred.append(pred_probs_test.item())
                    else:
                        func_epoch_test_pred.append(func_out_test.to("cpu").item())

                    # func_epoch_test_lab.append(func_lab_test.to("cpu").tolist())
                    # func_epoch_test_pred.append(func_out_test.to("cpu").tolist())

                if args.dichot:
                    test_auc = roc_auc_score(np.array(func_epoch_test_lab), np.array(func_epoch_test_pred))
                    print("Test AUC : " + str(test_auc))
                    func_test_epoch_auc.append(test_auc)

                func_test_mean_loss.append(np.mean(np.array(func_test_epoch_loss)))

                print('epoch: %d, function test loss: %.3f' %
                      (epoch + 1, func_test_mean_loss[epoch]))

        else:
            if args.include_test:
                for idx, (func_us_test, func_lab_test) in enumerate(func_test_dloader):

                    if len(func_us_test.to(args.device).float().squeeze().shape) > 2:
                        bs = func_us_test.to(args.device).float().squeeze().shape[0]
                    else:
                        bs = 1
                    func_out_test = net(func_us_test.to(args.device).float().view([bs, 1, args.dim, args.dim]), lab_out=False)

                    # loss_test = criterion(func_out_test.to(device=args.device).float(),
                    #                       torch.tensor(func_lab_test).to(args.device).to(device=args.device).float())
                    if args.dichot:
                        loss_test = criterion(func_out_test.to(device=args.device).float().view([1, 2]),
                                              torch.tensor(func_lab_test).to(args.device).to(device=args.device).long())
                    else:
                        loss_test = criterion(func_out_test.to(device=args.device).float(),
                                              torch.tensor(func_lab_test).to(args.device).to(device=args.device).float())

                    func_test_epoch_loss.append(loss_test.item())

                    func_test_label = func_lab_test.to("cpu").item()
                    func_epoch_test_lab.append(func_test_label)

                    if args.dichot:
                        pred_probs_test = np.max(np.array(func_out_test.to("cpu").tolist()), axis=1)
                        func_epoch_test_pred.append(pred_probs_test.item())
                    else:
                        func_epoch_test_pred.append(func_out_test.to("cpu").item())

                    # func_epoch_test_lab.append(func_lab_test.to("cpu").tolist())
                    # func_epoch_test_pred.append(func_out_test.to("cpu").tolist())

                if args.dichot:
                    test_auc = roc_auc_score(np.array(func_epoch_test_lab), np.array(func_epoch_test_pred))
                    print("Test AUC : " + str(test_auc))
                    func_test_epoch_auc.append(test_auc)

                func_test_mean_loss.append(np.mean(np.array(func_test_epoch_loss)))

                print('epoch: %d, function test loss: %.3f' %
                      (epoch + 1, func_test_mean_loss[epoch]))

    if args.save_pred:
        func_train_df = pd.DataFrame({"pred": flatten_list(func_epoch_train_pred),
                                      "lab": flatten_list(func_epoch_train_lab)})
        func_train_file = args.csv_outdir + "/TrainPred_func_" + file_lab + ".csv"
        func_train_df.to_csv(func_train_file)

        lab_train_df = pd.DataFrame({"pred": flatten_list(lab_epoch_train_pred),
                                      "lab": flatten_list(lab_epoch_train_lab)})
        lab_train_file = args.csv_outdir + "/TrainPred_lab_" + file_lab + ".csv"
        lab_train_df.to_csv(lab_train_file)

        if args.include_val:
            func_val_df = pd.DataFrame({"pred": flatten_list(func_epoch_val_pred),
                                        "lab": flatten_list(func_epoch_val_lab)})
            func_val_file = args.csv_outdir + "/ValPred_func_" + file_lab + ".csv"
            func_val_df.to_csv(func_val_file)

            lab_val_df = pd.DataFrame({"pred": flatten_list(lab_epoch_val_pred),
                                        "lab": flatten_list(lab_epoch_val_lab)})
            lab_val_file = args.csv_outdir + "/ValPred_lab_" + file_lab + ".csv"
            lab_val_df.to_csv(lab_val_file)

            if args.include_test:
                func_test_df = pd.DataFrame({"pred": flatten_list(func_epoch_test_pred),
                                             "lab": flatten_list(func_epoch_test_lab)})
                func_test_file = args.csv_outdir + "/TestPred_func_" + file_lab + ".csv"
                func_test_df.to_csv(func_test_file)

                lab_test_df = pd.DataFrame({"pred": flatten_list(lab_epoch_test_pred),
                                             "lab": flatten_list(lab_epoch_test_lab)})
                lab_test_file = args.csv_outdir + "/TestPred_lab_" + file_lab + ".csv"
                lab_test_df.to_csv(lab_test_file)

        else:
            if args.include_test:
                func_test_df = pd.DataFrame({"pred": flatten_list(func_epoch_test_pred),
                                             "lab": flatten_list(func_epoch_test_lab)})
                func_test_file = args.csv_outdir + "/TestPred_func_" + file_lab + ".csv"
                func_test_df.to_csv(func_test_file)

                lab_test_df = pd.DataFrame({"pred": flatten_list(lab_epoch_test_pred),
                                             "lab": flatten_list(lab_epoch_test_lab)})
                lab_test_file = args.csv_outdir + "/TestPred_lab_" + file_lab + ".csv"
                lab_test_df.to_csv(lab_test_file)

    if args.save_net:
        net_file = args.csv_outdir + "/Net_" + file_lab + ".pth"
        # print(net_file)
        torch.save(net, net_file)
        # torch.save(net, args.net_file)

    if args.include_val:
        if args.include_test:
            return {"train_loss_func": func_train_mean_loss, "val_loss_func": func_val_mean_loss, "test_loss_func": func_test_mean_loss,
                    "train_loss_lab": lab_train_mean_loss, "val_loss_lab": lab_val_mean_loss, "test_loss_lab": lab_test_mean_loss}
    else:
        if args.include_test:
            return {"train_loss_func": func_train_mean_loss, "test_loss_func": func_test_mean_loss,
                    "train_loss_lab": lab_train_mean_loss, "test_loss_lab": lab_test_mean_loss}
        else:
            return {"train_loss_func": func_train_mean_loss, "train_loss_lab": lab_train_mean_loss}

###
    ###
    ###        MAIN FUNCTION
    ###

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-us_dir', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/all-jpgs-dmsa/',
                        help="Directory of ultrasound images")
    parser.add_argument('-lab_us_dir', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/label_img/',
                        help="Directory of ultrasound images")

    parser.add_argument("-dichot", action='store_true', default=False, help="Use dichotomous (vs continuous) outcome")
    parser.add_argument("-RL", action='store_true', default=False, help="Include r/l labels or only build model on 4 labels")

    parser.add_argument("-run_lab", default="MSE_Model-noOther", help="String to add to output files")

    parser.add_argument('-func_train_datasheet', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/data/load_training_test_sets/DMSA-train-datasheet-top3view-USfunc-noVlab.csv',
                        help="directory of DMSA images")
    parser.add_argument('-func_val_datasheet', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/data/load_training_test_sets/DMSA-val-datasheet-top3view-USfunc-noVlab.csv',
                        help="directory of DMSA images")
    parser.add_argument('-func_test_datasheet', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/data/load_training_test_sets/DMSA-test-datasheet-top3view-USfunc-noVlab.csv',
                        help="directory of DMSA images")

    parser.add_argument('-lab_train_datasheet', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/data/load_training_test_sets/train-view_label_df_20200423-noOther.csv',
                        help="directory of DMSA images")
    parser.add_argument('-lab_val_datasheet', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/data/load_training_test_sets/val-view_label_df_20200423-noOther.csv',
                        help="directory of DMSA images")
    parser.add_argument('-lab_test_datasheet', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/data/load_training_test_sets/test-view_label_df_20200423-noOther.csv',
                        help="directory of DMSA images")

    parser.add_argument('-csv_outdir', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/data/load_training_test_sets/',
                        help="directory of DMSA images")

    # parser.add_argument('-out_csv', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/data/load_training_test_sets/test_out.csv',
    #                     help="directory of DMSA images")

    parser.add_argument("-include_val", action='store_true', default=True,
                        help="Use dichotomous (vs continuous) outcome")
    parser.add_argument("-include_test", action='store_true', default=True,
                        help="Use dichotomous (vs continuous) outcome")
    parser.add_argument("-dmsa_out", action='store_true', default=False,
                        help="Save NN?")
    parser.add_argument("-save_net", action='store_true', default=False,
                        help="Save NN?")
    parser.add_argument("-save_pred", action='store_true', default=True,
                        help="Save NN?")
    parser.add_argument("-get_kid_labels", action='store_true', default=False,
                        help="Save NN?")

    # parser.add_argument("-net_file", default='my_net.pth',
    #                     help="File to save NN to")

    parser.add_argument("-dim", default=256, help="Image dimensions")
    parser.add_argument('-device', default='cuda',help="device to run NN on")

    parser.add_argument("-lr", default=0.01, help="Learning rate")
    parser.add_argument("-mom", default=0.9, help="Momentum")
    parser.add_argument("-bs", default=32, help="Batch size")
    parser.add_argument("-max_epochs", default=10, help="Maximum number of epochs")

    opt = parser.parse_args() ## comment for debug

    print(opt)

    opt.lr = float(opt.lr)
    opt.mom = float(opt.mom)
    opt.max_epochs = int(opt.max_epochs)

    my_net = LabFuncMod

    analysis_time = "_".join(str(datetime.datetime.now()).split(" "))
    file_labs = opt.run_lab + "_AUC_LR" + str(opt.lr) + "_MOM" + str(opt.mom) + "_MAXEP" + str(opt.max_epochs) + "_BS" + str(opt.bs) + "_" + analysis_time

    loss_dict = training_loop(opt, my_net, file_labs)
    #
    loss_df = pd.DataFrame(loss_dict)

    out_csvfile = opt.csv_outdir + "/Loss_" + file_labs + ".csv"
    loss_df.to_csv(out_csvfile)


if __name__ == "__main__":
    main()

###
###  DEBUGGING
###
## args for debugging :)
# class make_opt():
#     def __init__(self):
#         self.us_dir = '/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/all-jpgs-dmsa/'
#         self.dmsa_dir = '/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/all-dmsa-cabs/dmsa-jpgs/'
#         self.dichot = False # for a 0/1 outcome, set to True
#         # self.datasheet = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Image organization Nov 2019/DMSA-datasheet-top3view.csv'
#         self.datasheet = '/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/data/load_training_test_sets/DMSA-datasheet-top3view.csv'
#             ## updataing to train/val/test
#         self.dim = 256
#
# opt = make_opt()
# my_data = DMSADataset(args=opt)
# my_dataloader = DataLoader(my_data, shuffle=True)
