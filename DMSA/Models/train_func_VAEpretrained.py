from argparse import ArgumentParser
from importlib import import_module
from math import ceil
from os import replace
from os.path import exists, join
from shutil import copy
from sys import stderr

import sys
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

from train_utils import extend_batch, get_validation_iwae
from VAE import VAE
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
import random
from PIL import Image
from dmsa_model_v0 import *
import os

# import wandb


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

def make_img_dict(path,file_list,dim):

    dict_out = dict()
    for img_file in file_list:
        dict_out[img_file] = read_image_file(path + "/" + img_file).view(1, dim, dim)

    return dict_out

###

    ###
    ###        DATASET
    ###


class DMSADataset(Dataset):
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
        self.dmsa_out = args.dmsa_out

        self.us_dir = args.us_dir
        self.dmsa_dir = args.dmsa_dir

        self.id = data_sheet['ID']

        self.sr_file = data_sheet['SR']
        self.sl_file = data_sheet['SL']
        self.tr_file = data_sheet['TR']
        self.tl_file = data_sheet['TL']
        self.b_file = data_sheet['B']

        if self.dmsa_out:
            self.dmsa_a_file = data_sheet['DMSA_A']
            self.dmsa_p_file = data_sheet['DMSA_P']

        if args.dichot:
            self.label = data_sheet['FUNC_DICH']#.to_numpy()
        else:
            self.label = data_sheet['FUNC_CONT']#.to_numpy()

    def __len__(self):
        return len(self.id)

    def __getitem__(self, index):
            ## Maybe add file path as argument to init? Maybe "opt"?
        sr = read_image_file(self.us_dir + self.sr_file[index]).view(1, self.dim, self.dim)
        sl = read_image_file(self.us_dir + self.sl_file[index]).view(1, self.dim, self.dim)
        tr = read_image_file(self.us_dir + self.tr_file[index]).view(1, self.dim, self.dim)
        tl = read_image_file(self.us_dir + self.tl_file[index]).view(1, self.dim, self.dim)
        b = read_image_file(self.us_dir + self.b_file[index]).view(1, self.dim, self.dim)
            ## make 5x256x256 tensor
        input_tensor = torch.cat((sr, sl, tr, tl, b), 0).float()

        out_label = torch.tensor(self.label)[index]
        # print("out_label")
        # print(out_label)

        if self.dmsa_out:
            dmsa_a = read_image_file(self.dmsa_dir + self.dmsa_a_file[index]).view(1, self.dim, self.dim)
            dmsa_p = read_image_file(self.dmsa_dir + self.dmsa_p_file[index]).view(1, self.dim, self.dim)
            ## make 2x256x256 tensor
            output_tensor = torch.cat((dmsa_a, dmsa_p),0).float()
            return input_tensor, output_tensor, out_label

        else:
            return input_tensor, out_label


class DMSADataset_PreloadImgs(Dataset):
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
        self.dmsa_out = args.dmsa_out

        # self.us_dir = args.us_dir
        # self.dmsa_dir = args.dmsa_dir

        self.id = data_sheet['ID']

        self.sr_file = data_sheet['SR']
        self.sl_file = data_sheet['SL']
        self.tr_file = data_sheet['TR']
        self.tl_file = data_sheet['TL']
        self.b_file = data_sheet['B']

        all_files = flatten_list([data_sheet['SR'].tolist(), data_sheet['SL'].tolist(),
                                  data_sheet['TR'].tolist(), data_sheet['TL'].tolist(),
                                  data_sheet['B'].tolist()])

        uniq_files = list(set(all_files))

        self.us_img_dict = make_img_dict(path=args.us_dir, file_list=uniq_files, dim=args.dim)

        if self.dmsa_out:
            self.dmsa_a_file = data_sheet['DMSA_A']
            self.dmsa_p_file = data_sheet['DMSA_P']

            self.dmsa_img_dict_a = make_img_dict(path=args.dmsa_dir, file_list=self.dmsa_a_file, dim=args.dim)
            self.dmsa_img_dict_p = make_img_dict(path=args.dmsa_dir, file_list=self.dmsa_p_file, dim=args.dim)

        if args.dichot:
            self.label = data_sheet['FUNC_DICH']#.to_numpy()
        else:
            self.label = data_sheet['FUNC_CONT']#.to_numpy()

    def __len__(self):
        return len(self.id)

    def __getitem__(self, index):
            ## Maybe add file path as argument to init? Maybe "opt"?
        sr = self.us_img_dict[self.sr_file[index]]
        sl = self.us_img_dict[self.sl_file[index]]
        tr = self.us_img_dict[self.tr_file[index]]
        tl = self.us_img_dict[self.tl_file[index]]
        b = self.us_img_dict[self.b_file[index]]
            ## make 5x256x256 tensor
        input_tensor = torch.cat((sr, sl, tr, tl, b), 0)

        out_label = torch.tensor(self.label)[index]
        # print("out_label")
        # print(out_label)

        if self.dmsa_out:
            dmsa_a = self.dmsa_img_dict_a[self.dmsa_a_file[index]]
            dmsa_p = self.dmsa_img_dict_a[self.dmsa_p_file[index]]
            ## make 2x256x256 tensor
            output_tensor = torch.cat((dmsa_a, dmsa_p), 0)
            return input_tensor, output_tensor, out_label

        else:
            return input_tensor, out_label

    ###
    ###        MODEL
    ###


class FCNet(nn.Module):
    def __init__(self, args):
        super(FCNet, self).__init__()

        self.in_fc = nn.Sequential(nn.Linear(512, 256),
                                   nn.BatchNorm1d(256),
                                   nn.ReLU(),
                                   nn.Dropout(0.5))
        self.mid_fc = nn.Sequential(nn.Linear(256, 64),
                                    nn.BatchNorm1d(256),
                                    nn.ReLU(),
                                    nn.Dropout(0.5))

        if args.dichot:
            self.out_fc = nn.Sequential(nn.Linear(64, 2),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU())
        else:
            self.out_fc = nn.Sequential(nn.Linear(64, 1),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU())

    def forward(self, x):

        x1 = self.in_fc(x.float())
        x2 = self.mid_fc(x1)
        x3 = self.out_fc(x2)

        return x3


###
    ###
    ###        MODEL TRAINING FUNCTIONS
    ###

def initialize_training(args, neural_net):

    train_datasheet = pd.read_csv(args.train_datasheet)
    test_datasheet = pd.read_csv(args.test_datasheet)

    net = neural_net(args).to(args.device)

    train_dat = DMSADataset_PreloadImgs(train_datasheet, args)
    test_dat = DMSADataset_PreloadImgs(test_datasheet, args)

    train_loader = DataLoader(train_dat, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(test_dat, batch_size=args.bs, shuffle=False)

    # os.chdir(args.github_dir + '/nephronetwork/DMSA/vae_model/')
    # import model

    # model_module = import_module(args.github_dir + '/nephronetwork/DMSA/Models/vae_model/model')

    # os.chdir(args.github_dir)
    sys.path.append(args.github_dir + '/nephronetwork/DMSA/Models/')
    import VAE
    # from prob_utils import normal_parse_params
    sys.path.append(args.github_dir + '/nephronetwork/DMSA/Models/vae_model/')
    import model

    vae_model = VAE(model.reconstruction_log_prob,
                    model.prior_network,
                    model.generative_network)

    # load the last checkpoint, if it exists
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    vae_model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # validation_iwae = checkpoint['validation_iwae']
    # train_vlb = checkpoint['train_vlb']

    if args.include_val:
        val_datasheet = pd.read_csv(args.val_datasheet)
        val_dat = DMSADataset_PreloadImgs(val_datasheet, args)
        val_loader = DataLoader(val_dat, batch_size=args.bs, shuffle=False)

        return net, vae_model, train_loader, val_loader, test_loader

    else:
        return net, vae_model, train_loader, test_loader


def training_loop(args, network, file_lab):

    if args.include_val:
        net, vae_mod, train_dloader, val_dloader, test_dloader = initialize_training(args, network)
    else:
        net, vae_mod, train_dloader, test_dloader = initialize_training(args, network)

    # if args.save_net:
    #     torch.save(network)

    if args.dichot:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCELoss()
    #     criterion = nn.MSELoss()

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.mom)

    train_mean_loss = []
    val_mean_loss = []
    test_mean_loss = []

    for epoch in range(args.max_epochs):
        train_epoch_loss = []
        val_epoch_loss = []
        test_epoch_loss = []

        epoch_train_lab = []
        epoch_train_pred = []

        epoch_val_lab = []
        epoch_val_pred = []

        epoch_test_lab = []
        epoch_test_pred = []

        split = 0

        for idx, (us, lab) in enumerate(train_dloader):

            # print("input dim")
            # print(us.shape)
            #
            # print("dmsa dim")
            # print(dmsa.shape)

            if args.dmsa_out:
                out = net(us.to(args.device)) ## update this to have 2 outcomes function prediction + image prediction

            else:
                vae_out = vae_mod.make_latent_distributions(us.to(args.device))
                out = net(vae_out)

            print(out.shape)
            print(out.shape)
            # print(lab.shape)
            # print(lab.squeeze())
            # print(out.squeeze())

            bs = us.shape[0]

            if args.dichot:
                loss = criterion(out.view([bs, 2]).to(device=args.device).float(), lab.to(args.device).squeeze().to(device=args.device).long())

            else:
                loss = criterion(out.view([bs, 1]).to(device=args.device).float(), lab.to(args.device).view([bs, 1]).to(device=args.device).float())
        #
        #     # print("predicted vals: ")
        #     # print(out.view([bs, 1]).to(device=args.device))
        #     #
        #     # print("true vals: ")
        #     # print(lab.view([bs, 1]).to(device=args.device))
        #
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #
        #     train_epoch_loss.append(loss.item())
        #     split = split + 1
        #     # print('epoch: %d, split: %d, train loss: %.3f' %
        #     #       (epoch + 1, split, loss.item()))
        #
        #     epoch_train_lab.append(lab.to("cpu").squeeze().float().tolist())
        #     epoch_train_pred.append(out.to("cpu").squeeze().float().tolist())
        #
        # train_mean_loss.append(np.mean(np.array(train_epoch_loss)))
        # print('epoch: %d, train loss: %.3f' %
        #       (epoch + 1, train_mean_loss[epoch]))

    #     if args.include_val:
    #         for idx, (us_val, lab_val) in enumerate(val_dloader):
    #
    #             bs = us_val.shape[0]
    #
    #             out_val = net(us_val.to(args.device))

    #             if args.dichot:
    #                 loss_val = criterion(out_val.view([bs, 2]).to(device=args.device).float(),
    #                                      lab_val.to(args.device).squeeze().to(device=args.device).long())
    #
    #             else:
    #                 loss_val = criterion(out_val.view([bs, 1]).to(device=args.device).float(),
    #                                      lab_val.to(args.device).view([bs, 1]).to(device=args.device).float())

    #
    #             val_epoch_loss.append(loss_val.item())
    #
    #         val_mean_loss.append(np.mean(np.array(val_epoch_loss)))
    #         print('epoch: %d, val loss: %.3f' %
    #               (epoch + 1, val_mean_loss[epoch]))
    #
    #         epoch_val_pred.append(out_val.to("cpu").squeeze().float().tolist())
    #         epoch_val_lab.append(lab_val.to("cpu").squeeze().float().tolist())
    #
    #         if args.include_test:
    #             for idx, (us_test, lab_test) in enumerate(test_dloader):
    #
    #                 bs = us_test.shape[0]
    #
    #                 out_test = net(us_test.to(args.device))

                    # if args.dichot:
                    #     loss_test = criterion(out_test.view([bs, 2]).to(device=args.device).float(),
                    #                           lab_test.to(args.device).squeeze().to(device=args.device).long())
                    #
                    # else:
                    #     loss_test = criterion(out_test.view([bs, 1]).to(device=args.device).float(),
                    #                           lab_test.to(args.device).view([bs, 1]).to(device=args.device).float())
    #
    #                 test_epoch_loss.append(loss_test.item())
    #
    #             test_mean_loss.append(np.mean(np.array(test_epoch_loss)))
    #
    #             print('epoch: %d, test loss: %.3f' %
    #                   (epoch + 1, test_mean_loss[epoch]))
    #
    #             epoch_test_lab.append(lab_test.to("cpu").squeeze().float().tolist())
    #             epoch_test_pred.append(out_test.to("cpu").squeeze().float().tolist())
    #
    #     else:
    #         if args.include_test:
    #             for idx, (us_test, dmsa_test, lab_test) in enumerate(test_dloader):
    #                 out_test = net(us_test.to(args.device))

                    # if args.dichot:
                    #     loss_test = criterion(out_test.view([bs, 2]).to(device=args.device).float(),
                    #                           lab_test.to(args.device).squeeze().to(device=args.device).long())
                    #
                    # else:
                    #     loss_test = criterion(out_test.view([bs, 1]).to(device=args.device).float(),
                    #                           lab_test.to(args.device).view([bs, 1]).to(device=args.device).float())
    #
    #                 test_epoch_loss.append(loss_test.item())
    #
    #             test_mean_loss.append(np.mean(np.array(test_epoch_loss)))
    #
    #             print('epoch: %d, test loss: %.3f' %
    #                   (epoch + 1, test_mean_loss[epoch]))
    #
    #             epoch_test_lab.append(lab_test.to("cpu").squeeze().float().tolist())
    #             epoch_test_pred.append(out_test.to("cpu").squeeze().float().tolist())
    #
    # if args.save_pred:
    #     train_df = pd.DataFrame({"pred": epoch_train_pred, "lab": epoch_train_lab})
    #     train_file = args.csv_outdir + "/TrainPred_" + file_lab + ".csv"
    #     train_df.to_csv(train_file)
    #
    #     if args.include_val:
    #         val_df = pd.DataFrame({"pred": epoch_val_pred, "lab": epoch_val_lab})
    #         val_file = args.csv_outdir + "/ValPred_" + file_lab + ".csv"
    #         val_df.to_csv(val_file)
    #
    #         if args.include_test:
    #             test_df = pd.DataFrame({"pred": epoch_test_pred, "lab": epoch_test_lab})
    #             test_file = args.csv_outdir + "/TestPred_" + file_lab + ".csv"
    #             test_df.to_csv(test_file)
    #
    #     else:
    #         if args.include_test:
    #             test_df = pd.DataFrame({"pred": epoch_test_pred, "lab": epoch_test_lab})
    #             test_file = args.csv_outdir + "/TestPred_" + file_lab + ".csv"
    #             test_df.to_csv(test_file)
    #
    # if args.save_net:
    #     net_file = args.csv_outdir + "/Net_" + file_lab + ".pth"
    #     print(net_file)
    #     torch.save(net, net_file)
    #     # torch.save(net, args.net_file)
    #
    # if args.include_val:
    #     if args.include_test:
    #         return {"train_loss": train_mean_loss, "val_loss": val_mean_loss, "test_loss": test_mean_loss}
    # else:
    #     if args.include_test:
    #         return {"train_loss": train_mean_loss, "test_loss": test_mean_loss}
    #     else:
    #         return {"train_loss": train_mean_loss}


###
    ###
    ###        MAIN FUNCTION
    ###


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mount_folder', default='',
                        help="Directory of ultrasound images")
    parser.add_argument('-github_dir', default='',
                        help="Directory of ultrasound images")

    parser.add_argument('-us_dir', default='/home/marta/nephronetwork-github/nephronetwork/1.Models/siamese_network/mountpoint/all-jpgs-dmsa/',
                        help="Directory of ultrasound images")
    parser.add_argument('-dmsa_dir', default='/home/marta/nephronetwork-github/nephronetwork/1.Models/siamese_network/mountpoint/all-dmsa-cabs/dmsa-jpgs/',
                        help="directory of DMSA images")

    parser.add_argument("-dichot", action='store_true', default=False, help="Use dichotomous (vs continuous) outcome")

    parser.add_argument("-run_lab", default="VAEfc_V1", help="String to add to output files")

    parser.add_argument('-train_datasheet', default='/home/marta/nephronetwork-github/nephronetwork/1.Models/siamese_network/mountpoint/data/load_training_test_sets/DMSA-train-datasheet-top2view.csv',
                        help="directory of DMSA images")
    parser.add_argument('-val_datasheet', default='/home/marta/nephronetwork-github/nephronetwork/1.Models/siamese_network/mountpoint/data/load_training_test_sets/DMSA-val-datasheet-top2view.csv',
                        help="directory of DMSA images")
    parser.add_argument('-test_datasheet', default='/home/marta/nephronetwork-github/nephronetwork/1.Models/siamese_network/mountpoint/data/load_training_test_sets/DMSA-test-datasheet-top2view.csv',
                        help="directory of DMSA images")

    parser.add_argument('-csv_outdir', default='/home/marta/lauren/',
                        help="directory of DMSA images")

    parser.add_argument('-checkpoint', default="/home/marta/nephronetwork-github/nephronetwork/1.Models/siamese_network/mountpoint/VAE/best_beta=1,lr=0.0002.tar",
                        help="directory of DMSA images")

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

    # parser.add_argument("-net_file", default='my_net.pth',
    #                     help="File to save NN to")

    parser.add_argument("-dim", default=256, help="Image dimensions")
    parser.add_argument('-device', default='cuda',help="device to run NN on")

    parser.add_argument("-lr", default=0.005, help="Image dimensions")
    parser.add_argument("-mom", default=0.9, help="Image dimensions")
    parser.add_argument("-bs", default=32, help="Image dimensions")
    parser.add_argument("-max_epochs", default=2, help="Image dimensions")

    opt = parser.parse_args() ## comment for debug

    my_net = FCNet

    analysis_time = "_".join(str(datetime.datetime.now()).split(" "))
    file_labs = opt.run_lab + "_AUC_LR" + str(opt.lr) + "_MOM" + str(opt.mom) + "_MAXEP" + str(opt.max_epochs) + "_BS" + str(opt.bs) + "_" + analysis_time

    ###### ********************
    ## ADD ARGS.MOUNT_FOLDER TO ALL FILE READ-INS
    loss_dict = training_loop(args=opt, network=my_net, file_lab=file_labs)

    #
    # loss_df = pd.DataFrame(loss_dict)
    #
    # out_csvfile = opt.csv_outdir + "/Loss_" + file_labs + ".csv"
    # loss_df.to_csv(out_csvfile)


if __name__ == "__main__":
    main()
