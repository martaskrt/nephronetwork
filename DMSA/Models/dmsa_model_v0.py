import numpy as np
# import os
import torch
from torch import nn
import torch.optim as optim
# import torch.nn.functional as F
# from torchvision import transforms
# import importlib.machinery
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import argparse
import datetime


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

def make_img_dict(path,file_list):

    dict_out = dict()
    for img_file in file_list:
        dict_out[img_file] = read_image_file(path + "/" + img_file).view(1, self.dim, self.dim)

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

        self.us_dir = args.us_dir
        self.dmsa_dir = args.dmsa_dir

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

        self.us_img_dict = make_img_dict(path=self.us_dir, file_list=uniq_files)

        if self.dmsa_out:
            self.dmsa_a_file = data_sheet['DMSA_A']
            self.dmsa_p_file = data_sheet['DMSA_P']

            self.dmsa_img_dict_a = make_img_dict(path=self.dmsa_dir, file_list=self.dmsa_a_file)
            self.dmsa_img_dict_p = make_img_dict(path=self.dmsa_dir, file_list=self.dmsa_p_file)

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
    ###
    ###        MODELS
    ###

class FuncMod(nn.Module):
    def __init__(self, args):
        super(FuncMod, self).__init__()

        # self.dichot = args.dichot

        self.conv0 = nn.Sequential(nn.Conv2d(5, 64, 5, padding=2),
                                   nn.MaxPool2d(2),
                                   nn.ReLU())
        self.conv1 = nn.Sequential(nn.Conv2d(64, 64, 5, padding=2),
                                   nn.MaxPool2d(2),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 32, 5, padding=2),
                                   nn.MaxPool2d(2),
                                   nn.ReLU())

        # self.linear1 = nn.Sequential(nn.Linear(64,32,bias=True),
        #                             nn.ReLU(),
        #                             nn.Dropout(0.5))
        # self.linear2 = nn.Sequential(nn.Linear(32,32,bias=True),
        #                             nn.ReLU(),
        #                             nn.Dropout(0.5))
        #
        # self.linear3 = nn.Sequential(nn.Linear(32, 2, bias=True),
        #                              nn.Sigmoid())

        self.linear1 = nn.Sequential(nn.Linear(2048, 512, bias=True),
                                    nn.ReLU(),
                                    nn.Dropout(0.5))
        self.linear2 = nn.Sequential(nn.Linear(512, 64, bias=True),
                                    nn.ReLU(),
                                    nn.Dropout(0.5))

        self.linear3 = nn.Sequential(nn.Linear(64, 1, bias=True),
                                     nn.Sigmoid())

        # if self.dichot:
        #     self.linear3 = nn.Sequential(nn.Linear(64,2,bias=True))
        # else:
        #     self.linear3 = nn.Sequential(nn.Linear(64,1,bias=True))

    def forward(self, x):

        bs = x.shape[0]

        x0 = self.conv0(x.float())
        x1 = self.conv1(x0)
        x2 = self.conv1(x1)
        x3 = self.conv1(x2)
        x4 = self.conv2(x3)

        x4_flat = x4.view([bs, 1, -1])
        # print("x4_flat.shape")
        # print(x4_flat.shape)

        x5 = self.linear1(x4_flat)
        x6 = self.linear2(x5)

        x7 = self.linear3(x6)

        return x7

class FuncModSiamese(nn.Module):
    def __init__(self, args):
        super(FuncModSiamese, self).__init__()

        # self.dichot = args.dichot

        self.conv0 = nn.Sequential(nn.Conv2d(1, 64, 5, padding=2),
                                   nn.MaxPool2d(2),
                                   nn.ReLU())
        self.conv1 = nn.Sequential(nn.Conv2d(64, 64, 5, padding=2),
                                   nn.MaxPool2d(2),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 32, 5, padding=2),
                                   nn.MaxPool2d(2),
                                   nn.ReLU())

        # self.linear1 = nn.Sequential(nn.Linear(64,32,bias=True),
        #                             nn.ReLU(),
        #                             nn.Dropout(0.5))
        # self.linear2 = nn.Sequential(nn.Linear(32,32,bias=True),
        #                             nn.ReLU(),
        #                             nn.Dropout(0.5))
        #
        # self.linear3 = nn.Sequential(nn.Linear(32, 2, bias=True),
        #                              nn.Sigmoid())

        self.linear0 = nn.Sequential(nn.Linear(2048*5,2048,bias=True),
                                    nn.ReLU(),
                                    nn.Dropout(0.5))
        self.linear1 = nn.Sequential(nn.Linear(2048,512,bias=True),
                                    nn.ReLU(),
                                    nn.Dropout(0.5))
        self.linear2 = nn.Sequential(nn.Linear(512,64,bias=True),
                                    nn.ReLU(),
                                    nn.Dropout(0.5))

        self.linear3 = nn.Sequential(nn.Linear(64, 1, bias=True),
                                     nn.Sigmoid())

        # if self.dichot:
        #     self.linear3 = nn.Sequential(nn.Linear(64,2,bias=True))
        # else:
        #     self.linear3 = nn.Sequential(nn.Linear(64,1,bias=True))

    def forward(self, x):
        bs = x.shape[0]

        for i in range(chan):

            x_in = x[:, i, :, :]

            x0 = self.conv0(x_in.float())
            x1 = self.conv1(x0)
            x2 = self.conv1(x1)
            x3 = self.conv1(x2)
            x4 = self.conv2(x3)

            if i == 0:
                x4_concat = x4.view([bs, 1, -1])
            else:
                x4_concat = torch.cat((x4_concat, x4.view([bs, 1, -1])),2)

        x5_0 = self.linear0(x4_concat)
        x5 = self.linear1(x5_0)
        x6 = self.linear2(x5)

        x7 = self.linear3(x6)

        return x7

## args for debugging :)
# class make_opt():
#     def __init__(self):
#         self.us_dir = '/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/all-jpgs-dmsa/'
#         self.dmsa_dir = '/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/all-dmsa-cabs/dmsa-jpgs/'
#         self.dichot = False # for a 0/1 outcome, set to True
#         # self.datasheet = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Image organization Nov 2019/DMSA-datasheet-top3view.csv'
#         self.datasheet = '/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/data/load_training_test_sets/DMSA-datasheet-top3view.csv'
#         self.dim = 256

## args for debugging :)
class make_opt():
    def __init__(self, split="train"):
        self.us_dir = '/data/alexchang/mnt/all-jpgs-dmsa/'
        self.dmsa_dir = '/data/alexchang/mnt/all-dmsa-cabs/dmsa-jpgs/'
        self.dichot = False # for a 0/1 outcome, set to True
        # self.datasheet = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Image organization Nov 2019/DMSA-datasheet-top3view.csv'
        if split == "train":
            
            self.datasheet = '/data/alexchang/mnt/data/load_training_test_sets/DMSA-train-datasheet-top2view.csv'

            # self.datasheet = '/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/data/load_training_test_sets/DMSA-datasheet-top3view.csv'
        elif split == "test":
            self.datasheet = '/data/alexchang/mnt/data/load_training_test_sets/DMSA-val-datasheet-top2view.csv'
        else:
            raise Exception("Invalid split.")

        self.dim = 256


###
    ###
    ###        MODEL TRAINING FUNCTIONS
    ###

def initialize_training(args,neural_net):

    train_datasheet = pd.read_csv(args.train_datasheet)
    test_datasheet = pd.read_csv(args.test_datasheet)

    net = neural_net(args).to(args.device)

    train_dat = DMSADataset_PreloadImgs(train_datasheet, args)
    test_dat = DMSADataset_PreloadImgs(test_datasheet, args)

    train_loader = DataLoader(train_dat, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(test_dat, batch_size=args.bs, shuffle=False)

    if args.include_val:
        val_datasheet = pd.read_csv(args.val_datasheet)
        val_dat = DMSADataset_PreloadImgs(val_datasheet, args)
        val_loader = DataLoader(val_dat, batch_size=args.bs, shuffle=False)

        return net, train_loader, val_loader, test_loader

    else:
        return net, train_loader, test_loader

# def cross_entropy(pred, soft_targets):
#     """
#
#     From: https://discuss.pytorch.org/t/how-should-i-implement-cross-entropy-loss-with-continuous-target-outputs/10720
#
#     :param pred: Predicted values
#     :param soft_targets: True values
#     :return: Cross-entropy loss
#     """
#
#     logsigmoid = nn.LogSigmoid()
#     return torch.mean(torch.sum(- soft_targets * logsigmoid(pred), 1))

def training_loop(args, network, time_str):

    if args.include_val:
        net, train_dloader, val_dloader, test_dloader = initialize_training(args, network)
    else:
        net, train_dloader, test_dloader = initialize_training(args, network)

    # if args.save_net:
    #     torch.save(network)

    # if args.dichot:
    criterion = nn.BCELoss()
    # else:
    #     criterion = nn.MSELoss()

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.mom)

    train_mean_loss = []
    val_mean_loss = []
    test_mean_loss = []

    for epoch in range(args.max_epochs):
        train_epoch_loss = []
        val_epoch_loss = []
        test_epoch_loss = []

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
                out = net(us.to(args.device))
            # print(out.shape)
            # print(out[:, :, 0].shape)
            # print(lab.shape)
            # print(lab.squeeze())
            # print(out.squeeze())

            bs = us.shape[0]

            loss = criterion(out.view([bs, 1]).to(device=args.device).float(), lab.to(args.device).view([bs, 1]).to(device=args.device).float())

            # print("predicted vals: ")
            # print(out.view([bs, 1]).to(device=args.device))
            #
            # print("true vals: ")
            # print(lab.view([bs, 1]).to(device=args.device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_epoch_loss.append(loss.item())
            split = split + 1
            # print('epoch: %d, split: %d, train loss: %.3f' %
            #       (epoch + 1, split, loss.item()))


        train_mean_loss.append(np.mean(np.array(train_epoch_loss)))
        print('epoch: %d, train loss: %.3f' %
              (epoch + 1, train_mean_loss[epoch]))

        if args.include_val:
            for idx, (us_val, lab_val) in enumerate(val_dloader):

                bs = us_val.shape[0]

                out_val = net(us_val.to(args.device))
                loss_val = criterion(out_val.view([bs, 1]).to(device=args.device).float(), lab_val.to(args.device).view([bs, 1]).to(device=args.device).float())

                val_epoch_loss.append(loss_val.item())

            val_mean_loss.append(np.mean(np.array(val_epoch_loss)))
            print('epoch: %d, val loss: %.3f' %
                (epoch + 1, val_mean_loss[epoch]))

            if args.include_test:
                for idx, (us_test, lab_test) in enumerate(test_dloader):

                    bs = us_test.shape[0]

                    out_test = net(us_test.to(args.device))
                    loss_test = criterion(out_test.view([bs, 1]).to(device=args.device).float(), lab_test.to(args.device).view([bs, 1]).to(device=args.device).float())

                    test_epoch_loss.append(loss_test.item())

                test_mean_loss.append(np.mean(np.array(test_epoch_loss)))

                print('epoch: %d, test loss: %.3f' %
                    (epoch + 1, test_mean_loss[epoch]))

        else:
            if args.include_test:
                for idx, (us_test, dmsa_test, lab_test) in enumerate(test_dloader):
                    out_test = net(us_test.to(args.device))
                    loss_test = criterion(out_test.squeeze().to(args.device), lab_test.squeeze().to(args.device))

                    test_epoch_loss.append(loss_test.item())

                test_mean_loss.append(np.mean(np.array(test_epoch_loss)))

                print('epoch: %d, test loss: %.3f' %
                      (epoch + 1, test_mean_loss[epoch]))

    if args.save_pred:
        train_pred = out.to("cpu").squeeze().float().tolist()
        train_lab = lab.to("cpu").squeeze().float().tolist()
        train_df = pd.DataFrame({"pred": train_pred, "lab": train_lab})
        train_file = args.csv_outdir + "/TrainPred_LR" + str(args.lr) + "_MOM" + str(args.mom) + "_MAXEP" + str(args.max_epochs) + "_BS" + str(args.bs) + "_" + time_str + ".pth"
        train_df.to_csv(train_file)

        if args.include_val:
            val_pred = out_val.to("cpu").squeeze().float().tolist()
            val_lab = lab_val.to("cpu").squeeze().float().tolist()
            val_df = pd.DataFrame({"pred": val_pred, "lab": val_lab})
            val_file = args.csv_outdir + "/ValPred_LR" + str(args.lr) + "_MOM" + str(args.mom) + "_MAXEP" + str(args.max_epochs) + "_BS" + str(args.bs) + "_" + time_str + ".pth"
            val_df.to_csv(val_file)

            if args.include_test:
                test_pred = out_test.to("cpu").squeeze().float().tolist()
                test_lab = lab_test.to("cpu").squeeze().float().tolist()
                test_df = pd.DataFrame({"pred": test_pred, "lab": test_lab})
                test_file = args.csv_outdir + "/TestPred_LR" + str(args.lr) + "_MOM" + str(args.mom) + "_MAXEP" + str(
                    args.max_epochs) + "_BS" + str(args.bs) + "_" + time_str + ".pth"
                test_df.to_csv(test_file)

        else:
            if args.include_test:
                test_pred = out_test.to("cpu").squeeze().float().tolist()
                test_lab = lab_test.to("cpu").squeeze().float().tolist()
                test_df = pd.DataFrame({"pred": test_pred, "lab": test_lab})
                test_file = args.csv_outdir + "/TestPred_LR" + str(args.lr) + "_MOM" + str(args.mom) + "_MAXEP" + str(
                    args.max_epochs) + "_BS" + str(args.bs) + "_" + time_str + ".pth"
                test_df.to_csv(test_file)

    if args.save_net:
        net_file = args.csv_outdir + "/Loss_LR" + str(args.lr) + "_MOM" + str(args.mom) + "_MAXEP" + str(args.max_epochs) + "_BS" + str(args.bs) + "_" + time_str + ".pth"
        print(net_file)
        torch.save(net, net_file)
        # torch.save(net, args.net_file)

    if args.include_val:
        if args.include_test:
            return {"train_loss": train_mean_loss, "val_loss": val_mean_loss, "test_loss": test_mean_loss}
    else:
        if args.include_test:
            return {"train_loss": train_mean_loss, "test_loss": test_mean_loss}
        else:
            return {"train_loss": train_mean_loss}

###
    ###
    ###        MAIN FUNCTION
    ###

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-us_dir', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/all-jpgs-dmsa/',
                        help="Directory of ultrasound images")
    parser.add_argument('-dmsa_dir', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/all-dmsa-cabs/dmsa-jpgs/',
                        help="directory of DMSA images")

    parser.add_argument("-dichot", action='store_true', default=False, help="Use dichotomous (vs continuous) outcome")

    parser.add_argument('-train_datasheet', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/data/load_training_test_sets/DMSA-train-datasheet-top2view.csv',
                        help="directory of DMSA images")
    parser.add_argument('-val_datasheet', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/data/load_training_test_sets/DMSA-val-datasheet-top2view.csv',
                        help="directory of DMSA images")
    parser.add_argument('-test_datasheet', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/data/load_training_test_sets/DMSA-test-datasheet-top2view.csv',
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

    # parser.add_argument("-net_file", default='my_net.pth',
    #                     help="File to save NN to")

    parser.add_argument("-dim", default=256, help="Image dimensions")
    parser.add_argument('-device', default='cuda',help="device to run NN on")

    parser.add_argument("-lr", default=0.005, help="Image dimensions")
    parser.add_argument("-mom", default=0.9, help="Image dimensions")
    parser.add_argument("-bs", default=128, help="Image dimensions")
    parser.add_argument("-max_epochs", default=20, help="Image dimensions")

    opt = parser.parse_args() ## comment for debug

    my_net = FuncMod

    analysis_time = "_".join(str(datetime.datetime.now()).split(" "))

    loss_dict = training_loop(opt, my_net, analysis_time)

    loss_df = pd.DataFrame(loss_dict)

    out_csvfile = opt.csv_outdir + "/Loss_AUC_LR" + str(opt.lr) + "_MOM" + str(opt.mom) + "_MAXEP" + str(opt.max_epochs) + "_BS" + str(opt.bs) + "_" + analysis_time + ".csv"
    loss_df.to_csv(out_csvfile)


# if __name__ == "__main__":
#     main()

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
