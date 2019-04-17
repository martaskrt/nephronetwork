import codecs
import errno
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.datasets.mnist
from torchvision import transforms
from tqdm import tqdm
import importlib.machinery
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import argparse
from torch.autograd import Variable
load_dataset = importlib.machinery.SourceFileLoader('load_dataset','../../preprocess/load_dataset.py').load_module()
process_results = importlib.machinery.SourceFileLoader('process_results','../process_results.py').load_module()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
softmax = torch.nn.Softmax(dim=1)

class KidneyDataset(torch.utils.data.Dataset):

    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = y

    def __getitem__(self, index):
        imgs, target = self.X[index], self.y[index]
        return imgs, target

    def __len__(self):
        return len(self.X)


class SiamNet(nn.Module):
    def __init__(self, classes=2):
        super(SiamNet, self).__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv1_s1',nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=0))
        self.conv.add_module('batch_s1', nn.BatchNorm2d(96))
        self.conv.add_module('relu1_s1',nn.ReLU(inplace=True))
        self.conv.add_module('pool1_s1',nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv.add_module('lrn1_s1',LRN(local_size=5, alpha=0.0001, beta=0.75))

        self.conv.add_module('conv2_s1',nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2))
        self.conv.add_module('batch_s1', nn.BatchNorm2d(256))
        self.conv.add_module('relu2_s1',nn.ReLU(inplace=True))
        self.conv.add_module('pool2_s1',nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv.add_module('lrn2_s1',LRN(local_size=5, alpha=0.0001, beta=0.75))

        self.conv.add_module('conv3_s1',nn.Conv2d(256, 384, kernel_size=3, padding=1))
        self.conv.add_module('batch_s1', nn.BatchNorm2d(384))
        self.conv.add_module('relu3_s1',nn.ReLU(inplace=True))

        self.conv.add_module('conv4_s1',nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2))
        self.conv.add_module('batch_s1', nn.BatchNorm2d(384))
        self.conv.add_module('relu4_s1',nn.ReLU(inplace=True))

        self.conv.add_module('conv5_s1',nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2))
        self.conv.add_module('batch_s1', nn.BatchNorm2d(256))
        self.conv.add_module('relu5_s1',nn.ReLU(inplace=True))
        self.conv.add_module('pool5_s1',nn.MaxPool2d(kernel_size=3, stride=2))
        # *************************** changed layers *********************** #
        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6_s1', nn.Conv2d(256, 1024, kernel_size=3, stride=1, padding=1))
        self.fc6.add_module('batch_s1', nn.BatchNorm2d(1024))

        self.fc6b = nn.Sequential()
        self.fc6b.add_module('conv6b_s1', nn.Conv2d(1024, 256, kernel_size=3, stride=2))
        self.fc6b.add_module('batch_s1', nn.BatchNorm2d(256))
        self.fc6b.add_module('conv6_s1', nn.ReLU(inplace=True))
        self.fc6b.add_module('pool6_s1', nn.MaxPool2d(kernel_size=3, stride=2))

        self.fc6c = nn.Sequential()
        self.fc6c.add_module('fc7', nn.Linear(1024, 512))
        self.fc6c.add_module('relu7', nn.ReLU(inplace=True))
        self.fc6c.add_module('drop7', nn.Dropout(p=0.5))

        self.fc7_new = nn.Sequential()
        self.fc7_new.add_module('fc7', nn.Linear(2 * 512, 4096))
        self.fc7_new.add_module('relu7', nn.ReLU(inplace=True))
        self.fc7_new.add_module('drop7', nn.Dropout(p=0.5))

        self.classifier_new = nn.Sequential()
        self.classifier_new.add_module('fc8', nn.Linear(4096, classes))

    def load(self,checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)
        pretrained_dict = {k: v for k, v in list(pretrained_dict.items()) if k in model_dict and 'fc8' not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print([k for k, v in list(pretrained_dict.items())])

    def save(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

    def forward(self, x):
        B, T, C, H = x.size()
        x = x.transpose(0, 1)
        x_list = []

        for i in range(2):
            curr_x = torch.unsqueeze(x[i], 1)
            curr_x = curr_x.expand(-1, 3, -1, -1)
            if torch.cuda.is_available():
                input = torch.cuda.FloatTensor(curr_x.to(device))
            else:
                input = torch.cpu.FloatTensor(curr_x)
            z = self.conv(input)
            z = self.fc6(z)
            z = self.fc6b(z)
            z = z.view([B, 1, -1])
            z = self.fc6c(z)
            z = z.view([B, 1, -1])
            x_list.append(z)

        x = torch.cat(x_list, 1)
        x = self.fc7_new(x.view(B, -1))
        pred = self.classifier_new(x)

        return pred


class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,padding=int((local_size-1.0)/2))
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


def train(args, train_dataset, val_dataset, max_epochs):
    net = SiamNet().to(device)
    if args.checkpoint != "":
        pretrained_dict = torch.load(args.checkpoint)
        model_dict = net.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        pretrained_dict['fc6.fc6_s1.weight'] = pretrained_dict['fc6.fc6_s1.weight'].view(1024, 256, 3, 3)
        for k,v in model_dict.items():
            if k not in pretrained_dict:
                pretrained_dict[k] = model_dict[k]
        # pretrained_dict['fc6b.conv6b_s1.weight'] = model_dict['fc6b.conv6b_s1.weight']
        # pretrained_dict['fc6b.conv6b_s1.bias'] = model_dict['fc6b.conv6b_s1.bias']
        # pretrained_dict['fc6c.fc7.weight'] = model_dict['fc6c.fc7.weight']
        # pretrained_dict['fc6c.fc7.bias'] = model_dict['fc6c.fc7.bias']
        # pretrained_dict['fc7_new.fc7.weight'] = model_dict['fc7_new.fc7.weight']
        # pretrained_dict['fc7_new.fc7.bias'] = model_dict['fc7_new.fc7.bias']
        # pretrained_dict['classifier_new.fc8.weight'] = model_dict['classifier_new.fc8.weight']
        # pretrained_dict['classifier_new.fc8.bias'] = model_dict['classifier_new.fc8.bias']
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        net.load_state_dict(pretrained_dict)

    # print(summary(net, (2, 256, 256)))
    # import sys
    # sys.exit(0)
    hyperparams = {'lr': args.lr,
                   'momentum': args.momentum,
                   'weight_decay': args.weight_decay
                }
    # optimizer = torch.optim.SGD(net.parameters(), lr=hyperparams['lr'], momentum=hyperparams['momentum'],
    #                             weight_decay=hyperparams['weight_decay'])

    optimizer = torch.optim.Adam(net.parameters(), lr=hyperparams['lr'], weight_decay=hyperparams['weight_decay'])

    for epoch in range(max_epochs):
        all_labels = 0
        accurate_labels = 0
        loss_accum = 0
        counter =0
        all_targets = []
        all_pred_prob = []
        all_pred_label = []
        for batch_idx, (data, target) in enumerate(train_dataset):
            optimizer.zero_grad()
            output = net(data.to(device))
            target = Variable(target.type(torch.LongTensor), requires_grad=False).to(device)

            loss = F.cross_entropy(output, target)
            loss_accum += loss.item() * len(target)
            counter += len(target)
            loss.backward()

            accurate_labels += torch.sum(torch.argmax(output, dim=1) == target).cpu()
            optimizer.step()
            all_labels += len(target)

            output_softmax = softmax(output)
            pred_prob = output_softmax[:, 1]
            pred_label = torch.argmax(output, dim=1)

            #pred_prob = pred_prob.squeeze()

            all_pred_prob.append(pred_prob)
            all_targets.append(target)
            all_pred_label.append(pred_label)
            # results = process_results.get_metrics(y_score=pred_prob.cpu().detach().numpy(),
            #                                       y_true=target.cpu().detach().numpy())

        all_pred_prob = torch.cat(all_pred_prob)
        all_targets = torch.cat(all_targets)
        all_pred_label = torch.cat(all_pred_label)
        results = process_results.get_metrics(y_score=all_pred_prob.cpu().detach().numpy(),
                                              y_true=all_targets.cpu().detach().numpy(),
                                              y_pred=all_pred_label.cpu().detach().numpy())
        print('TrainEpoch\t{}\tACC\t{:.0f}%\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
              'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(epoch, 100.*accurate_labels/all_labels,
                                                                    loss_accum/counter, results['auc'],results['auprc'],
                                                                    results['tn'], results['tp'], results['fn'],
                                                                    results['tp']))
        # print("TRAIN" + '\t' + "AUC" + '\t' + str(results['auc']) + '\t' + "AUPRC" + '\t' + str(results['auprc']))

        if ((epoch+1) % 10) == 0:
            checkpoint = {'epoch': epoch,
                          'loss': loss,
                          'hyperparams': hyperparams,
                          'model_state_dict': net.state_dict(),
                          'optimizer': optimizer.state_dict()}
            if not os.path.isdir(args.dir):
                os.makedirs(args.dir)
            path_to_checkpoint = args.dir + '/' + "checkpoint_" + str(epoch) + '.pth'
            torch.save(checkpoint, path_to_checkpoint)


        with torch.set_grad_enabled(False):
            accurate_labels_val = 0
            all_labels_val = 0
            loss_accum = 0
            counter = 0
            all_targets = []
            all_pred_prob = []
            all_pred_label = []
            for batch_idx, (data, target) in enumerate(val_dataset):
                net.zero_grad()
                optimizer.zero_grad()
                output = net(data)
                target = target.type(torch.LongTensor).to(device)

                loss = F.cross_entropy(output, target)
                loss_accum += loss.item() * len(target)
                counter += len(target)
                output_softmax = softmax(output)

                accurate_labels_val += torch.sum(torch.argmax(output, dim=1) == target).cpu()
                all_labels_val += len(target)

                pred_prob = output_softmax[:,1]
                pred_prob = pred_prob.squeeze()
                pred_label = torch.argmax(output, dim=1)

                all_pred_prob.append(pred_prob)
                all_targets.append(target)
                all_pred_label.append(pred_label)

                assert pred_prob.shape == target.shape
            # results = process_results.get_metrics(y_score=pred_prob.cpu().numpy(), y_true=target.cpu().numpy())
            # print("VAL.............AUC: " + str(results['auc']) + " | AUPRC: " + str(results['auprc']))
            #
            #
            # accuracy = 100. * accurate_labels_val / all_labels_val
            # print('Test accuracy: {}/{} ({:.3f}%)\tLoss: {:.6f}'.format(accurate_labels_val, all_labels_val, accuracy, loss_accum/counter))

            all_pred_prob = torch.cat(all_pred_prob)
            all_targets = torch.cat(all_targets)
            all_pred_label = torch.cat(all_pred_label)
            results = process_results.get_metrics(y_score=all_pred_prob.cpu().detach().numpy(),
                                                  y_true=all_targets.cpu().detach().numpy(),
                                                  y_pred=all_pred_label.cpu().detach().numpy())
            print('ValEpoch\t{}\tACC\t{:.0f}%\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(epoch, 100. * accurate_labels / all_labels,
                                                                        loss_accum / counter, results['auc'],
                                                                        results['auprc'],
                                                                        results['tn'], results['tp'], results['fn'],
                                                                        results['tp']))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=50, type=int, help="Number of epochs")
    parser.add_argument('--batch_size', default=256, type=int, help="Batch size")
    parser.add_argument('--lr', default=0.001, type=float, help="Learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, help="Momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay")
    parser.add_argument("--num_workers", default=1, type=int, help="Number of CPU workers")
    parser.add_argument("--dir", default="./", help="Directory to save model checkpoints to")
    parser.add_argument("--contrast", default=0, type=int, help="Image contrast to train on")
    parser.add_argument("--checkpoint", default="", help="Path to load pretrained model checkpoint from")
    parser.add_argument("--datafile", default="~/nephronetwork-github/nephronetwork/preprocess/"
                                              "preprocessed_images_20190315.pickle", help="File containing pandas dataframe with images stored as numpy array")

    args = parser.parse_args()

    max_epochs = args.epochs

    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.num_workers}

    train_X, train_y, test_X, test_y = load_dataset.load_dataset(views_to_get="siamese", pickle_file=args.datafile,
                                                                 contrast=args.contrast)

    training_set = KidneyDataset(train_X, train_y)
    training_generator = DataLoader(training_set, **params)

    validation_set = KidneyDataset(test_X, test_y)
    validation_generator = DataLoader(validation_set, **params)

    train(args, training_generator, validation_generator, max_epochs)


if __name__ == '__main__':
    main()