from sklearn.utils import shuffle
# from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
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
from SiameseNetwork import SiamNet
# from SiameseNetworkUNet import SiamNet
load_dataset = importlib.machinery.SourceFileLoader('load_dataset','../../preprocess/load_dataset.py').load_module()
process_results = importlib.machinery.SourceFileLoader('process_results','../process_results.py').load_module()

SEED = 42

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

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


def train(args, train_X, train_y, test_X, test_y, max_epochs):
    net = SiamNet().to(device)
    if args.checkpoint != "":
        pretrained_dict = torch.load(args.checkpoint)
        model_dict = net.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        pretrained_dict['fc6.fc6_s1.weight'] = pretrained_dict['fc6.fc6_s1.weight'].view(1024, 256, 2, 2)
        for k,v in model_dict.items():
            if k not in pretrained_dict:
                pretrained_dict[k] = model_dict[k]
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

    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.num_workers}


    for epoch in range(max_epochs):
        accurate_labels_train = 0
        accurate_labels_val = 0

        loss_accum_train = 0
        loss_accum_val = 0

        all_targets_train = []
        all_pred_prob_train = []
        all_pred_label_train = []

        all_targets_val = []
        all_pred_prob_val = []
        all_pred_label_val = []

        n_splits = 5
        counter_train = 0
        counter_val = 0
        skf = StratifiedKFold(n_splits=n_splits)
        train_y = np.array(train_y)
        for train_index, test_index in skf.split(train_X, train_y):
            train_X_CV = train_X[train_index]
            train_y_CV = train_y[train_index]
            val_X_CV = train_X[test_index]
            val_y_CV = train_y[test_index]

            training_set = KidneyDataset(train_X_CV, train_y_CV)
            training_generator = DataLoader(training_set, **params)

            validation_set = KidneyDataset(val_X_CV, val_y_CV)
            validation_generator = DataLoader(validation_set, **params)

            for batch_idx, (data, target) in enumerate(training_generator):
                optimizer.zero_grad()

                output = net(data.to(device))
                target = Variable(target.type(torch.LongTensor), requires_grad=False).to(device)

                loss = F.cross_entropy(output, target)
                loss_accum_train += loss.item() * len(target)

                loss.backward()

                accurate_labels_train += torch.sum(torch.argmax(output, dim=1) == target).cpu()
                optimizer.step()
                counter_train += len(target)

                output_softmax = softmax(output)
                pred_prob = output_softmax[:, 1]
                pred_label = torch.argmax(output_softmax, dim=1)

                assert len(pred_prob) == len(target)
                assert len(pred_label) == len(target)

                all_pred_prob_train.append(pred_prob)
                all_targets_train.append(target)
                all_pred_label_train.append(pred_label)

            with torch.set_grad_enabled(False):
                for batch_idx, (data, target) in enumerate(validation_generator):
                    net.zero_grad()
                    optimizer.zero_grad()
                    output = net(data)
                    target = target.type(torch.LongTensor).to(device)

                    loss = F.cross_entropy(output, target)
                    loss_accum_val += loss.item() * len(target)
                    counter_val += len(target)
                    output_softmax = softmax(output)

                    accurate_labels_val += torch.sum(torch.argmax(output, dim=1) == target).cpu()

                    pred_prob = output_softmax[:,1]
                    pred_prob = pred_prob.squeeze()
                    pred_label = torch.argmax(output, dim=1)

                    assert len(pred_prob) == len(target)
                    assert len(pred_label) == len(target)

                    all_pred_prob_val.append(pred_prob)
                    all_targets_val.append(target)
                    all_pred_label_val.append(pred_label)


        all_pred_prob_train = torch.cat(all_pred_prob_train)
        all_targets_train = torch.cat(all_targets_train)
        all_pred_label_train = torch.cat(all_pred_label_train)

        assert len(all_targets_train) == len(train_y)*(n_splits-1)
        assert len(all_pred_prob_train) == len(train_y)*(n_splits-1)
        assert len(all_pred_label_train) == len(train_y)*(n_splits-1)

        results_train = process_results.get_metrics(y_score=all_pred_prob_train.cpu().detach().numpy(),
                                              y_true=all_targets_train.cpu().detach().numpy(),
                                              y_pred=all_pred_label_train.cpu().detach().numpy())

        print('TrainEpoch\t{}\tACC\t{:.0f}%\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
              'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(epoch, 100.*accurate_labels_train/counter_train,
                                                                    loss_accum_train/counter_train, results_train['auc'],
                                                                    results_train['auprc'], results_train['tn'],
                                                                    results_train['fp'], results_train['fn'],
                                                                    results_train['tp']))
        all_pred_prob_val = torch.cat(all_pred_prob_val)
        all_targets_val = torch.cat(all_targets_val)
        all_pred_label_val = torch.cat(all_pred_label_val)

        assert len(all_targets_val) == len(train_y)
        assert len(all_pred_prob_val) == len(train_y)
        assert len(all_pred_label_val) == len(train_y)

        results_val = process_results.get_metrics(y_score=all_pred_prob_val.cpu().detach().numpy(),
                                                  y_true=all_targets_val.cpu().detach().numpy(),
                                                  y_pred=all_pred_label_val.cpu().detach().numpy())
        print('ValEpoch\t{}\tACC\t{:.0f}%\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
              'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(epoch, 100. * accurate_labels_val / counter_val,
                                                                     loss_accum_val / counter_val, results_val['auc'],
                                                                     results_val['auprc'], results_val['tn'],
                                                                     results_val['fp'], results_val['fn'],
                                                                     results_val['tp']))
            #
            # if ((epoch+1) % 10) == 0:
            #     checkpoint = {'epoch': epoch,
            #                   'loss': loss,
            #                   'hyperparams': hyperparams,
            #                   'model_state_dict': net.state_dict(),
            #                   'optimizer': optimizer.state_dict()}
            #     if not os.path.isdir(args.dir):
            #         os.makedirs(args.dir)
            #     path_to_checkpoint = args.dir + '/' + "checkpoint_" + str(epoch) + '.pth'
            #     torch.save(checkpoint, path_to_checkpoint)



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



    train_X, train_y, test_X, test_y = load_dataset.load_dataset(views_to_get="siamese", sort_by_date=True,
                                                                 pickle_file=args.datafile, contrast=args.contrast,
                                                                 split=0.9)


    train(args,  train_X, train_y, test_X, test_y, max_epochs)


if __name__ == '__main__':
    main()
