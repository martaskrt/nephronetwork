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
# from FraternalSiameseNetwork import SiamNet
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
        #to_pil = transforms.ToPILImage()
        #to_tensor = transforms.ToTensor()
        #for n in range(2):
         #   temp_img = imgs[n]
          #  m, s = temp_img.view(1, -1).mean(dim=1).numpy(), temp_img.view(1, -1).std(dim=1).numpy()
           # s[s == 0] = 1
           # norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
           # temp_img = norm(to_tensor(to_pil(temp_img)))
           # imgs[n] = temp_img
        
        return imgs, target

    def __len__(self):
        return len(self.X)


def train(args, train_X, train_y, test_X, test_y, max_epochs):
    hyperparams = {'lr': args.lr,
                   'momentum': args.momentum,
                   'weight_decay': args.weight_decay
                   }
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.num_workers}
    test_set = KidneyDataset(test_X, test_y)

    test_generator = DataLoader(test_set, **params)

    fold=1
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_y = np.array(train_y)

    train_X, train_y = shuffle(train_X, train_y, random_state=42)

    for train_index, test_index in skf.split(train_X, train_y):

        if args.view != "siamese":
            net = SiamNet(num_inputs=1).to(device)
        else:
            net = SiamNet().to(device)
        if args.checkpoint != "":
            pretrained_dict = torch.load(args.checkpoint)
            model_dict = net.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            pretrained_dict['fc6.fc6_s1.weight'] = pretrained_dict['fc6.fc6_s1.weight'].view(1024, 256, 2, 2)
            for k, v in model_dict.items():
                if k not in pretrained_dict:
                    pretrained_dict[k] = model_dict[k]
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            net.load_state_dict(pretrained_dict)

        if args.adam:
            optimizer = torch.optim.Adam(net.parameters(), lr=hyperparams['lr'],
                                         weight_decay=hyperparams['weight_decay'])
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=hyperparams['lr'], momentum=hyperparams['momentum'],
                                        weight_decay=hyperparams['weight_decay'])


        train_X_CV = train_X[train_index]
        train_y_CV = train_y[train_index]
        val_X_CV = train_X[test_index]
        val_y_CV = train_y[test_index]

        training_set = KidneyDataset(train_X_CV, train_y_CV)
        training_generator = DataLoader(training_set, **params)

        validation_set = KidneyDataset(val_X_CV, val_y_CV)
        validation_generator = DataLoader(validation_set, **params)

        for epoch in range(max_epochs):
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


            counter_train = 0
            counter_val = 0
            counter_test = 0

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

            with torch.set_grad_enabled(False):

                for batch_idx, (data, target) in enumerate(test_generator):
                    net.zero_grad()
                    optimizer.zero_grad()
                    output = net(data)
                    target = target.type(torch.LongTensor).to(device)

                    loss = F.cross_entropy(output, target)
                    loss_accum_test += loss.item() * len(target)
                    counter_test += len(target)
                    output_softmax = softmax(output)

                    accurate_labels_test += torch.sum(torch.argmax(output, dim=1) == target).cpu()

                    pred_prob = output_softmax[:, 1]
                    pred_prob = pred_prob.squeeze()
                    pred_label = torch.argmax(output, dim=1)

                    assert len(pred_prob) == len(target)
                    assert len(pred_label) == len(target)

                    all_pred_prob_test.append(pred_prob)
                    all_targets_test.append(target)
                    all_pred_label_test.append(pred_label)

            all_pred_prob_train = torch.cat(all_pred_prob_train)
            all_targets_train = torch.cat(all_targets_train)
            all_pred_label_train = torch.cat(all_pred_label_train)

            assert len(all_targets_train) == len(training_set)
            assert len(all_pred_prob_train) == len(training_set)
            assert len(all_pred_label_train) == len(training_set)

            results_train = process_results.get_metrics(y_score=all_pred_prob_train.cpu().detach().numpy(),
                                                  y_true=all_targets_train.cpu().detach().numpy(),
                                                  y_pred=all_pred_label_train.cpu().detach().numpy())

            print('Fold\t{}\tTrainEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch, int(accurate_labels_train)/counter_train,
                                                                        loss_accum_train/counter_train, results_train['auc'],
                                                                        results_train['auprc'], results_train['tn'],
                                                                        results_train['fp'], results_train['fn'],
                                                                        results_train['tp']))
            all_pred_prob_val = torch.cat(all_pred_prob_val)
            all_targets_val = torch.cat(all_targets_val)
            all_pred_label_val = torch.cat(all_pred_label_val)

            assert len(all_targets_val) == len(validation_set)
            assert len(all_pred_prob_val) == len(validation_set)
            assert len(all_pred_label_val) == len(validation_set)

            results_val = process_results.get_metrics(y_score=all_pred_prob_val.cpu().detach().numpy(),
                                                      y_true=all_targets_val.cpu().detach().numpy(),
                                                      y_pred=all_pred_label_val.cpu().detach().numpy())
            print('Fold\t{}\tValEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch, int(accurate_labels_val) / counter_val,
                                                                         loss_accum_val / counter_val, results_val['auc'],
                                                                         results_val['auprc'], results_val['tn'],
                                                                         results_val['fp'], results_val['fn'],
                                                                         results_val['tp']))

            all_pred_prob_test = torch.cat(all_pred_prob_test)
            all_targets_test = torch.cat(all_targets_test)
            all_pred_label_test = torch.cat(all_pred_label_test)

            assert len(all_targets_test) == len(test_y)
            assert len(all_pred_label_test) == len(test_y)
            assert len(all_pred_prob_test) == len(test_y)

            results_test = process_results.get_metrics(y_score=all_pred_prob_test.cpu().detach().numpy(),
                                                      y_true=all_targets_test.cpu().detach().numpy(),
                                                      y_pred=all_pred_label_test.cpu().detach().numpy())
            print('Fold\t{}\tTestEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch, int(accurate_labels_test) / counter_test,
                                                                         loss_accum_test / counter_test, results_test['auc'],
                                                                         results_test['auprc'], results_test['tn'],
                                                                         results_test['fp'], results_test['fn'],
                                                                         results_test['tp']))


            # if ((epoch+1) % 5) == 0 and epoch > 0:
            checkpoint = {'epoch': epoch,
                          'loss': loss,
                          'hyperparams': hyperparams,
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
                          'all_targets_test': all_targets_test}

            if not os.path.isdir(args.dir):
                os.makedirs(args.dir)
            path_to_checkpoint = args.dir + '/fold_' + str(fold) + "_checkpoint_" + str(epoch) + '.pth'
            #torch.save(checkpoint, path_to_checkpoint)

        fold += 1



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=50, type=int, help="Number of epochs")
    parser.add_argument('--batch_size', default=256, type=int, help="Batch size")
    parser.add_argument('--lr', default=0.001, type=float, help="Learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, help="Momentum")
    parser.add_argument('--adam', action="store_true", help="Use Adam optimizer instead of SGD")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay")
    parser.add_argument("--num_workers", default=1, type=int, help="Number of CPU workers")
    parser.add_argument("--dir", default="./", help="Directory to save model checkpoints to")
    parser.add_argument("--contrast", default=0, type=int, help="Image contrast to train on")
    parser.add_argument("--view", default = "siamese", help="siamese, sag, trans")
    parser.add_argument("--checkpoint", default="", help="Path to load pretrained model checkpoint from")
    # parser.add_argument("--datafile", default="~/nephronetwork-github/nephronetwork/preprocess/"
    #                                           "preprocessed_images_20190315.pickle", help="File containing pandas dataframe with images stored as numpy array")
    parser.add_argument("--datafile", default="../../preprocess/preprocessed_images_20190315.pickle",
                        help="File containing pandas dataframe with images stored as numpy array")
    args = parser.parse_args()

    max_epochs = args.epochs

    train_X, train_y, test_X, test_y = load_dataset.load_dataset(views_to_get=args.view, sort_by_date=True,
                                                                 pickle_file=args.datafile, contrast=args.contrast,
                                                                  split=0.9)

    #n_splits = 5
    #fold = 4
    #counter =1
    #skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    #train_y = np.array(train_y)

    #train_X, train_y = shuffle(train_X, train_y, random_state=42)

    #for train_index, test_index in skf.split(train_X, train_y):
     #   if counter != fold:
      #      counter += 1
       #     continue
        #counter += 1
        #val_X_CV = train_X[test_index]

        #load_dataset.view_images(val_X_CV, num_images_to_view=300)
    train(args,  train_X, train_y, test_X, test_y, max_epochs)


if __name__ == '__main__':
    main()
