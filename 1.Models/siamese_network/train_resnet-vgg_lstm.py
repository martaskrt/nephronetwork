import sys
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from sklearn.utils import shuffle
import importlib.machinery
from collections import defaultdict
# from torchvision import transforms
# from sklearn.utils import class_weight
# from torch import nn
# from sklearn.model_selection import KFold
# import codecs
# import errno
# import matplotlib.pyplot as plt
# from PIL import Image
# import random
# from torch import optim
# import torchvision.datasets.mnist
# from tqdm import tqdm
# from torchsummary import summary

SEED = 42
local = True
debug = True
# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
softmax = torch.nn.Softmax(dim=1)


class KidneyDataset(torch.utils.data.Dataset):

    def __init__(self, X, y, cov):
        # X = [torch.from_numpy(X).float() for e in X]
        self.X = torch.from_numpy(X).float()
        self.y = y
        self.cov = cov

    def __getitem__(self, index):
        imgs, target, cov = self.X[index], self.y[index], self.cov[index]
        return imgs, target, cov

    def __len__(self):
        return len(self.X)


def train(args, train_X, train_y, train_cov, test_X, test_y, test_cov, max_epochs):
    if not local:
        process_results = importlib.machinery.SourceFileLoader('process_results',
                                                               args.git_dir + '/nephronetwork/2.Results/process_results.py').load_module()
        sys.path.insert(0, '/Users/sulagshan/Documents/Thesis/nephronetwork/1.Models/siamese_network/')
        sys.path.insert(0, args.git_dir + '/nephronetwork/1.Models/siamese_network/')
    else:
        process_results = importlib.machinery.SourceFileLoader('process_results',
                                                               '/Users/sulagshan/Documents/Thesis/nephronetwork/2.Results/process_results.py').load_module()
        sys.path.insert(0, args.git_dir + '/nephronetwork/1.Models/siamese_network/')
        sys.path.insert(0, "/Users/sulagshan/Documents/Thesis/nephronetwork/1.Models/siamese_network")

    num_inputs = 1 if args.view != "siamese" else 2
    model_pretrain = args.pretrained if args.cv else False

    from VGGResNetSiameseLSTM import RevisedResNetLstm
    net = RevisedResNetLstm(pretrain=model_pretrain, num_inputs=num_inputs).to(device)
    print("importing ResNet18 + LSTM")

    hyperparams = {'lr': args.lr,
                   'momentum': args.momentum,
                   'weight_decay': args.weight_decay,
                   'train/test_split': args.split
                   }
    if args.adam:
        optimizer = torch.optim.Adam(net.parameters(), lr=hyperparams['lr'],
                                     weight_decay=hyperparams['weight_decay'])
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=hyperparams['lr'], momentum=hyperparams['momentum'],
                                    weight_decay=hyperparams['weight_decay'])

    args.batch_size = 1
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.num_workers}

    # Be careful to not shuffle order of image seq within a patient
    train_X, train_y, train_cov = shuffle(train_X, train_y, train_cov, random_state=SEED)
    if debug:
        train_X, train_y, train_cov, test_X, test_y, test_cov = train_X[40:42], train_y[40:42], train_cov[40:42], test_X[0:2], test_y[0:2], test_cov[0:2]

    training_set = KidneyDataset(train_X, train_y, train_cov)
    test_set = KidneyDataset(test_X, test_y, test_cov)
    training_generator = DataLoader(training_set, **params)
    test_generator = DataLoader(test_set, **params)
    print("Dataset generated")

    if debug:
        training_generator.num_workers = 0
        test_generator.num_workers = 0

    for epoch in range(args.stop_epoch + 1):
        print("Epoch " + str(epoch) + " started.")
        accurate_labels_train = 0
        accurate_labels_test = 0

        loss_accum_train = 0
        loss_accum_test = 0

        all_targets_train = []
        all_pred_prob_train = []
        all_pred_label_train = []

        all_targets_test = []
        all_pred_prob_test = []
        all_pred_label_test = []

        patient_ID_test = []
        patient_ID_train = []

        counter_train = 0
        counter_test = 0

        net.train()
        cur_patient = ''
        cur_patient_data = []
        for batch_idx, (data, target, cov) in enumerate(training_generator):
            '''
            # add if else to gather all patients then process them together
            if cur_patient == '' or cur_patient == getPatientID(cov):
                cur_patient_data
            '''
            print("batch " + str(batch_idx) + " started")
            optimizer.zero_grad()
            # print("Input data.size(): ")
            # print(data.size())
            output = net(data.to(device))
            # print("network run with batch")
            target = torch.tensor(target)  # change target to: tensor([0,1,0,1]) from: [tensor([0]), tensor([1]), tensor([0]), tensor([1])]
            target = Variable(target.type(torch.LongTensor), requires_grad=False).to(device)
            # print(output)
            # print(target)
            loss = F.cross_entropy(output, target)
            # print("loss calculated")
            print(loss)
            loss_accum_train += loss.item() * len(target)
            loss.backward()
            accurate_labels_train += torch.sum(torch.argmax(output, dim=1) == target).cpu()
            optimizer.step()
            counter_train += len(target)
            output_softmax = softmax(output)
            pred_prob = output_softmax[:, 1]
            pred_label = torch.argmax(output_softmax, dim=1)
            # print(pred_prob)
            # print(pred_label)
            assert len(pred_prob) == len(target)
            assert len(pred_label) == len(target)
            all_pred_prob_train.append(pred_prob)
            all_targets_train.append(target)
            all_pred_label_train.append(pred_label)
            patient_ID_train.extend(cov)

        net.eval()
        with torch.no_grad():

            for batch_idx, (data, target, cov) in enumerate(test_generator):
                net.zero_grad()
                optimizer.zero_grad()
                output = net(data)
                target = torch.tensor(target)
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

                patient_ID_test.extend(cov)

        all_pred_prob_train = torch.cat(all_pred_prob_train)
        all_targets_train = torch.cat(all_targets_train)
        all_pred_label_train = torch.cat(all_pred_label_train)
        totalTrainItems = sum(len(e) for e in train_y)

        assert len(all_targets_train) == totalTrainItems
        assert len(all_pred_prob_train) == totalTrainItems
        assert len(all_pred_label_train) == totalTrainItems
        assert len(patient_ID_train) == totalTrainItems

        results_train = process_results.get_metrics(y_score=all_pred_prob_train.cpu().detach().numpy(),
                                                    y_true=all_targets_train.cpu().detach().numpy(),
                                                    y_pred=all_pred_label_train.cpu().detach().numpy())

        print('TrainEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
              'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(epoch,
                                                                     int(accurate_labels_train) / counter_train,
                                                                     loss_accum_train / counter_train,
                                                                     results_train['auc'],
                                                                     results_train['auprc'],
                                                                     results_train['tn'],
                                                                     results_train['fp'], results_train['fn'],
                                                                     results_train['tp']))

        all_pred_prob_test = torch.cat(all_pred_prob_test)
        all_targets_test = torch.cat(all_targets_test)
        all_pred_label_test = torch.cat(all_pred_label_test)
        totalTestItems = sum(len(e) for e in test_y)

        assert len(all_targets_test) == totalTestItems
        assert len(all_pred_label_test) == totalTestItems
        assert len(all_pred_prob_test) == totalTestItems
        assert len(patient_ID_test) == totalTestItems

        results_test = process_results.get_metrics(y_score=all_pred_prob_test.cpu().detach().numpy(),
                                                   y_true=all_targets_test.cpu().detach().numpy(),
                                                   y_pred=all_pred_label_test.cpu().detach().numpy())
        print('TestEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
              'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(epoch,
                                                                     int(accurate_labels_test) / counter_test,
                                                                     loss_accum_test / counter_test,
                                                                     results_test['auc'],
                                                                     results_test['auprc'], results_test['tn'],
                                                                     results_test['fp'], results_test['fn'],
                                                                     results_test['tp']))

        if ((epoch+1) % 5) == 0 and epoch > 0:
            checkpoint = {'epoch': epoch,
                          'loss': loss,
                          'hyperparams': hyperparams,
                          'args': args,
                          'model_state_dict': net.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'loss_train': loss_accum_train / counter_train,
                          'loss_test': loss_accum_test / counter_test,
                          'accuracy_train': int(accurate_labels_train) / counter_train,
                          'accuracy_test': int(accurate_labels_test) / counter_test,
                          'results_train': results_train,
                          'results_test': results_test,
                          'all_pred_prob_train': all_pred_prob_train,
                          'all_pred_prob_test': all_pred_prob_test,
                          'all_targets_train': all_targets_train,
                          'all_targets_test': all_targets_test,
                          'patient_ID_train': patient_ID_train,
                          'patient_ID_test': patient_ID_test}

        if not os.path.isdir(args.dir):
            os.makedirs(args.dir)
        # if not os.path.isdir(args.dir + "/" + str(fold)):
        # os.makedirs(args.dir + "/" + str(fold))

        ## UNCOMMENT THIS WHEN YOU WANT TO START SAVING YOUR MODELS!
        # path_to_checkpoint = args.dir + "/" + str(fold) + "_checkpoint_" + str(epoch) + '.pth'
        # torch.save(checkpoint, path_to_checkpoint)

        if epoch == args.stop_epoch:
            path_to_checkpoint = args.dir + "/checkpoint_" + str(epoch) + '.pth'
            torch.save(checkpoint, path_to_checkpoint)

'''
1.0_1.0_M_2_Right_2016-10-07_2017-07-11_philips-medical-systems
todo:implementations needs to be finished
'''
def organizeDataForLstm(train_x, train_y, train_cov, test_x, test_y, test_cov):
    # This code can be used to confirm that organizeData is keeping data in sync by using oldMatch to conform after
    # oldMatch = {}
    # for i in range(len(train_x)):
    #     oldMatch[train_cov[i]] = (train_x[i], train_y[i])
    # oldMatchTest = {}
    # for i in range(len(test_x)):
    #     oldMatchTest[test_cov[i]] = (test_x[i], test_y[i])

    def sortData(t_x, t_y, t_cov):
        train_cov, train_x, train_y = zip(*sorted(zip(t_cov, t_x, t_y), key=lambda x: float(x[0].split("_")[0])))
        return train_x, train_y, train_cov
    
    def group(t_x, t_y, t_cov):
        x, y, cov = defaultdict(list), defaultdict(list), defaultdict(list)
        for i in range(len(t_cov)):
            id = t_cov[i].split("_")[0]
            x[id].append(t_x[i])
            y[id].append(t_y[i])
            cov[id].append(t_cov[i])
        # convert to np array
        organized_train_x = np.asarray([np.asarray(e) for e in list(x.values())])
        return organized_train_x, np.asarray(list(y.values())), np.asarray(list(cov.values()))

    def allowOnlyNVisits(t_x, t_y, t_cov, n=4):
        x, y, cov = [],[],[]
        for i, e in enumerate(t_x):
            if len(e) == n:
                x.append(e)
                y.append(t_y[i])
                cov.append(t_cov[i])
        return np.asarray(x, dtype=np.float64), y, cov
    
    train_x, train_y, train_cov = sortData(train_x, train_y, train_cov)
    train_x, train_y, train_cov = group(train_x, train_y, train_cov)
    train_x, train_y, train_cov = allowOnlyNVisits(train_x, train_y, train_cov)
    test_x, test_y, test_cov = sortData(test_x, test_y, test_cov)
    test_x, test_y, test_cov = group(test_x, test_y, test_cov)
    test_x, test_y, test_cov = allowOnlyNVisits(test_x, test_y, test_cov)
    
    return train_x, train_y, train_cov, test_x, test_y, test_cov

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',         default=30,     type=int,   help="Number of epochs")
    parser.add_argument('--batch_size',     default=16,     type=int,   help="Batch size")
    parser.add_argument('--lr',             default=0.001,  type=float, help="Learning rate")
    parser.add_argument('--momentum',       default=0.9,    type=float, help="Momentum")
    parser.add_argument("--weight_decay",   default=5e-4,   type=float, help="Weight decay")
    parser.add_argument("--num_workers",    default=1,      type=int,   help="Number of CPU workers")
    parser.add_argument("--contrast",       default=1,      type=int,   help="Image contrast to train on")
    parser.add_argument("--split",          default=0.7,    type=float, help="proportion of dataset to use as training")
    parser.add_argument("--bottom_cut",     default=0.0,    type=float, help="proportion of dataset to cut from bottom")
    parser.add_argument("--crop",           default=0,      type=int,   help="Crop setting (0=big, 1=tight)")
    parser.add_argument("--output_dim",     default=128,    type=int,   help="output dim for last linear layer")
    parser.add_argument("--git_dir",        default="C:/Users/Lauren/Desktop/DS Core/Projects/Urology/")
    parser.add_argument("--stop_epoch",     default=18,     type=int,   help="If not running cross validation, which epoch to finish with")
    parser.add_argument("--cv_stop_epoch",  default=18,     type=int,   help="get a pth file from a specific epoch")
    parser.add_argument("--view",           default="siamese",  help="siamese, sag, trans")
    parser.add_argument("--dir",            default="./",       help="Directory to save model checkpoints to")
    parser.add_argument("--checkpoint",     default="",         help="Path to load pretrained model checkpoint from")
    parser.add_argument("--etiology",       default="B",        help="O (obstruction), R (reflux), B (both)")
    parser.add_argument('--cv',             action='store_true', help="Flag to run cross validation")
    parser.add_argument('--adam',           action="store_true", help="Use Adam optimizer instead of SGD")
    parser.add_argument('--vgg',            action='store_true', help="Run VGG16 architecture, not using this flag runs ResNet")
    parser.add_argument('--vgg_bn',         action='store_true', help="Run VGG16 batch norm architecture")
    parser.add_argument('--densenet',       action='store_true', help="Run DenseNet")
    parser.add_argument('--resnet18',       action='store_true', help="Run ResNet18 architecture, not using this flag runs ResNet16")
    parser.add_argument('--resnet50',       action='store_true', help="Run ResNet50 architecture, not using this flag runs ResNet50")
    parser.add_argument('--pretrained',     action="store_true", help="Use pretrained model with cross validation if cv requested")
    # parser.add_argument('--unet',         action="store_true", help="UNet architecthure")
    return parser.parse_args()

def main():
    args = parseArgs()

    # print("batch size: " + str(args.batch_size))
    # print("lr: " + str(args.lr))
    # print("momentum: " + str(args.momentum))
    # print("adam optimizer: " + str(args.adam))
    # print("weight decay: " + str(args.weight_decay))
    # print("view: " + str(args.view))
    # print("pretrained weights:" + str(args.pretrained))

    max_epochs = args.epochs
    if not local:
        datafile = args.git_dir + "nephronetwork/0.Preprocess/preprocessed_images_20190617.pickle"

        load_dataset_LE = importlib.machinery.SourceFileLoader('load_dataset_LE',
                                                               args.git_dir + '/nephronetwork/0.Preprocess/load_dataset_LE.py').load_module()

        train_X, train_y, train_cov, test_X, test_y, test_cov = load_dataset_LE.load_dataset(
            views_to_get=args.view,
            sort_by_date=True,
            pickle_file=datafile,
            contrast=args.contrast,
            split=args.split,
            get_cov=True,
            bottom_cut=args.bottom_cut,
            etiology=args.etiology,
            crop=args.crop,
            git_dir=args.git_dir
        )
    else:
        data_losader = importlib.machinery.SourceFileLoader("loadData", "/Users/sulagshan/Documents/Thesis/logs/loadData.py").load_module()
        train_X, train_y, train_cov, test_X, test_y, test_cov = data_loader.load()

    train_X, train_y, train_cov, test_X, test_y, test_cov = organizeDataForLstm(train_X, train_y, train_cov, test_X, test_y, test_cov)
    train(args, train_X, train_y, train_cov, test_X, test_y, test_cov, max_epochs)
    print("len: train_X, train_y, train_cov, test_X, test_y, test_cov")
    print(len(train_X), len(train_y), len(train_cov), len(test_X), len(test_y), len(test_cov))
    print("note length is number of kidneys being processed, each of which has a seq of images")

if __name__ == '__main__':
    main()

#########################################
######## DEBUGGING MAIN FUNCTION
#########################################

# args_dict = {'epochs': 35,
#         'batch_size': 16,
#         'lr': 0.001, # 0.005
#         'momentum': 0.9,
#         'adam': False,
#         'weight_decay': 5e-4,
#         'num_workers': 0,
#         'dir': './',
#         'contrast': 1,
#         'view': 'sag', ## trans, sag
#         'checkpoint':"", ## use if you want to do transfer learning -- could be interesting to use pretrained resnet and vgg,
#         'split':0.7,
#         'bottom_cut':0.0,
#         'etiology':'B',
#         'crop':0,
#         'output_dim':128, ## may be unnecessary in this code,
#         'git_dir':"C:/Users/Lauren/Desktop/DS Core/Projects/Urology/",
#         'vgg': False,
#         'resnet18': True,
#         'resnet50': False,
#         'cv': False,
#         'stop_epoch':18
#         }
#
# class myargs():
#     def __init__(self,args_dict):
#         self.epochs = args_dict['epochs']
#         self.batch_size = args_dict['batch_size']
#         self.lr = args_dict['lr']
#         self.momentum = args_dict['momentum']
#         self.adam = args_dict['adam']
#         self.weight_decay = args_dict['weight_decay']
#         self.num_workers = args_dict['num_workers']
#         self.dir = args_dict['dir']
#         self.contrast = args_dict['contrast']
#         self.view = args_dict['view']
#         self.checkpoint = args_dict['checkpoint']
#         self.split = args_dict['split']
#         self.bottom_cut = args_dict['bottom_cut']
#         self.etiology = args_dict['etiology']
#         self.crop = args_dict['crop']
#         self.output_dim = args_dict['output_dim']
#         self.git_dir = args_dict['git_dir']
#         self.vgg = args_dict['vgg']
#         self.resnet18 = args_dict['resnet18']
#         self.resnet50 = args_dict['resnet50']
#         self.cv = args_dict['cv']
#         self.stop_epoch = args_dict['stop_epoch']
#
# args = myargs(args_dict)
# args.git_dir
#
# datafile = args.git_dir + "nephronetwork/0.Preprocess/preprocessed_images_20190601.pickle" ## CHANGE THIS IN THE MAIN FUNCTION
#
# sys.path.insert(0, args.git_dir + '/nephronetwork/0.Preprocess/')
# import load_dataset_LE
# sys.path.insert(0, args.git_dir + '/nephronetwork/2.Results/')
# import process_results
#
# max_epochs = args.epochs
# train_X, train_y, train_cov, test_X, test_y, test_cov = load_dataset_LE.load_dataset(views_to_get=args.view,
#                                                                                   sort_by_date=True,
#                                                                                   pickle_file=datafile,
#                                                                                   contrast=args.contrast,
#                                                                                   split=args.split,
#                                                                                   get_cov=True,
#                                                                                   bottom_cut=args.bottom_cut,
#                                                                                   etiology=args.etiology,
#                                                                                   crop=args.crop,
#                                                                                   git_dir=args.git_dir)
#
# # if args.view == "sag" or args.view == "trans":
# #     train_X_single = []
# #     test_X_single = []
# #
# #     for item in train_X:
# #         if args.view == "sag":
# #             train_X_single.append(item[0])
# #         elif args.view  == "trans":
# #             train_X_single.append(item[1])
# #     for item in test_X:
# #         if args.view == "sag":
# #             test_X_single.append(item[0])
# #         elif args.view == "trans":
# #             test_X_single.append(item[1])
# #
# #     train_X = np.array(train_X_single)
# #     test_X = np.array(test_X_single)
#
# train(args, train_X, train_y, train_cov, test_X, test_y, test_cov, max_epochs)
#
# print(len(train_X), len(train_y), len(train_cov), len(test_X), len(test_y), len(test_cov))

'''
    if args.vgg:
        from VGGResNetSiamese import RevisedVGG
        net = RevisedVGG(pretrain=model_pretrain, num_inputs=num_inputs).to(device)
        print("importing VGG16 without BN")
    elif args.vgg_bn:
        from VGGResNetSiamese import RevisedVGG_bn
        net = RevisedVGG_bn(pretrain=model_pretrain, num_inputs=num_inputs).to(device)
        print("importing VGG16 with BN")
    elif args.densenet:
        from VGGResNetSiamese import RevisedDenseNet
        net = RevisedDenseNet(pretrain=model_pretrain, num_inputs=num_inputs).to(device)
        print("Importing DenseNet")
    elif args.resnet18:
        from VGGResNetSiamese import RevisedResNet
        net = RevisedResNet(pretrain=model_pretrain, num_inputs=num_inputs).to(device)
        print("importing ResNet18")
    elif args.resnet50:
        from VGGResNetSiamese import RevisedResNet50
        net = RevisedResNet50(pretrain=model_pretrain, num_inputs=num_inputs).to(device)
        print("importing ResNet50")
    elif args.lstm:
    '''