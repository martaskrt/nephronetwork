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
from sklearn.utils import class_weight

# from FraternalSiameseNetwork import SiamNet
load_dataset = importlib.machinery.SourceFileLoader('load_dataset', '../../0.Preprocess/load_dataset.py').load_module()
process_results = importlib.machinery.SourceFileLoader('process_results', '../../2.Results/process_results.py').load_module()

SEED = 42

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
softmax = torch.nn.Softmax(dim=1)


class KidneyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, cov):
        self.X = torch.from_numpy(X).float()
        self.y = y
        self.cov = cov

    def __getitem__(self, index):
        imgs, target, cov = self.X[index], self.y[index], self.cov[index]

        return imgs, target, cov

    def __len__(self):
        return len(self.X)


def train(args, train_X, train_y, train_cov, test_X, test_y, test_cov, max_epochs):
    if args.unet:
        print("importing UNET")
        if args.sc == 5:
            from SiameseNetworkUNet import SiamNet
        elif args.sc == 4:
            from SiameseNetworkUNet_sc4_nosc3 import SiamNet
        elif args.sc == 3:
            from SiameseNetworkUNet_sc3 import SiamNet
        elif args.sc == 2:
            from SiameseNetworkUNet_sc2 import SiamNet
        elif args.sc == 1:
            from SiameseNetworkUNet_sc1 import SiamNet
        elif args.sc == 0:
            if args.init == "none":
                #from FraternalSiameseNetwork_20190619 import SiamNet
                from SiameseNetworkUNet_upconv_1c_1ch import SiamNet
            elif args.init == "fanin":
                from SiameseNetworkUNet_upconv_1c_1ch_fanin import SiamNet
            elif args.init == "fanout":
                from SiameseNetworkUNet_upconv_1c_1ch_fanout import SiamNet
    else:
        print("importing SIAMNET")
        from SiameseNetwork import SiamNet

    hyperparams = {'lr': args.lr,
                   'momentum': args.momentum,
                   'weight_decay': args.weight_decay,
                   'train/test_split': args.split
                   }
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.num_workers}

    test_set = KidneyDataset(test_X, test_y, test_cov)

    test_generator = DataLoader(test_set, **params)

    train_X, train_y, train_cov = shuffle(train_X, train_y, train_cov, random_state=42)

    jigsaw = False
    if "jigsaw" in args.dir and "unet" in args.dir:
        jigsaw = True
    if args.view != "siamese":
        net = SiamNet(num_inputs=1, output_dim=args.output_dim, jigsaw=jigsaw).to(device)
    else:
        net = SiamNet(output_dim=args.output_dim, jigsaw=jigsaw).to(device)
    if args.checkpoint != "":
        if "jigsaw" in args.dir and "unet" in args.dir:
            print("Loading Jigsaw into UNet")
            pretrained_dict = torch.load(args.checkpoint)
            model_dict = net.state_dict()
            unet_dict = {}

            for k, v in model_dict.items():
                unet_dict[k] = model_dict[k]

            unet_dict['conv1.conv1_s1.weight'] = pretrained_dict['conv.conv1_s1.weight']
            unet_dict['conv1.conv1_s1.bias'] = pretrained_dict['conv.conv1_s1.bias']
            unet_dict['conv2.conv2_s1.weight'] = pretrained_dict['conv.conv2_s1.weight']
            unet_dict['conv2.conv2_s1.bias'] = pretrained_dict['conv.conv2_s1.bias']
            unet_dict['conv3.conv3_s1.weight'] = pretrained_dict['conv.conv3_s1.weight']
            unet_dict['conv3.conv3_s1.bias'] = pretrained_dict['conv.conv3_s1.bias']
            unet_dict['conv4.conv4_s1.weight'] = pretrained_dict['conv.conv4_s1.weight']
            unet_dict['conv4.conv4_s1.bias'] = pretrained_dict['conv.conv4_s1.bias']
            unet_dict['conv5.conv5_s1.weight'] = pretrained_dict['conv.conv5_s1.weight']
            unet_dict['conv5.conv5_s1.bias'] = pretrained_dict['conv.conv5_s1.bias']

            unet_dict['fc6.fc6_s1.weight'] = pretrained_dict['fc6.fc6_s1.weight'].view(1024, 256, 2, 2)
            unet_dict['fc6.fc6_s1.bias'] = pretrained_dict['fc6.fc6_s1.bias']

            model_dict.update(unet_dict)
            # 3. load the new state dict
            net.load_state_dict(unet_dict)
        elif "carson" in args.checkpoint:
                if "mnist" in args.checkpoint:
                    print("Loading mnist into SiamNet")
                elif "oct" in args.checkpoint:
                    print("Loading oct into SiamNet")
                pretrained_dict = torch.load(args.checkpoint)['model_state_dict']
                model_dict = net.state_dict()
                unet_dict = {}

                for k, v in model_dict.items():
                    unet_dict[k] = model_dict[k]

                #unet_dict['conv1.conv1_s1.weight'] = pretrained_dict['conv.conv1_s1.weight'][:, 0, :, :].unsqueeze(1)
                unet_dict['conv1.conv1_s1.weight'] = pretrained_dict['conv.conv1_s1.weight'].mean(1).unsqueeze(1)
                unet_dict['conv1.conv1_s1.bias'] = pretrained_dict['conv.conv1_s1.bias']
                unet_dict['conv1.batch1_s1.weight'] = pretrained_dict['conv.batch1_s1.weight']
                unet_dict['conv1.batch1_s1.bias'] = pretrained_dict['conv.batch1_s1.bias']
                unet_dict['conv1.batch1_s1.running_mean'] = pretrained_dict['conv.batch1_s1.running_mean']
                unet_dict['conv1.batch1_s1.running_var'] = pretrained_dict['conv.batch1_s1.running_var']
                unet_dict['conv1.batch1_s1.num_batches_tracked'] = pretrained_dict['conv.batch1_s1.num_batches_tracked']

                unet_dict['conv2.conv2_s1.weight'] = pretrained_dict['conv.conv2_s1.weight']
                unet_dict['conv2.conv2_s1.bias'] = pretrained_dict['conv.conv2_s1.bias']
                unet_dict['conv2.batch2_s1.weight'] = pretrained_dict['conv.batch2_s1.weight']
                unet_dict['conv2.batch2_s1.bias'] = pretrained_dict['conv.batch2_s1.bias']
                unet_dict['conv2.batch2_s1.running_mean'] = pretrained_dict['conv.batch2_s1.running_mean']
                unet_dict['conv2.batch2_s1.running_var'] = pretrained_dict['conv.batch2_s1.running_var']
                unet_dict['conv2.batch2_s1.num_batches_tracked'] = pretrained_dict['conv.batch2_s1.num_batches_tracked']

                unet_dict['conv3.conv3_s1.weight'] = pretrained_dict['conv.conv3_s1.weight']
                unet_dict['conv3.conv3_s1.bias'] = pretrained_dict['conv.conv3_s1.bias']
                unet_dict['conv3.batch3_s1.weight'] = pretrained_dict['conv.batch3_s1.weight']
                unet_dict['conv3.batch3_s1.bias'] = pretrained_dict['conv.batch3_s1.bias']
                unet_dict['conv3.batch3_s1.running_mean'] = pretrained_dict['conv.batch3_s1.running_mean']
                unet_dict['conv3.batch3_s1.running_var'] = pretrained_dict['conv.batch3_s1.running_var']
                unet_dict['conv3.batch3_s1.num_batches_tracked'] = pretrained_dict['conv.batch3_s1.num_batches_tracked']
               
                unet_dict['conv4.conv4_s1.weight'] = pretrained_dict['conv.conv4_s1.weight']
                unet_dict['conv4.conv4_s1.bias'] = pretrained_dict['conv.conv4_s1.bias']
                unet_dict['conv4.batch4_s1.weight'] = pretrained_dict['conv.batch4_s1.weight']
                unet_dict['conv4.batch4_s1.bias'] = pretrained_dict['conv.batch4_s1.bias']
                unet_dict['conv4.batch4_s1.running_mean'] = pretrained_dict['conv.batch4_s1.running_mean']
                unet_dict['conv4.batch4_s1.running_var'] = pretrained_dict['conv.batch4_s1.running_var']
                unet_dict['conv4.batch4_s1.num_batches_tracked'] = pretrained_dict['conv.batch4_s1.num_batches_tracked']

                unet_dict['conv5.conv5_s1.weight'] = pretrained_dict['conv.conv5_s1.weight']
                unet_dict['conv5.conv5_s1.bias'] = pretrained_dict['conv.conv5_s1.bias']
                unet_dict['conv5.batch5_s1.weight'] = pretrained_dict['conv.batch5_s1.weight']
                unet_dict['conv5.batch5_s1.bias'] = pretrained_dict['conv.batch5_s1.bias']
                unet_dict['conv5.batch5_s1.running_mean'] = pretrained_dict['conv.batch5_s1.running_mean']
                unet_dict['conv5.batch5_s1.running_var'] = pretrained_dict['conv.batch5_s1.running_var']
                unet_dict['conv5.batch5_s1.num_batches_tracked'] = pretrained_dict['conv.batch5_s1.num_batches_tracked']

                unet_dict['fc6.fc6_s1.weight'] = pretrained_dict['fc6.fc6_s1.weight'].view(1024, 256, 2, 2)
                #unet_dict['fc6.fc6_s1.weight'] = pretrained_dict['fc6.fc6_s1.weight']
                unet_dict['fc6.fc6_s1.bias'] = pretrained_dict['fc6.fc6_s1.bias']
                unet_dict['fc6.batch6_s1.weight'] = pretrained_dict['fc6.batch6_s1.weight']
                unet_dict['fc6.batch6_s1.bias'] = pretrained_dict['fc6.batch6_s1.bias']
                unet_dict['fc6.batch6_s1.running_mean'] = pretrained_dict['fc6.batch6_s1.running_mean']
                unet_dict['fc6.batch6_s1.running_var'] = pretrained_dict['fc6.batch6_s1.running_var']
                unet_dict['fc6.batch6_s1.num_batches_tracked'] = pretrained_dict['fc6.batch6_s1.num_batches_tracked']

                unet_dict['uconnect1.conv.weight'] = pretrained_dict['fc6b.conv6b_s1.weight']
                unet_dict['uconnect1.conv.bias'] = pretrained_dict['fc6b.conv6b_s1.bias']

                #unet_dict['fc6c.fc7.weight'] = pretrained_dict['fc6c.fc7.weight']
                #unet_dict['fc6c.fc7.bias'] = pretrained_dict['fc6c.fc7.bias']
                model_dict.update(unet_dict)
                # 3. load the new state dict
                net.load_state_dict(unet_dict)
        else:
            pretrained_dict = torch.load(args.checkpoint)['model_state_dict']
            model_dict = net.state_dict()
            print("loading checkpoint..............")
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k not in ["fc6c.fc7.weight","fc6c.fc7.bias", "fc7_new.fc7.weight", "fc7_new.fc7.bias", "classifier_new.fc8.weight", "classifier_new.fc8.bias"]}

            pretrained_dict['conv1.conv1_s1.weight'] = pretrained_dict['conv1.conv1_s1.weight'].mean(1).unsqueeze(1)
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)
            print("Checkpoint loaded...............")
    if args.adam:
        optimizer = torch.optim.Adam(net.parameters(), lr=hyperparams['lr'],
                                     weight_decay=hyperparams['weight_decay'])
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=hyperparams['lr'], momentum=hyperparams['momentum'],
                                    weight_decay=hyperparams['weight_decay'])



    training_set = KidneyDataset(train_X, train_y, train_cov)
    training_generator = DataLoader(training_set, **params)


    for epoch in range(max_epochs):
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
        for batch_idx, (data, target, cov) in enumerate(training_generator):
            optimizer.zero_grad()
            #net.train() # 20190619 
            output = net(data.to(device))
            target = Variable(target.type(torch.LongTensor), requires_grad=False).to(device)
            # print(output)
            # print(target)
            loss = F.cross_entropy(output, target)
            # loss = cross_entropy(output, target)
            # print(loss)
            loss_accum_train += loss.item() * len(target)

            loss.backward()

            accurate_labels_train += torch.sum(torch.argmax(output, dim=1) == target).cpu()
            optimizer.step()
            counter_train += len(target)

            output_softmax = softmax(output)
            # output_softmax = output
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
        #with torch.set_grad_enabled(False):

            for batch_idx, (data, target, cov) in enumerate(test_generator):
                net.zero_grad()
                #net.eval() # 20190619
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

                patient_ID_test.extend(cov)

        all_pred_prob_train = torch.cat(all_pred_prob_train)
        all_targets_train = torch.cat(all_targets_train)
        all_pred_label_train = torch.cat(all_pred_label_train)

        # patient_ID_train = torch.cat(patient_ID_train)

        assert len(all_targets_train) == len(training_set)
        assert len(all_pred_prob_train) == len(training_set)
        assert len(all_pred_label_train) == len(training_set)
        assert len(patient_ID_train) == len(training_set)

        results_train = process_results.get_metrics(y_score=all_pred_prob_train.cpu().detach().numpy(),
                                                    y_true=all_targets_train.cpu().detach().numpy(),
                                                    y_pred=all_pred_label_train.cpu().detach().numpy())

        print('TrainEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
              'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(epoch,
                                                                     int(accurate_labels_train) / counter_train,
                                                                     loss_accum_train / counter_train,
                                                                     results_train['auc'],
                                                                     results_train['auprc'], results_train['tn'],
                                                                     results_train['fp'], results_train['fn'],
                                                                     results_train['tp']))


        all_pred_prob_test = torch.cat(all_pred_prob_test)
        all_targets_test = torch.cat(all_targets_test)
        all_pred_label_test = torch.cat(all_pred_label_test)

        # patient_ID_test = torch.cat(patient_ID_test)

        assert len(all_targets_test) == len(test_y)
        assert len(all_pred_label_test) == len(test_y)
        assert len(all_pred_prob_test) == len(test_y)
        assert len(patient_ID_test) == len(test_y)

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
        path_to_checkpoint = args.dir + "/checkpoint_" + str(epoch) + '.pth'
        torch.save(checkpoint, path_to_checkpoint)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=50, type=int, help="Number of epochs")
    parser.add_argument('--batch_size', default=128, type=int, help="Batch size")
    parser.add_argument('--lr', default=0.001, type=float, help="Learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, help="Momentum")
    parser.add_argument('--adam', action="store_true", help="Use Adam optimizer instead of SGD")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay")
    parser.add_argument("--num_workers", default=1, type=int, help="Number of CPU workers")
    parser.add_argument("--dir", default="./", help="Directory to save model checkpoints to")
    parser.add_argument("--contrast", default=1, type=int, help="Image contrast to train on")
    parser.add_argument("--view", default="siamese", help="siamese, sag, trans")
    parser.add_argument("--checkpoint", default="", help="Path to load pretrained model checkpoint from")
    parser.add_argument("--split", default=0.7, type=float, help="proportion of dataset to use as training")
    parser.add_argument("--bottom_cut", default=0.0, type=float, help="proportion of dataset to cut from bottom")
    parser.add_argument("--etiology", default="B", help="O (obstruction), R (reflux), B (both)")
    parser.add_argument('--unet', action="store_true", help="UNet architecthure")
    parser.add_argument("--sc", default=5, type=int, help="number of skip connections for unet (0, 1, 2, 3, 4, or 5)")
    parser.add_argument("--init", default="none")
    parser.add_argument("--hydro_only", action="store_true")
    parser.add_argument("--crop", default=0, type=int, help="Crop setting (0=big, 1=tight)")
    #parser.add_argument("--datafile", default="../../0.Preprocess/preprocessed_images_20190601.pickle",
     #                   help="File containing pandas dataframe with images stored as numpy array")
    parser.add_argument("--datafile", default="../../0.Preprocess/preprocessed_images_20190617.pickle")
    parser.add_argument("--output_dim", default=256, type=int, help="output dim for last linear layer")
    parser.add_argument("--gender", default=None, type=str, help="choose from 'male' and 'female'")
    args = parser.parse_args()

    print("ARGS" + '\t' + str(args))

    max_epochs = args.epochs
    train_X, train_y, train_cov, test_X, test_y, test_cov = load_dataset.load_dataset(views_to_get="siamese",
                                                                                      sort_by_date=True,
                                                                                      pickle_file=args.datafile,
                                                                                      contrast=args.contrast,
                                                                                      split=args.split,
                                                                                      get_cov=True,
                                                                                      bottom_cut=args.bottom_cut,
                                                                                      etiology=args.etiology,
                                                                                      crop=args.crop, hydro_only=args.hydro_only, gender=args.gender)

    if args.view == "sag" or args.view == "trans":
        train_X_single = []
        test_X_single = []

        for item in train_X:
            if args.view == "sag":
                train_X_single.append(item[0])
            elif args.view == "trans":
                train_X_single.append(item[1])
        for item in test_X:
            if args.view == "sag":
                test_X_single.append(item[0])
            elif args.view == "trans":
                test_X_single.append(item[1])

        train_X = train_X_single
        test_X = test_X_single

    print(len(train_X), len(train_y), len(train_cov), len(test_X), len(test_y), len(test_cov))

    train_X = np.array(train_X)
    train_y = np.array(train_y)
    test_X = np.array(test_X)
    test_y = np.array(test_y)

    train(args, train_X, train_y, train_cov, test_X, test_y, test_cov, max_epochs)

if __name__ == '__main__':
    main()
