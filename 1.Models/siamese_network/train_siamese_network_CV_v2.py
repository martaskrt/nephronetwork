from sklearn.utils import shuffle
# from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
# import codecs
# import errno
# import matplotlib.pyplot as plt
import numpy as np
import os
# from PIL import Image
# import random
import torch
from torch import nn
#from torch import optim
import torch.nn.functional as F
import torchvision.datasets.mnist
from torchvision import transforms
# from tqdm import tqdm
import importlib.machinery
from torch.utils.data import Dataset, DataLoader
# from torchsummary import summary
import argparse
from torch.autograd import Variable
from sklearn.utils import class_weight

# from FraternalSiameseNetwork import SiamNet
load_dataset = importlib.machinery.SourceFileLoader('load_dataset','../../0.Preprocess/load_dataset.py').load_module()
process_results = importlib.machinery.SourceFileLoader('process_results','../../2.Results/process_results.py').load_module()

SEED = 42

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

    ##
    ## NN
    ##

class SiamNet(nn.Module):
    def __init__(self, classes=2, num_inputs=2, dropout_rate=0.5, output_dim=128):
        super(SiamNet, self).__init__()

        self.output_dim = output_dim
        print("LL DIM: " + str(self.output_dim))
        self.num_inputs = num_inputs

        self.conv = nn.Sequential()
        self.conv.add_module('conv1_s1', nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=0))
        self.conv.add_module('batch1_s1', nn.BatchNorm2d(96))
        self.conv.add_module('relu1_s1', nn.ReLU(inplace=True))
        self.conv.add_module('pool1_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv.add_module('lrn1_s1', LRN(local_size=5, alpha=0.0001, beta=0.75))

        self.conv.add_module('conv2_s1', nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2))
        self.conv.add_module('batch2_s1', nn.BatchNorm2d(256))
        self.conv.add_module('relu2_s1', nn.ReLU(inplace=True))
        self.conv.add_module('pool2_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv.add_module('lrn2_s1', LRN(local_size=5, alpha=0.0001, beta=0.75))

        self.conv.add_module('conv3_s1', nn.Conv2d(256, 384, kernel_size=3, padding=1))
        self.conv.add_module('batch3_s1', nn.BatchNorm2d(384))
        self.conv.add_module('relu3_s1', nn.ReLU(inplace=True))

        self.conv.add_module('conv4_s1', nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2))
        self.conv.add_module('batch4_s1', nn.BatchNorm2d(384))
        self.conv.add_module('relu4_s1', nn.ReLU(inplace=True))

        self.conv.add_module('conv5_s1', nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2))
        self.conv.add_module('batch5_s1', nn.BatchNorm2d(256))
        self.conv.add_module('relu5_s1', nn.ReLU(inplace=True))
        self.conv.add_module('pool5_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        # *************************** changed layers *********************** #
        self.fc6 = nn.Sequential()
        # self.fc6.add_module('fc6_s1', nn.Conv2d(256, 1024, kernel_size=3, stride=1, padding=1))
        self.fc6.add_module('fc6_s1', nn.Conv2d(256, 1024, kernel_size=2, stride=1, padding=1))
        # self.fc6.add_module('fc6_s1', nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=1))
        self.fc6.add_module('batch6_s1', nn.BatchNorm2d(1024))
        self.fc6.add_module('relu6_s1', nn.ReLU(inplace=True))

        self.fc6b = nn.Sequential()
        self.fc6b.add_module('conv6b_s1', nn.Conv2d(1024, 256, kernel_size=3, stride=2))
        self.fc6b.add_module('batch6b_s1', nn.BatchNorm2d(256))
        self.fc6b.add_module('relu6_s1', nn.ReLU(inplace=True))
        self.fc6b.add_module('pool6b_s1', nn.MaxPool2d(kernel_size=3, stride=2))

        self.fc6c = nn.Sequential()
        # self.fc6c.add_module('fc7', nn.Linear(256*2*2, 512))
        self.fc6c.add_module('fc7', nn.Linear(256 * 3 * 3, 512))
        self.fc6c.add_module('relu7', nn.ReLU(inplace=True))
        # self.fc6c.add_module('drop7', nn.Dropout(p=dropout_rate))

        self.fc7_new = nn.Sequential()
        self.fc7_new.add_module('fc7', nn.Linear(self.num_inputs * 512, self.output_dim))
        self.fc7_new.add_module('relu7', nn.ReLU(inplace=True))
        # self.fc7_new.add_module('drop7', nn.Dropout(p=dropout_rate))

        self.classifier_new = nn.Sequential()
        self.classifier_new.add_module('fc8', nn.Linear(self.output_dim, classes))

        # self.fc7_new = nn.Sequential()
        # self.fc7_new.add_module('fc7', nn.Linear(self.num_inputs * 512, 512))
        # self.fc7_new.add_module('relu7', nn.ReLU(inplace=True))
        # self.fc7_new.add_module('fc7_2', nn.Linear(512, 128))
        # self.fc7_new.add_module('relu7_2', nn.ReLU(inplace=True))

        # self.classifier_new = nn.Sequential()
        # self.classifier_new.add_module('fc8', nn.Linear(128, classes))

    def load(self, checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)
        pretrained_dict = {k: v for k, v in list(pretrained_dict.items()) if k in model_dict and 'fc8' not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print([k for k, v in list(pretrained_dict.items())])

    def save(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

    def forward(self, x):
        # x = x.unsqueeze(0)
        if self.num_inputs == 1:
            x = x.unsqueeze(1)
        #   B, C, H = x.size()
        # else:
        B, T, C, H = x.size()
        x = x.transpose(0, 1)
        x_list = []
        for i in range(self.num_inputs):
            if self.num_inputs == 1:
                curr_x = torch.unsqueeze(x[i], 1)
            else:
                curr_x = torch.unsqueeze(x[i], 1)
            # if self.num_inputs == 1:
            #   curr_x = curr_x.expand(-1, 3, -1)
            # else:
            curr_x = curr_x.expand(-1, 3, -1, -1)
            if torch.cuda.is_available():
                input = torch.cuda.FloatTensor(curr_x.to(device))
            else:
                input = torch.FloatTensor(curr_x.to(device))
            z = self.conv(input)
            z = self.fc6(z)
            z = self.fc6b(z)
            z = z.view([B, 1, -1])
            z = self.fc6c(z)
            z = z.view([B, 1, -1])
            x_list.append(z)

        x = torch.cat(x_list, 1)
        # x = torch.sum(x, 1)
        x = self.fc7_new(x.view(B, -1))
        pred = self.classifier_new(x)

        return pred


class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                                        stride=1, padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                                        stride=1, padding=int((local_size - 1.0) / 2))
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
softmax = torch.nn.Softmax(dim=1)

    ##
    ## DATA LOADER
    ##

class KidneyDataset(torch.utils.data.Dataset):

    def __init__(self, X, y, cov):
        self.X = torch.from_numpy(X).float()
        self.y = y
        self.cov = cov

    def __getitem__(self, index):
        imgs, target, cov = self.X[index], self.y[index], self.cov[index]
        # cov = []
        # for i in range(len(self.cov)): # 0: study_id, 1: age_at_baseline, 2: gender (0 if male), 3: view (0 if saggital)...skip), 4: sample_num, 5: date_of_US_1
        #     if i == 2:
        #         if self.cov[i][index] == 0:
        #             cov.append("M")
        #         elif self.cov[i][index] == 0:
        #             cov.append("F")
        #     elif i == 3:
        #         continue
        #     elif i == 4:
        #         cov.append(int(self.cov[i][index]))
        #     else:
        #         cov.append(self.cov[i][index])
        #
        # cov_id = ""
        # for item in cov:
        #     cov_id += str(item) + "_"
        # cov_id = cov_id[:-1]

        #to_pil = transforms.ToPILImage()
        #to_tensor = transforms.ToTensor()
        #for n in range(2):
         #   temp_img = imgs[n]
          #  m, s = temp_img.view(1, -1).mean(dim=1).numpy(), temp_img.view(1, -1).std(dim=1).numpy()
           # s[s == 0] = 1
           # norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
           # temp_img = norm(to_tensor(to_pil(temp_img)))
           # imgs[n] = temp_img
        
        return imgs, target, cov

    def __len__(self):
        return len(self.X)

    ##
    ## INITIALIZATION AND TRAINING
    ##

def init_weights(m):
    #if type(m) == nn.Linear:
    print(m.weight)
    torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #torch.nn.init.xavier_uniform(m.weight)
    m.bias.data.fill_(0.01)
    print(m.weight)

def train(args, train_X, train_y, train_cov, test_X, test_y, test_cov, max_epochs):
    # if args.unet:
    #     print("importing UNET")
    #
    #     if args.sc == 5:
    #         from SiameseNetworkUNet import SiamNet
    #     elif args.sc == 4:
    #         from SiameseNetworkUNet_sc4_nosc3 import SiamNet
    #     elif args.sc == 3:
    #         from SiameseNetworkUNet_sc3 import SiamNet
    #     elif args.sc == 2:
    #         from SiameseNetworkUNet_sc2 import SiamNet
    #     elif args.sc == 1:
    #         from SiameseNetworkUNet_sc1 import SiamNet
    #     elif args.sc == 0:
    #         if args.init == "none":
    #             #from FraternalSiameseNetwork_20190619 import SiamNet
    #             from SiameseNetworkUNet_upconv_1c_1ch import SiamNet
    #         elif args.init == "fanin":
    #             from SiameseNetworkUNet_upconv_1c_1ch_fanin import SiamNet
    #         elif args.init == "fanout":
    #             from SiameseNetworkUNet_upconv_1c_1ch_fanout import SiamNet
      
    # else:
    # print("importing SIAMNET")
    # from SiameseNetwork import SiamNet

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

    fold = 1
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_y = np.array(train_y)
    train_cov = np.array(train_cov)

    train_X, train_y, train_cov = shuffle(train_X, train_y, train_cov, random_state=42)
    # for i in range(len(train_cov)):
    #     train_cov[i] = shuffle(train_cov[i], random_state=42)
    #class_weights = class_weight.compute_class_weight('balanced',
     #                                            np.unique(train_y),
      #                                          train_y)
    #print(class_weights)
    for train_index, test_index in skf.split(train_X, train_y):
      #  class_weights=torch.tensor([1/num_0, 1/num_1]).to(device)
        #class_weights=torch.tensor([0.5, 2.0]).to(device)
        #cw=torch.from_numpy(class_weights).float().to(device)
        #print("CLASS WEIGHTS: " + str(cw))
        #cross_entropy = nn.CrossEntropyLoss(weight=cw)
        #cross_entropy = nn.CrossEntropyLoss()
        #if fold == 1 or fold == 2:
         #   fold += 1
          #  continue
        #if fold != 5:
         #   fold += 1
          #  continue
        jigsaw = False
        if "jigsaw" in args.dir and "unet" in args.dir:
            jigsaw = True
        if args.view != "siamese":
            net = SiamNet(num_inputs=1, output_dim=args.output_dim, jigsaw=jigsaw).to(device)
        else:
            net = SiamNet(output_dim=args.output_dim, jigsaw=jigsaw).to(device)
        net.zero_grad()
        #net.apply(init_weights)
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
                #unet_dict['fc6.fc6_s1.weight'] = pretrained_dict['fc6.fc6_s1.weight']
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
                #pretrained_dict = torch.load(args.checkpoint)
                pretrained_dict = torch.load(args.checkpoint)['model_state_dict']
                model_dict = net.state_dict()
                print("loading checkpoint..............")
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k not in ["fc6c.fc7.weight","fc6c.fc7.bias", "fc7_new.fc7.weight", "fc7_new.fc7.bias", "classifier_new.fc8.weight", "classifier_new.fc8.bias"]}
           
                pretrained_dict['conv1.conv1_s1.weight'] = pretrained_dict['conv1.conv1_s1.weight'].mean(1).unsqueeze(1)            
                #pretrained_dict['fc6.fc6_s1.weight'] = pretrained_dict['fc6.fc6_s1.weight'].view(1024, 256, 2, 2)
                #for k, v in model_dict.items():
                 #   if k not in pretrained_dict:
                  #      pretrained_dict[k] = model_dict[k]
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
                # 3. load the new state dict
                net.load_state_dict(model_dict)
                print("Checkpoint loaded...............")

        if args.adam:
            optimizer = torch.optim.Adam(net.parameters(), lr=hyperparams['lr'],
                                         weight_decay=hyperparams['weight_decay'])
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=hyperparams['lr'], momentum=hyperparams['momentum'],
                                        weight_decay=hyperparams['weight_decay'])


        train_X_CV = train_X[train_index]
        train_y_CV = train_y[train_index]
        train_cov_CV = train_cov[train_index]
        # for i in range(len(train_cov)):
        #     print(i)
        #     train_cov_CV.append([])
        #     print(train_cov[i])
        #     train_cov_CV.append(train_cov[i][train_index])

        val_X_CV = train_X[test_index]
        val_y_CV = train_y[test_index]
        val_cov_CV = train_cov[test_index]
        # for i in range(len(train_cov)):
        #     val_cov_CV.append([])
        #     val_cov_CV[i] = train_cov[i][test_index]

        training_set = KidneyDataset(train_X_CV, train_y_CV, train_cov_CV)
        training_generator = DataLoader(training_set, **params)

        validation_set = KidneyDataset(val_X_CV, val_y_CV, val_cov_CV)
        validation_generator = DataLoader(validation_set, **params)

        ## run all epochs on given split of the data
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

            patient_ID_test = []
            patient_ID_train = []
            patient_ID_val = []

            counter_train = 0
            counter_val = 0
            counter_test = 0
            net.train()

            ## Run training set
            for batch_idx, (data, target, cov) in enumerate(training_generator):
                optimizer.zero_grad()
                #net.train() # 20190619
                output = net(data.to(device))
                target = Variable(target.type(torch.LongTensor), requires_grad=False).to(device)
                #print(output)
                #print(target)
                #print(output.shape, target.shape)
                if len(output.shape) == 1:
                    output = output.unsqueeze(0)
                loss = F.cross_entropy(output, target)
                #loss = cross_entropy(output, target)
                #print(loss)
                loss_accum_train += loss.item() * len(target)

                loss.backward()

                accurate_labels_train += torch.sum(torch.argmax(output, dim=1) == target).cpu()
                optimizer.step()
                counter_train += len(target)

                output_softmax = softmax(output)
                #output_softmax = output
                pred_prob = output_softmax[:, 1]
                pred_label = torch.argmax(output_softmax, dim=1)
                #print(pred_prob)
                #print(pred_label)
                assert len(pred_prob) == len(target)
                assert len(pred_label) == len(target)


                all_pred_prob_train.append(pred_prob)
                all_targets_train.append(target)
                all_pred_label_train.append(pred_label)

                patient_ID_train.extend(cov)
            net.eval()
            ## Run val set
            with torch.no_grad():
            #with torch.set_grad_enabled(False):
                for batch_idx, (data, target, cov) in enumerate(validation_generator):
                    net.zero_grad()
                    #net.eval() # 20190619
                    optimizer.zero_grad()
                    output = net(data)
                    target = target.type(torch.LongTensor).to(device)

                    loss = F.cross_entropy(output, target)
                    #loss = cross_entropy(output, target)
                    loss_accum_val += loss.item() * len(target)
                    counter_val += len(target)
                    #output_softmax = output
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

                    patient_ID_val.extend(cov)
            net.eval()
            ## Run test set
            with torch.no_grad():
            #with torch.set_grad_enabled(False):
                for batch_idx, (data, target, cov) in enumerate(test_generator):
                    net.zero_grad()
                    net.eval() # 20190619
                    optimizer.zero_grad()
                    output = net(data)
                    target = target.type(torch.LongTensor).to(device)

                    loss = F.cross_entropy(output, target)
                    #loss = cross_entropy(output, target)
                    loss_accum_test += loss.item() * len(target)
                    counter_test += len(target)
                    output_softmax = softmax(output)
                    #output_softmax = output
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

            print('Fold\t{}\tTrainEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch, int(accurate_labels_train)/counter_train,
                                                                        loss_accum_train/counter_train, results_train['auc'],
                                                                        results_train['auprc'], results_train['tn'],
                                                                        results_train['fp'], results_train['fn'],
                                                                        results_train['tp']))
            all_pred_prob_val = torch.cat(all_pred_prob_val)
            all_targets_val = torch.cat(all_targets_val)
            all_pred_label_val = torch.cat(all_pred_label_val)

            #patient_ID_val = torch.cat(patient_ID_val)

            assert len(all_targets_val) == len(validation_set)
            assert len(all_pred_prob_val) == len(validation_set)
            assert len(all_pred_label_val) == len(validation_set)
            assert len(patient_ID_val) == len(validation_set)

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

            # patient_ID_test = torch.cat(patient_ID_test)

            assert len(all_targets_test) == len(test_y)
            assert len(all_pred_label_test) == len(test_y)
            assert len(all_pred_prob_test) == len(test_y)
            assert len(patient_ID_test) == len(test_y)

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
                          'args': args,
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
                          'all_targets_test': all_targets_test,
                          'patient_ID_train': patient_ID_train,
                          'patient_ID_val': patient_ID_val,
                          'patient_ID_test': patient_ID_test}

            if not os.path.isdir(args.dir):
                os.makedirs(args.dir)
            #if not os.path.isdir(args.dir + "/" + str(fold)):
                #os.makedirs(args.dir + "/" + str(fold))
            path_to_checkpoint = args.dir + "/" + str(fold) + "_checkpoint_" + str(epoch) + '.pth'
            torch.save(checkpoint, path_to_checkpoint)
            
        fold += 1

    ##
    ## MAIN FUNCTION
    ##

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=50, type=int, help="Number of epochs")
    parser.add_argument('--batch_size', default=64, type=int, help="Batch size")
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
    parser.add_argument("--crop", default=0, type=int, help="Crop setting (0=big, 1=tight)")
    parser.add_argument("--sc", default=5, type=int, help="number of skip connections for unet (0, 1, 2, 3, 4, or 5)")
    parser.add_argument("--init", default="none")
    parser.add_argument("--hydro_only", action="store_true")
    parser.add_argument("--output_dim", default=256, type=int, help="output dim for last linear layer")
    parser.add_argument("--datafile", default="../../0.Preprocess/preprocessed_images_20190617.pickle", help="File containing pandas dataframe with images stored as numpy array")
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
        train_X_single=[]
        test_X_single=[]

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

        train_X=train_X_single
        test_X=test_X_single
        train_X=np.array(train_X_single)
        test_X=np.array(test_X_single)

        
    print(len(train_X), len(train_y), len(train_cov), len(test_X), len(test_y), len(test_cov))        
    train(args, train_X, train_y, train_cov, test_X, test_y, test_cov, max_epochs)


if __name__ == '__main__':
    main()
