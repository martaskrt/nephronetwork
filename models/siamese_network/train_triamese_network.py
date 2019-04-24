from sklearn.utils import shuffle
# from sklearn.model_selection import KFold
import codecs
import errno
import numpy as np
import os
#from PIL import Image
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
# =============================================================================
# import torchvision.datasets.mnist
# =============================================================================
from torchvision import transforms
# from tqdm import tqdm
import importlib.machinery
from torch.utils.data import Dataset, DataLoader
import argparse
from torch.autograd import Variable
import pandas as pd
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: " + str(device))
softmax = torch.nn.Softmax(dim=1)

# FUNCTION to load all us triad seq paths
def get_pickle_files(file_dir):
    pickle_files = []
    for subdir, dirs, files in os.walk(file_dir):
        for file in files:
            if file.lower()[-7:] == ".pickle":
                pickle_files.append(os.path.join(subdir, file))
    return pickle_files    

# FUNCTION loads all triad seqs, converts them to a torch tensor 
def get_training_test_data(opt,split_batches = True,test_share = 0.1):
    pickle_files = get_pickle_files(opt.train_dir)  
    
    training_stack = []
    label_stack = []
    for file in pickle_files:
        in_df = pd.read_pickle(file)
        training_stack.extend(in_df.triad)
        label_stack.extend(in_df.label)
    
    ts_np = np.array(training_stack)
    tl_np = np.array(label_stack)
    
    n_samples = ts_np.shape[0]
    n_test_samples = round(n_samples*0.1)    
    
    perm = np.random.permutation(n_samples)
    
    ts_perm = ts_np[perm,:,:,:]
    tl_perm = tl_np[perm]
    
    ts_test = ts_perm[:n_test_samples,:,:,:]
    tl_test = ts_perm[0:n_test_samples]

    ts_train = ts_perm[(n_test_samples+1):,:,:,:]    
    tl_train = tl_perm[(n_test_samples+1):]    
    
    return ts_train, tl_train, ts_test, tl_test


def load_sample(infile):
    in_df = pd.read_pickle(infile)
    images  = in_df.triad
    label = in_df.label
    return images,label

## Create torch dataset from np dataset defined in previous function 
class KidneyDataset(torch.utils.data.Dataset):

    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = y

    def __getitem__(self, index):
        imgs, target = self.X[index], self.y[index]
        return imgs, target

    def __len__(self):
        return len(self.X)

class TriadDataset(torch.utils.data.Dataset):
    def __init__(self,pickle_dir):
        self.filelist = get_pickle_files(pickle_dir)

    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self,index):
        imgs,target = load_sample(self.filelist[index])
        return (imgs,target)

def train(args, training_generator, validation_generator, max_epochs,SiamNet,process_results):
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
    optimizer = torch.optim.SGD(net.parameters(), lr=hyperparams['lr'], momentum=hyperparams['momentum'],
                                weight_decay=hyperparams['weight_decay'])

    # optimizer = torch.optim.Adam(net.parameters(), lr=hyperparams['lr'], weight_decay=hyperparams['weight_decay'])


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

        counter_train = 0
        counter_val = 0
# =============================================================================
#         for train_index, test_index in skf.split(train_X, train_y):
#             train_X_CV = train_X[train_index]
#             train_y_CV = train_y[train_index]
#             val_X_CV = train_X[test_index]
#             val_y_CV = train_y[test_index]
# 
#             training_set = KidneyDataset(train_X_CV, train_y_CV)
#             training_generator = DataLoader(training_set, **params)
# 
#             validation_set = KidneyDataset(val_X_CV, val_y_CV)
#             validation_generator = DataLoader(validation_set, **params)
# 
# =============================================================================
        for batch_idx, (data, target) in enumerate(training_generator):
            data = torch.stack(data)
            data = torch.transpose(data,0,1).float()  
            
            optimizer.zero_grad()
            output = net(data)
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
                data = torch.stack(data)
                data = torch.transpose(data,0,1).float()  
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

        all_pred_prob_val = torch.cat(all_pred_prob_val)
        all_targets_val = torch.cat(all_targets_val)
        all_pred_label_val = torch.cat(all_pred_label_val)

        # print(all_pred_prob_train)
        # print(all_targets_train)
        # print(all_pred_label_train)

        # print(all_pred_prob_val)
        # print(all_targets_val)
        # print(all_pred_label_val)

        #assert len(all_targets_val) == len(train_y)
        #assert len(all_pred_prob_val) == len(train_y)
        #assert len(all_pred_label_val) == len(train_y)
        results_train = process_results.get_metrics(y_score=all_pred_prob_train.cpu().detach().numpy(),
                                                  y_true=all_targets_train.cpu().detach().numpy(),
                                                  y_pred=all_pred_label_train.cpu().detach().numpy())
        print('TrainEpoch\t{}\tACC\t{:.0f}%\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
              'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(epoch, 100. * accurate_labels_train / counter_train,
                                                                     loss_accum_train / counter_train, results_train['auc'],
                                                                     results_train['auprc'], results_train['tn'],
                                                                     results_train['fp'], results_train['fn'],
                                                                     results_train['tp']))
    
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
        if ((epoch+1) % 10) == 0:
            checkpoint = {'epoch': epoch,
                          'loss': loss,
                          'hyperparams': hyperparams,
                          'model_state_dict': net.state_dict(),
                          'optimizer': optimizer.state_dict()}
            #if not os.path.isdir(args.dir):
            #    os.makedirs(args.dir)
            # if epoch > 50:
            #     path_to_prev_checkpoint = args.git_dir + '/models/siamese_networks/' + "triamese_checkpoint_" + str(epoch-50) + '.pth'
            #     os.remove(path_to_prev_checkpoint)
            path_to_checkpoint = args.git_dir + '/models/siamese_network/' + "triamese25k_checkpoint_" + str(epoch) + '.pth'
            torch.save(checkpoint, path_to_checkpoint)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='/storage/ind_train_us_seq/',help="Number of epochs")
    parser.add_argument('--valid_dir', default='/storage/ind_val_us_seq/', help="Number of epochs")
    parser.add_argument('--epochs', default=100, type=int, help="Number of epochs")
    parser.add_argument('--batch_size', default=256, type=int, help="Batch size")
    parser.add_argument('--lr', default=0.01, type=float, help="Learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, help="Momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay")
    parser.add_argument("--num_workers", default=1, type=int, help="Number of CPU workers")
    parser.add_argument("--dir", default="./", help="Directory to save model checkpoints to")
    parser.add_argument("--contrast", default=0, type=int, help="Image contrast to train on")
    parser.add_argument("--checkpoint", default="", help="Path to load pretrained model checkpoint from")
    parser.add_argument("--git_dir", default="/home/lauren/nephronetwork", help="Path to github repo with necessary modules to load")
    parser.add_argument("--siam_unet", default=False, help="Use Regular Siam Net or Siame UNet")
    
    args = parser.parse_args()

    process_results = importlib.machinery.SourceFileLoader('process_results',os.path.join(args.git_dir,'models/process_results.py')).load_module()
    
    
    print(args.git_dir +'/models/siamese_network/')
    print(args.train_dir)
    print(args.valid_dir)

    os.chdir(args.git_dir +'/models/siamese_network/')
    if args.siam_unet:
        from SiameseNetworkUNet import SiamNet
    else:
        from SiameseNetwork import SiamNet
    
    max_epochs = args.epochs
    
    training_triad_data = TriadDataset(args.train_dir)    
    triad_training_dataloader = torch.utils.data.DataLoader(dataset=training_triad_data,
                                                   batch_size=args.batch_size,
                                                   shuffle=True)
    
    valid_triad_data = TriadDataset(args.valid_dir)
    triad_valid_dataloader = torch.utils.data.DataLoader(dataset=valid_triad_data,
                                                   batch_size=args.batch_size,
                                                   shuffle=True)
 
    train(args,triad_training_dataloader, triad_valid_dataloader,max_epochs,SiamNet,process_results)

if __name__ == '__main__':
    main()


###
### DEBUGGING MAIN FUCTION 
###

# =============================================================================
# class options(object):
#     pass
# args = options()
# 
# args.train_dir = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Image-analysis/test-dir/test/'
# args.valid_dir = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Image-analysis/test-dir/test/'
# args.epochs = 4
# args.batch_size = 32
# args.lr = 0.001
# args.momentum = 0.9
# args.weight_decay = 5e-4
# args.num_workers = 1
# args.dir = "./"
# args.contrast = 0
# args.checkpoint = ""
# args.git_dir = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/github/nephronetwork/'
# args.siam_unet = False
# 
# process_results = importlib.machinery.SourceFileLoader('process_results',os.path.join(args.git_dir,'models/process_results.py')).load_module()
# 
# os.chdir(args.git_dir +'/models/siamese_network/')
# if args.siam_unet:
#     from SiameseNetworkUNet import SiamNet
# else:
#     from SiameseNetwork import SiamNet
# 
# max_epochs = args.epochs
# 
# training_triad_data = TriadDataset(args.train_dir)
# 
# triad_training_dataloader = torch.utils.data.DataLoader(dataset=training_triad_data,
#                                                batch_size=args.batch_size,
#                                                shuffle=True)
# 
# valid_triad_data = TriadDataset(args.valid_dir)
# triad_valid_dataloader = torch.utils.data.DataLoader(dataset=valid_triad_data,
#                                                batch_size=args.batch_size,
#                                                shuffle=True)
# 
# 
# 
# for batch_idx, (data,labels) in enumerate(triad_training_dataloader):
#     
#     data = torch.stack(data)
#     data = torch.transpose(data,0,1).float()  
#     print(data.type())
# 
# 
# ## batchdata = output[1][0]
#     
# 
# 
# training_triad_data.__len__()
# 
# for (imgs,target) in enumerate(training_triad_data):
#     print((imgs,target))
# 
# 
# batch = random.sample(range(training_triad_data.__len__()),32)
# 
# my_batch = triad_data.__getitem__(batch)
# 
# train(args,train_X,train_y,test_X,test_y,max_epochs)
# 
# 
# 
# =============================================================================
