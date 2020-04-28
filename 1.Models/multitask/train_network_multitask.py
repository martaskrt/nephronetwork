from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.datasets.mnist
from torchvision import transforms
import importlib.machinery
from torch.utils.data import Dataset, DataLoader
import argparse
from torch.autograd import Variable
from sklearn.utils import class_weight

#process_results = importlib.machinery.SourceFileLoader('process_results','../../2.Results/process_results.py').load_module()
import process_results

SEED = 42

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
softmax = torch.nn.Softmax(dim=1)

class KidneyDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.X = []
        self.y = []
        self.cov = []
        
        for key in dataset:
            self.cov.append(key)
            x = np.zeros((8, 256, 256))
            y = [-1, -1, -1]
            if 's' in dataset[key]:
                x[:2,:,:], y[0] = dataset[key]['s']
            if 'r' in dataset[key]:
                x[2:4,:,:], y[1] = dataset[key]['r']
            if 'f' in dataset[key]:
                x[4:,:,:], y[2] = dataset[key]['f']
            self.X.append(x)
            self.y.append(y)
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.cov = np.array(self.cov)

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.cov[index]

    def __len__(self):
        return len(self.X)

def get_proba(x):
    x_softmax = softmax(x)
    pred_prob = x_softmax[:, 1]
    pred_label = torch.argmax(x_softmax, dim=1)
    
    return pred_prob, pred_label

def get_data_stats(y, label_type):
    print("{}:::{} class_0/{} class_1/{} total".format(label_type, len(y)-sum(y), sum(y), len(y)))

def train(args, dataset_train, dataset_test,  max_epochs):

    from CNN_multitask import CNN

    hyperparams = {'lr': args.lr,
                   'momentum': args.momentum,
                   'weight_decay': args.weight_decay,
                   'train/test_split': args.split 
                  }
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.num_workers}

    test_set = KidneyDataset(dataset_test)
    test_generator = DataLoader(test_set, **params)

    fold=1
   # n_splits = 5
   # skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
   # train_y = np.array(train_y)
   # train_cov = np.array(train_cov)
   # train_label_type = np.array(train_label_type)

    #train_X, train_y, train_cov, train_label_type = shuffle(train_X, train_y, train_cov, train_label_type, random_state=42)
    #for train_index, test_index in skf.split(train_X, train_y):
    for _ in range(1): 
        if args.view != "siamese":
            net = CNN(num_views=1, three_channel=False).to(device)
        else:
            net = CNN(num_views=2, three_channel=False).to(device)

        net.zero_grad()
        optimizer = torch.optim.SGD(net.parameters(), lr=hyperparams['lr'], momentum=hyperparams['momentum'],
                                        weight_decay=hyperparams['weight_decay'])


        #train_X_CV = train_X[train_index]
        #train_y_CV = train_y[train_index]
        #train_cov_CV = train_cov[train_index]
        #train_label_type_CV = train_label_type[train_index]

        #val_X_CV = train_X[test_index]
        #val_y_CV = train_y[test_index]
        #val_cov_CV = train_cov[test_index]
        #val_label_type_CV = train_label_type[test_index]

        training_set = KidneyDataset(dataset_train)
        training_generator = DataLoader(training_set, **params)
       
        #training_set = KidneyDataset(train_X_CV, train_y_CV, train_cov_CV, train_label_type_CV)
        #training_generator = DataLoader(training_set, **params)

        #validation_set = KidneyDataset(val_X_CV, val_y_CV, val_cov_CV, val_label_type_CV)
        #validation_generator = DataLoader(validation_set, **params)

        for epoch in range(max_epochs):
            loss_accum_train = 0
            loss_accum_test = 0

            surgery_targets_train = []
            surgery_pred_prob_train = []
            surgery_pred_label_train = []
            reflux_targets_train = []
            reflux_pred_prob_train = []
            reflux_pred_label_train = []
            func_targets_train = []
            func_pred_prob_train = []
            func_pred_label_train = []


            surgery_patient_ID_train = []
            reflux_patient_ID_train = []
            func_patient_ID_train = []

            surgery_targets_test = []
            surgery_pred_prob_test = []
            surgery_pred_label_test = []
            reflux_targets_test = []
            reflux_pred_prob_test = []
            reflux_pred_label_test = []
            func_targets_test = []
            func_pred_prob_test = []
            func_pred_label_test = []


            surgery_patient_ID_test = []
            reflux_patient_ID_test = []
            func_patient_ID_test = []

            counter_train = 0
            counter_test = 0
            net.train()
            for batch_idx, (x, y, cov) in enumerate(training_generator):
                optimizer.zero_grad()
                surg_batch_x, y_surgery, surg_batch_cov = torch.stack([x[i][:2] for i in range(len(x)) if y[i][0] != -1]), torch.stack([y[i][0] for i in range(len(y)) if y[i][0] != -1]), [cov[i] for i in range(len(cov)) if y[i][0] != -1]
                reflux_batch_x, y_reflux, reflux_batch_cov = torch.stack([x[i][2:4] for i in range(len(x)) if y[i][1] != -1]), torch.stack([y[i][1] for i in range(len(y)) if y[i][1] != -1]), [cov[i] for i in range(len(cov)) if y[i][1] != -1]
                func_batch_x, y_func, func_batch_cov =  torch.stack([x[i][4:] for i in range(len(x)) if y[i][2] != -1]), torch.stack([y[i][2] for i in range(len(y)) if y[i][2] != -1]), [cov[i] for i in range(len(cov)) if y[i][2] != -1]
                loss_surgery, loss_reflux, loss_func = 0, 0, 0
                if surg_batch_x.size()[0] > 0:    
                    pred_surgery = net(surg_batch_x.float().to(device), 's')
                    if len(pred_surgery.shape) == 1:
                        pred_surgery = pred_surgery.unsqueeze(0)
                    assert len(pred_surgery) == len(y_surgery)
                    y_surgery = Variable(y_surgery.type(torch.LongTensor), requires_grad=False).to(device)
                    loss_surgery = F.cross_entropy(pred_surgery, y_surgery)
                if reflux_batch_x.size()[0] > 0:
                    pred_reflux = net(reflux_batch_x.float().to(device), 'r')
                    if len(pred_reflux.shape) == 1:
                        pred_reflux = pred_reflux.unsqueeze(0)
                    assert len(pred_reflux) == len(y_reflux)
                    y_reflux = Variable(y_reflux.type(torch.LongTensor), requires_grad=False).to(device)
                    loss_reflux = F.cross_entropy(pred_reflux, y_reflux)
                if func_batch_x.size()[0] > 0:
                    pred_func = net(func_batch_x.float().to(device), 'f')
                    if len(pred_func.shape) == 1:
                        pred_func = pred_func.unsqueeze(0)
                    assert len(pred_func) == len(y_func)
                    y_func = Variable(y_func.type(torch.LongTensor), requires_grad=False).to(device)
                    loss_func = F.cross_entropy(pred_func, y_func)
                                
                loss = loss_surgery + loss_reflux + loss_func
                loss_accum_train += loss.item() * (len(y_surgery) + len(y_reflux) + len(y_func))

                loss.backward()

                #accurate_labels_train += torch.sum(torch.argmax(output, dim=1) == target).cpu()
                optimizer.step()
                counter_train += len(y_surgery) + len(y_reflux) + len(y_func)

                surgery_pred_prob, surgery_pred_label = get_proba(pred_surgery)
                reflux_pred_prob, reflux_pred_label = get_proba(pred_reflux)
                func_pred_prob, func_pred_label = get_proba(pred_func) 

                surgery_pred_prob_train.append(surgery_pred_prob)
                surgery_targets_train.append(y_surgery)
                surgery_pred_label_train.append(surgery_pred_label)
                surgery_patient_ID_train.extend(surg_batch_cov)

                reflux_pred_prob_train.append(reflux_pred_prob)
                reflux_targets_train.append(y_reflux)
                reflux_pred_label_train.append(reflux_pred_label)
                reflux_patient_ID_train.extend(reflux_batch_cov)

                func_pred_prob_train.append(func_pred_prob)
                func_targets_train.append(y_func)
                func_pred_label_train.append(func_pred_label)
                func_patient_ID_train.extend(func_batch_cov)
            surgery_pred_prob_train = torch.cat(surgery_pred_prob_train)
            surgery_targets_train = torch.cat(surgery_targets_train)
            surgery_pred_label_train = torch.cat(surgery_pred_label_train)
            reflux_pred_prob_train = torch.cat(reflux_pred_prob_train)
            reflux_targets_train = torch.cat(reflux_targets_train)
            reflux_pred_label_train = torch.cat(reflux_pred_label_train)
            func_pred_prob_train = torch.cat(func_pred_prob_train)
            func_targets_train = torch.cat(func_targets_train)
            func_pred_label_train = torch.cat(func_pred_label_train)
            print("Fold\t{}\tTrainEpoch\t{}\tLoss\t{}".format(fold, epoch, loss_accum_train/counter_train))

            surgery_results_train = process_results.get_metrics(y_score=surgery_pred_prob_train.cpu().detach().numpy(),
                                                  y_true=surgery_targets_train.cpu().detach().numpy(),
                                                  y_pred=surgery_pred_label_train.cpu().detach().numpy())

            print('Fold\t{}\tTrainEpoch\t{}\tSURGERY\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch,
                                                                        surgery_results_train['auc'],
                                                                        surgery_results_train['auprc'], surgery_results_train['tn'],
                                                                        surgery_results_train['fp'], surgery_results_train['fn'],
                                                                        surgery_results_train['tp']))
            reflux_results_train = process_results.get_metrics(y_score=reflux_pred_prob_train.cpu().detach().numpy(),
                                                  y_true=reflux_targets_train.cpu().detach().numpy(),
                                                  y_pred=reflux_pred_label_train.cpu().detach().numpy())

            print('Fold\t{}\tTrainEpoch\t{}\tREFLUX\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch,
                                                                        reflux_results_train['auc'],
                                                                        reflux_results_train['auprc'], reflux_results_train['tn'],
                                                                        reflux_results_train['fp'], reflux_results_train['fn'],
                                                                        reflux_results_train['tp']))
            func_results_train = process_results.get_metrics(y_score=func_pred_prob_train.cpu().detach().numpy(),
                                                  y_true=func_targets_train.cpu().detach().numpy(),
                                                  y_pred=func_pred_label_train.cpu().detach().numpy())

            print('Fold\t{}\tTrainEpoch\t{}\tFUNC\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch,
                                                                        func_results_train['auc'],
                                                                        func_results_train['auprc'], func_results_train['tn'],
                                                                        func_results_train['fp'], func_results_train['fn'],
                                                                        func_results_train['tp']))
    #        net.eval()
   #         with torch.no_grad():
  #              for batch_idx, (data, target, cov) in enumerate(validation_generator):
   #                 net.zero_grad()
    #                optimizer.zero_grad()
     #               output = net(data)
      #              target = target.type(torch.LongTensor).to(device)
###
   #                 loss = F.cross_entropy(output, target)
    #                loss_accum_val += loss.item() * len(target)
     #               counter_val += len(target)
      #              output_softmax = softmax(output)
#
 #                   accurate_labels_val += torch.sum(torch.argmax(output, dim=1) == target).cpu()
#
 #                   pred_prob = output_softmax[:,1]
  #                  pred_prob = pred_prob.squeeze()
   #                 pred_label = torch.argmax(output, dim=1)
#
 #                   assert len(pred_prob) == len(target)
  #                  assert len(pred_label) == len(target)
#
 #                   all_pred_prob_val.append(pred_prob)
  #                  all_targets_val.append(target)
   #                 all_pred_label_val.append(pred_label)
#
 #                   patient_ID_val.extend(cov)
            net.eval()
            with torch.no_grad():
                for batch_idx, (x, y, cov) in enumerate(test_generator):
                    net.zero_grad()
                    net.eval() # 20190619
                    optimizer.zero_grad()
                    surg_batch_x, y_surgery, surg_batch_cov = torch.stack([x[i][:2] for i in range(len(x)) if y[i][0] != -1]), torch.stack([y[i][0] for i in range(len(y)) if y[i][0] != -1]), [cov[i] for i in range(len(cov)) if y[i][0] != -1]
                    reflux_batch_x, y_reflux, reflux_batch_cov = torch.stack([x[i][2:4] for i in range(len(x)) if y[i][1] != -1]), torch.stack([y[i][1] for i in range(len(y)) if y[i][1] != -1]), [cov[i] for i in range(len(cov)) if y[i][1] != -1]
                    func_batch_x, y_func, func_batch_cov =  torch.stack([x[i][4:] for i in range(len(x)) if y[i][2] != -1]), torch.stack([y[i][2] for i in range(len(y)) if y[i][2] != -1]), [cov[i] for i in range(len(cov)) if y[i][2] != -1]
                    loss_surgery, loss_reflux, loss_func = 0, 0, 0
                    if surg_batch_x.size()[0] > 0:
                        pred_surgery = net(surg_batch_x.float().to(device), 's')
                        if len(pred_surgery.shape) == 1:
                            pred_surgery = pred_surgery.unsqueeze(0)
                        assert len(pred_surgery) == len(y_surgery)
                        y_surgery = Variable(y_surgery.type(torch.LongTensor), requires_grad=False).to(device)
                        loss_surgery = F.cross_entropy(pred_surgery, y_surgery)
                    if reflux_batch_x.size()[0] > 0:
                        pred_reflux = net(reflux_batch_x.float().to(device), 'r')
                        if len(pred_reflux.shape) == 1:
                            pred_reflux = pred_reflux.unsqueeze(0)
                        assert len(pred_reflux) == len(y_reflux)
                        y_reflux = Variable(y_reflux.type(torch.LongTensor), requires_grad=False).to(device)
                        loss_reflux = F.cross_entropy(pred_reflux, y_reflux)
                    if func_batch_x.size()[0] > 0:
                        pred_func = net(func_batch_x.float().to(device), 'f')
                        if len(pred_func.shape) == 1:
                            pred_func = pred_func.unsqueeze(0)
                        assert len(pred_func) == len(y_func)
                        y_func = Variable(y_func.type(torch.LongTensor), requires_grad=False).to(device)
                        loss_func = F.cross_entropy(pred_func, y_func)
                    loss = loss_surgery + loss_reflux + loss_func
                    loss_accum_test += loss.item() * (len(y_surgery) + len(y_reflux) + len(y_func))

                    #loss.backward()

                    #accurate_labels_train += torch.sum(torch.argmax(output, dim=1) == target).cpu()
                    optimizer.step()
                    counter_test += len(y_surgery) + len(y_reflux) + len(y_func)

                    surgery_pred_prob, surgery_pred_label = get_proba(pred_surgery)
                    reflux_pred_prob, reflux_pred_label = get_proba(pred_reflux)
                    func_pred_prob, func_pred_label = get_proba(pred_func)

                    surgery_pred_prob_test.append(surgery_pred_prob)
                    surgery_targets_test.append(y_surgery)
                    surgery_pred_label_test.append(surgery_pred_label)
                    surgery_patient_ID_test.extend(surg_batch_cov)

                    reflux_pred_prob_test.append(reflux_pred_prob)
                    reflux_targets_test.append(y_reflux)
                    reflux_pred_label_test.append(reflux_pred_label)
                    reflux_patient_ID_test.extend(reflux_batch_cov)

                    func_pred_prob_test.append(func_pred_prob)
                    func_targets_test.append(y_func)
                    func_pred_label_test.append(func_pred_label)
                    func_patient_ID_test.extend(func_batch_cov)
                surgery_pred_prob_test = torch.cat(surgery_pred_prob_test)
                surgery_targets_test = torch.cat(surgery_targets_test)
                surgery_pred_label_test = torch.cat(surgery_pred_label_test)
                reflux_pred_prob_test = torch.cat(reflux_pred_prob_test)
                reflux_targets_test = torch.cat(reflux_targets_test)
                reflux_pred_label_test = torch.cat(reflux_pred_label_test)
                func_pred_prob_test = torch.cat(func_pred_prob_test)
                func_targets_test = torch.cat(func_targets_test)
                func_pred_label_test = torch.cat(func_pred_label_test)
             
                print("Fold\t{}\tTestEpoch\t{}\tLoss\t{}".format(fold, epoch, loss_accum_test/counter_test))

                surgery_results_test = process_results.get_metrics(y_score=surgery_pred_prob_test.cpu().detach().numpy(),
                                                      y_true=surgery_targets_test.cpu().detach().numpy(),
                                                      y_pred=surgery_pred_label_test.cpu().detach().numpy())

                print('Fold\t{}\tTestEpoch\t{}\tSURGERY\tAUC\t{:.6f}\t'
                      'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch,
                                                                            surgery_results_test['auc'],
                                                                            surgery_results_test['auprc'], surgery_results_test['tn'],
                                                                            surgery_results_test['fp'], surgery_results_test['fn'],
                                                                            surgery_results_test['tp']))
                reflux_results_test = process_results.get_metrics(y_score=reflux_pred_prob_test.cpu().detach().numpy(),
                                                      y_true=reflux_targets_test.cpu().detach().numpy(),
                                                      y_pred=reflux_pred_label_test.cpu().detach().numpy())

                print('Fold\t{}\tTestEpoch\t{}\tREFLUX\tAUC\t{:.6f}\t'
                      'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch,
                                                                            reflux_results_test['auc'],
                                                                            reflux_results_test['auprc'], reflux_results_test['tn'],
                                                                            reflux_results_test['fp'], reflux_results_test['fn'],
                                                                            reflux_results_test['tp']))
                func_results_test = process_results.get_metrics(y_score=func_pred_prob_test.cpu().detach().numpy(),
                                                      y_true=func_targets_test.cpu().detach().numpy(),
                                                      y_pred=func_pred_label_test.cpu().detach().numpy())

                print('Fold\t{}\tTestEpoch\t{}\tFUNC\tAUC\t{:.6f}\t'
                      'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch,
                                                                            func_results_test['auc'],
                                                                            func_results_test['auprc'], func_results_test['tn'],
                                                                            func_results_test['fp'], func_results_test['fn'],
                                                                            func_results_test['tp']))

           
            # if ((epoch+1) % 5) == 0 and epoch > 0:
            checkpoint = {'epoch': epoch,
                          'loss': loss,
                          'hyperparams': hyperparams,
                          'args': args,
                          'model_state_dict': net.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'loss_train': loss_accum_train / counter_train,
                          'surgery_results_train': surgery_results_train,
                          'reflux_results_train': reflux_results_train,
                          'func_results_train': func_results_train,
                          'surgery_pred_prob_train': surgery_pred_prob_train,
                          'reflux_pred_prob_train': reflux_pred_prob_train,
                          'func_pred_prob_train': func_pred_prob_train,
                          'surgery_targets_train': surgery_targets_train,
                          'reflux_targets_train': reflux_targets_train,
                          'func_targets_train': func_targets_train,
                          'surgery_patient_ID_train': surgery_patient_ID_train,
                          'reflux_patient_ID_train': reflux_patient_ID_train,
                          'func_patient_ID_train': func_patient_ID_train,
                          'loss_test': loss_accum_test / counter_test,
                          'surgery_results_test': surgery_results_test,
                          'reflux_results_test': reflux_results_test,
                          'func_results_test': func_results_test,
                          'surgery_pred_prob_test': surgery_pred_prob_test,
                          'reflux_pred_prob_test': reflux_pred_prob_test,
                          'func_pred_prob_test': func_pred_prob_test,
                          'surgery_targets_test': surgery_targets_test,
                          'reflux_targets_test': reflux_targets_test,
                          'func_targets_test': func_targets_test,
                          'surgery_patient_ID_test': surgery_patient_ID_test,
                          'reflux_patient_ID_test': reflux_patient_ID_test,
                          'func_patient_ID_test': func_patient_ID_test}

            if not os.path.isdir(args.dir):
                os.makedirs(args.dir)
            #if not os.path.isdir(args.dir + "/" + str(fold)):
                #os.makedirs(args.dir + "/" + str(fold))
            path_to_checkpoint = args.dir + "/" + str(fold) + "_checkpoint_" + str(epoch) + '.pth'
            torch.save(checkpoint, path_to_checkpoint)
            
        fold += 1
        



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=50, type=int, help="Number of epochs")
    parser.add_argument('--batch_size', default=64, type=int, help="Batch size")
    parser.add_argument('--lr', default=0.001, type=float, help="Learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, help="Momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay")
    parser.add_argument("--num_workers", default=1, type=int, help="Number of CPU workers")
    parser.add_argument("--dir", default="./", help="Directory to save model checkpoints to")
    parser.add_argument("--contrast", default=1, type=int, help="Image contrast to train on")
    parser.add_argument("--view", default="siamese", help="siamese, sag, trans")
    parser.add_argument("--checkpoint", default="", help="Path to load pretrained model checkpoint from")
    parser.add_argument("--split", default=0.7, type=float, help="proportion of dataset to use as training")
    parser.add_argument("--bottom_cut", default=0.0, type=float, help="proportion of dataset to cut from bottom")
    parser.add_argument("--etiology", default="B", help="O (obstruction), R (reflux), B (both)")
    parser.add_argument("--hydro_only", action="store_true")
    parser.add_argument("--output_dim", default=256, type=int, help="output dim for last linear layer")
    parser.add_argument("--datafile", default="../../0.Preprocess/preprocessed_images_20190617.pickle", help="File containing pandas dataframe with images stored as numpy array")
    parser.add_argument("--gender", default=None, type=str, help="choose from 'male' and 'female'")
    args = parser.parse_args()

    print("ARGS" + '\t' + str(args))

    max_epochs = args.epochs
    
    #load_dataset_surgery = importlib.machinery.SourceFileLoader('load_dataset','../../0.Preprocess/load_dataset.py').load_module()
    import load_dataset as load_dataset_surgery
    train_X_surg, train_y_surg, train_cov_surg, test_X_surg, test_y_surg, test_cov_surg = load_dataset_surgery.load_dataset(views_to_get="siamese", pickle_file=args.datafile,
                                                                                      contrast=args.contrast, split=args.split, get_cov=True)
    #load_dataset_vur = importlib.machinery.SourceFileLoader('load_dataset','../../0.Preprocess/load_dataset_vur.py').load_module()
    import load_dataset_vur
    train_X_reflux, train_y_reflux, train_cov_reflux, test_X_reflux, test_y_reflux, test_cov_reflux = load_dataset_vur.load_dataset(views_to_get="siamese", pickle_file="vur_images_256_20200422.pickle", image_dim=256, contrast=args.contrast, split=args.split, get_cov=True)
    
    #load_dataset_func = importlib.machinery.SourceFileLoader('load_dataset','../../0.Preprocess/load_dataset_func.py').load_module()
    import load_dataset_func
    train_X_func, train_y_func, train_cov_func, test_X_func, test_y_func, test_cov_func = load_dataset_func.load_dataset(views_to_get="siamese", pickle_file="func_images_256_20200422.pickle", image_dim=256,  contrast=args.contrast, split=args.split, get_cov=True)
    train_y_func = [1 if item >=0.6 or item <= 0.4 else 0 for item in train_y_func]
    test_y_func = [1 if item >=0.6 or item <= 0.4 else 0 for item in test_y_func] 
    get_data_stats(train_y_surg, "SURG_TRAIN")
    get_data_stats(test_y_surg, "SURG_TEST")
    print("-----------------------------------")
    get_data_stats(train_y_reflux, "REFLUX_TRAIN")
    get_data_stats(test_y_reflux, "REFLUX_TEST")
    print("-----------------------------------")
    get_data_stats(train_y_func, "FUNC_TRAIN")
    get_data_stats(test_y_func, "FUNC_TEST")
    print("-----------------------------------")
    import sys; sys.exit(0)
    train_X = []
    train_y = []
    train_cov = []
    for i in range(len(train_cov_reflux)):
        item_ = train_cov_reflux[i].split("_")[:-1]
        item_[0], item_[1] = str(float(item_[0])), str(float(item_[1]))
        item_ = "_".join(item_)
        train_cov_reflux[i] = item_
    for i in range(len(train_cov_func)):
        train_cov_func[i] = "_".join(train_cov_func[i].split("_")[:-1])
         
    dataset_train = {}
    for i in range(len(train_X_surg)): 
        if train_cov_surg[i] not in dataset_train:
            dataset_train[train_cov_surg[i]] = {}
        dataset_train[train_cov_surg[i]]['s'] = (train_X_surg[i], train_y_surg[i])
    for i in range(len(train_X_reflux)):
        if train_cov_reflux[i] not in dataset_train:
            dataset_train[train_cov_reflux[i]] = {}
        dataset_train[train_cov_reflux[i]]['r'] = (train_X_reflux[i], train_y_reflux[i])
    for i in range(len(train_X_func)):
        if train_cov_func[i] not in dataset_train:
            dataset_train[train_cov_func[i]] = {}
        dataset_train[train_cov_func[i]]['f'] = (train_X_func[i], train_y_func[i])
    
    test_X = []
    test_y = []
    test_cov = []
    for i in range(len(test_cov_reflux)):
        item_ = test_cov_reflux[i].split("_")[:-1]
        item_[0], item_[1] = str(float(item_[0])), str(float(item_[1]))
        item_ = "_".join(item_)
        test_cov_reflux[i] = item_
    for i in range(len(test_cov_func)):
        test_cov_func[i] = "_".join(test_cov_func[i].split("_")[:-1])

    dataset_test = {}
    for i in range(len(test_X_surg)):
        if test_cov_surg[i] not in dataset_test:
            dataset_test[test_cov_surg[i]] = {}
        dataset_test[test_cov_surg[i]]['s'] = (test_X_surg[i], test_y_surg[i])
    for i in range(len(test_X_reflux)):
        if test_cov_reflux[i] not in dataset_test:
            dataset_test[test_cov_reflux[i]] = {}
        dataset_test[test_cov_reflux[i]]['r'] = (test_X_reflux[i], test_y_reflux[i])
    for i in range(len(test_X_func)):
        if test_cov_func[i] not in dataset_test:
            dataset_test[test_cov_func[i]] = {}
        dataset_test[test_cov_func[i]]['f'] = (test_X_func[i], test_y_func[i])

    train(args, dataset_train, dataset_test, max_epochs)


if __name__ == '__main__':
    main()
