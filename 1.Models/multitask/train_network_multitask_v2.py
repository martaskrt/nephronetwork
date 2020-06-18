from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
#import torchvision.datasets.mnist
from torchvision import transforms
import importlib.machinery
from torch.utils.data import Dataset, DataLoader
import argparse
from torch.autograd import Variable
from sklearn.utils import class_weight
import PIL
from PIL import Image
import pickle
# import botorch
from hyperopt import hp, tpe, fmin

process_results = importlib.machinery.SourceFileLoader('process_results','../../2.Results/process_results.py').load_module()
#import process_results

SEED = 42

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
softmax = torch.nn.Softmax(dim=1)
criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")

class KidneyDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, dataset_type):
        self.X = []
        self.y = []
        self.cov = []
        dic = {}        
        for key in dataset:
            self.cov.append(key)
            self.y.append(dataset[key]['labels'])
            x = np.zeros((4, 256, 256)) # change this later for single view
            if dataset[key]['left'] is not None:
                x[:2,:,:] = dataset[key]['left']
            if dataset[key]['right'] is not None:
                x[2:,:,:] = dataset[key]['right']
            self.X.append(x)
            
         #   dic[key] = {'img': x, 'labels': dataset[key]['labels']}
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.cov = np.array(self.cov)
        
        num_1 = 0
        num_0 = 0
        for item in self.y:
            if item[-1] == 1:
                num_1 += 1
            elif item[-1] == 0:
                num_0 += 1
        print("{}_FUNC:::{} class_0/{} class_1".format(dataset_type.upper(), num_0, num_1))
       # with open("multitask_{}_dataset.pickle".format(dataset_type), 'wb') as fhandle:
        #    pickle.dump(dic, fhandle)
        #for i in range(len(self.cov)):
         #   imgs_comb = PIL.Image.fromarray(np.hstack((self.X[i][0]*255, self.X[i][1]*255, self.X[i][2]*255, self.X[i][3]*255))).convert('RGB')
          #  imgs_comb.save('{}/{}.jpg'.format(dataset_type, self.cov[i]))
    def __getitem__(self, index):
        return self.X[index], self.y[index], self.cov[index]

    def __len__(self):
        return len(self.X)

def get_proba(x):
    pred_prob, pred_label = torch.tensor([]).to(device), torch.tensor([]).to(device)
    try:
        x = torch.stack(x)
        x_softmax = softmax(x)
        pred_prob = x_softmax[:, 1]
        pred_label = torch.argmax(x_softmax, dim=1)
    except:
        pass 
    return pred_prob, pred_label

def get_data_stats(y, label_type):
    print("{}:::{} class_0/{} class_1/{} total".format(label_type, len(y)-sum(y), sum(y), len(y)))

def train(args, dataset_train, dataset_test,  max_epochs):

    from CNN_multitask_2 import CNN

    hyperparams = {'lr': args.lr,
                   'momentum': args.momentum,
                   'weight_decay': args.weight_decay,
                   'train/test_split': args.split 
                  }
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.num_workers}

    if args.val:
        train_cov_shuffled = shuffle(sorted(list(dataset_train.keys())), random_state=42)
        val_covs = train_cov_shuffled[int(len(train_cov_shuffled)*0.85):]
        train_covs = train_cov_shuffled[:int(len(train_cov_shuffled)*0.85)]

        dataset_val = {key: dataset_train[key] for key in val_covs}
        dataset_train = {key: dataset_train[key] for key in train_covs}  
            
        val_set = KidneyDataset(dataset_val, "val")
        val_generator = DataLoader(val_set, **params)

    training_set = KidneyDataset(dataset_train, "train")
    training_generator = DataLoader(training_set, **params)
 

    test_set = KidneyDataset(dataset_test, "test")
    test_generator = DataLoader(test_set, **params)
    fold=1
    for _ in range(1): 
        if args.view != "siamese":
            net = CNN(num_views=1, three_channel=False).to(device)
        else:
            net = CNN(num_views=2, three_channel=False).to(device)

        net.zero_grad()
        optimizer = torch.optim.SGD(net.parameters(), lr=hyperparams['lr'], momentum=hyperparams['momentum'],
                                        weight_decay=hyperparams['weight_decay'])

       

        for epoch in range(max_epochs):
            loss_accum_train, loss_accum_val, loss_accum_test = 0, 0, 0

            surgery_targets_train, surgery_pred_prob_train, surgery_pred_label_train = [], [], []
            reflux_targets_train, reflux_pred_prob_train, reflux_pred_label_train = [], [], []
            func_targets_train, func_pred_prob_train, func_pred_label_train = [], [], []

            surgery_patient_ID_train, reflux_patient_ID_train, func_patient_ID_train = [], [], []

            surgery_targets_val, surgery_pred_prob_val, surgery_pred_label_val = [], [], []
            reflux_targets_val, reflux_pred_prob_val, reflux_pred_label_val = [], [], []
            func_targets_val, func_pred_prob_val, func_pred_label_val = [], [], []

            surgery_patient_ID_val, reflux_patient_ID_val, func_patient_ID_val = [], [], []

            surgery_targets_test, surgery_pred_prob_test, surgery_pred_label_test = [], [], []
            reflux_targets_test, reflux_pred_prob_test, reflux_pred_label_test = [], [], []
            func_targets_test, func_pred_prob_test, func_pred_label_test = [], [], []

            surgery_patient_ID_test, reflux_patient_ID_test, func_patient_ID_test = [], [], []

            counter_train, counter_val, counter_test = 0, 0, 0
            
            net.train()
            for batch_idx, (x, y, cov) in enumerate(training_generator):
                optimizer.zero_grad()
                preds = net(x.float().to(device))
                #if len(preds.shape) == 1:
                 #   preds = pred_surgery.unsqueeze(0)
                assert len(preds) == 5
                assert len(y) == len(x)
                y = Variable(y.type(torch.LongTensor), requires_grad=False).to(device)
                loss=0
                for i in range(5):
                    if i == 4:
                        loss += criterion(preds[i], y[:, i])
                    else:
                        loss += 0.283789623517092*criterion(preds[i], y[:, i])
                
                #loss_accum_train += loss.item() * (len(y_surgery) + len(y_reflux) + len(y_func))
                loss.backward()
                optimizer.step()
                #loss.backward()

                #optimizer.step()
                pred_surgery, pred_reflux, pred_func = [], [], []
                surg_batch_cov, reflux_batch_cov, func_batch_cov = [], [], []
                y_surgery, y_reflux, y_func = [], [], []
                
                for i in range(len(y)):
                    if y[i][0] != -100:
                        y_surgery.append(y[i][0].cpu().detach().numpy()); pred_surgery.append(preds[0][i]); surg_batch_cov.append(cov[i])
                    if y[i][1] != -100:
                        y_surgery.append(y[i][1].cpu().detach().numpy()); pred_surgery.append(preds[1][i]); surg_batch_cov.append(cov[i])
                    if y[i][2] != -100:
                        y_reflux.append(y[i][2].cpu().detach().numpy()); pred_reflux.append(preds[2][i]); reflux_batch_cov.append(cov[i])
                    if y[i][3] != -100:
                        y_reflux.append(y[i][3].cpu().detach().numpy()); pred_reflux.append(preds[3][i]); reflux_batch_cov.append(cov[i])
                    if y[i][4] != -100:
                        y_func.append(y[i][4].cpu().detach().numpy()); pred_func.append(preds[4][i]); func_batch_cov.append(cov[i])
                counter_train += len(pred_surgery) + len(pred_reflux) + len(pred_func)
                loss_accum_train += loss.item() * (len(pred_surgery) + len(pred_reflux) + len(pred_func))


                surgery_pred_prob, surgery_pred_label = get_proba(pred_surgery)
                reflux_pred_prob, reflux_pred_label = get_proba(pred_reflux)
                func_pred_prob, func_pred_label = get_proba(pred_func) 

                surgery_pred_prob_train.append(surgery_pred_prob.cpu().detach().numpy())
                surgery_targets_train.append(y_surgery)
                surgery_pred_label_train.append(surgery_pred_label.cpu().detach().numpy())
                surgery_patient_ID_train.extend(surg_batch_cov)

                reflux_pred_prob_train.append(reflux_pred_prob.cpu().detach().numpy())
                reflux_targets_train.append(y_reflux)
                reflux_pred_label_train.append(reflux_pred_label.cpu().detach().numpy())
                reflux_patient_ID_train.extend(reflux_batch_cov)

                func_pred_prob_train.append(func_pred_prob.cpu().detach().numpy())
                func_targets_train.append(y_func)
                func_pred_label_train.append(func_pred_label.cpu().detach().numpy())
                func_patient_ID_train.extend(func_batch_cov)
            surgery_pred_prob_train = np.concatenate(np.array(surgery_pred_prob_train))
            surgery_targets_train = np.concatenate(np.array(surgery_targets_train))
            surgery_pred_label_train = np.concatenate(np.array(surgery_pred_label_train))
            reflux_pred_prob_train = np.concatenate(reflux_pred_prob_train)
            reflux_targets_train = np.concatenate(reflux_targets_train)
            reflux_pred_label_train = np.concatenate(reflux_pred_label_train)
            func_pred_prob_train = np.concatenate(func_pred_prob_train)
            func_targets_train = np.concatenate(func_targets_train)
            func_pred_label_train = np.concatenate(func_pred_label_train)
            print("Fold\t{}\tTrainEpoch\t{}\tLoss\t{}".format(fold, epoch, loss_accum_train/counter_train))
            surgery_results_train = process_results.get_metrics(y_score=surgery_pred_prob_train, y_true=surgery_targets_train, y_pred=surgery_pred_label_train)

            print('Fold\t{}\tTrainEpoch\t{}\tSURGERY\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch,
                                                                        surgery_results_train['auc'],
                                                                        surgery_results_train['auprc'], surgery_results_train['tn'],
                                                                        surgery_results_train['fp'], surgery_results_train['fn'],
                                                                        surgery_results_train['tp']))
            reflux_results_train = process_results.get_metrics(y_score=reflux_pred_prob_train,
                                                  y_true=reflux_targets_train,
                                                  y_pred=reflux_pred_label_train)

            print('Fold\t{}\tTrainEpoch\t{}\tREFLUX\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch,
                                                                        reflux_results_train['auc'],
                                                                        reflux_results_train['auprc'], reflux_results_train['tn'],
                                                                        reflux_results_train['fp'], reflux_results_train['fn'],
                                                                        reflux_results_train['tp']))
            func_results_train = process_results.get_metrics(y_score=func_pred_prob_train,
                                                  y_true=func_targets_train,
                                                  y_pred=func_pred_label_train)

            print('Fold\t{}\tTrainEpoch\t{}\tFUNC\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch,
                                                                        func_results_train['auc'],
                                                                        func_results_train['auprc'], func_results_train['tn'],
                                                                        func_results_train['fp'], func_results_train['fn'],
                                                                        func_results_train['tp']))
            if args.val:
                with torch.no_grad():
                    for batch_idx, (x, y, cov) in enumerate(val_generator):
                        net.zero_grad()
                        net.eval()  # 20190619
                        optimizer.zero_grad()
                        preds = net(x.float().to(device))
                        assert len(preds) == 5
                        assert len(y) == len(x)
                        loss=0
                        y = Variable(y.type(torch.LongTensor), requires_grad=False).to(device)
                        for i in range(5):
                            if i == 4:
                                loss += criterion(preds[i], y[:, i])
                            else:
                                loss += 0.283789623517092*criterion(preds[i], y[:, i])
    
                        loss_accum_val += loss.item() * (len(y_surgery) + len(y_reflux) + len(y_func))
    
                        #loss.backward()
    
                        #optimizer.step()
                        pred_surgery, pred_reflux, pred_func = [], [], []
                        surg_batch_cov, reflux_batch_cov, func_batch_cov = [], [], []
                        y_surgery, y_reflux, y_func = [], [], []
    
                        for i in range(len(y)):
                            if y[i][0] != -100:
                                y_surgery.append(y[i][0].cpu().detach().numpy()); pred_surgery.append(preds[0][i]); surg_batch_cov.append(cov[i])
                            if y[i][1] != -100:
                                y_surgery.append(y[i][1].cpu().detach().numpy()); pred_surgery.append(preds[1][i]); surg_batch_cov.append(cov[i])
                            if y[i][2] != -100:
                                y_reflux.append(y[i][2].cpu().detach().numpy()); pred_reflux.append(preds[2][i]); reflux_batch_cov.append(cov[i])
                            if y[i][3] != -100:
                                y_reflux.append(y[i][3].cpu().detach().numpy()); pred_reflux.append(preds[3][i]); reflux_batch_cov.append(cov[i])
                            if y[i][4] != -100:
                                y_func.append(y[i][4].cpu().detach().numpy()); pred_func.append(preds[4][i]); func_batch_cov.append(cov[i])
                        counter_val += len(pred_surgery) + len(pred_reflux) + len(pred_func)
    
    
                        surgery_pred_prob, surgery_pred_label = get_proba(pred_surgery)
                        reflux_pred_prob, reflux_pred_label = get_proba(pred_reflux)
                        func_pred_prob, func_pred_label = get_proba(pred_func)
    
                        surgery_pred_prob_val.append(surgery_pred_prob.cpu().detach().numpy())
                        surgery_targets_val.append(y_surgery)
                        surgery_pred_label_val.append(surgery_pred_label.cpu().detach().numpy())
                        surgery_patient_ID_val.extend(surg_batch_cov)
    
                        reflux_pred_prob_val.append(reflux_pred_prob.cpu().detach().numpy())
                        reflux_targets_val.append(y_reflux)
                        reflux_pred_label_val.append(reflux_pred_label.cpu().detach().numpy())
                        reflux_patient_ID_val.extend(reflux_batch_cov)
    
                        func_pred_prob_val.append(func_pred_prob.cpu().detach().numpy())
                        func_targets_val.append(y_func)
                        func_pred_label_val.append(func_pred_label.cpu().detach().numpy())
                        func_patient_ID_val.extend(func_batch_cov)
                    surgery_pred_prob_val = np.concatenate(surgery_pred_prob_val)
                    surgery_targets_val = np.concatenate(surgery_targets_val)
                    surgery_pred_label_val = np.concatenate(surgery_pred_label_val)
                    reflux_pred_prob_val = np.concatenate(reflux_pred_prob_val)
                    reflux_targets_val = np.concatenate(reflux_targets_val)
                    reflux_pred_label_val = np.concatenate(reflux_pred_label_val)
                    func_pred_prob_val = np.concatenate(func_pred_prob_val)
                    func_targets_val = np.concatenate(func_targets_val)
                    func_pred_label_val = np.concatenate(func_pred_label_val)
                 
                    print("Fold\t{}\tValEpoch\t{}\tLoss\t{}".format(fold, epoch, loss_accum_val/counter_val))
    
                    surgery_results_val = process_results.get_metrics(y_score=surgery_pred_prob_val,
                                                          y_true=surgery_targets_val,
                                                          y_pred=surgery_pred_label_val)
    
                    print('Fold\t{}\tValEpoch\t{}\tSURGERY\tAUC\t{:.6f}\t'
                          'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch,
                                                                                surgery_results_val['auc'],
                                                                                surgery_results_val['auprc'], surgery_results_val['tn'],
                                                                                surgery_results_val['fp'], surgery_results_val['fn'],
                                                                                surgery_results_val['tp']))
                    reflux_results_val = process_results.get_metrics(y_score=reflux_pred_prob_val,
                                                          y_true=reflux_targets_val,
                                                          y_pred=reflux_pred_label_val)
    
                    print('Fold\t{}\tValEpoch\t{}\tREFLUX\tAUC\t{:.6f}\t'
                          'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch,
                                                                                reflux_results_val['auc'],
                                                                                reflux_results_val['auprc'], reflux_results_val['tn'],
                                                                                reflux_results_val['fp'], reflux_results_val['fn'],
                                                                                reflux_results_val['tp']))
                    func_results_val = process_results.get_metrics(y_score=func_pred_prob_val,
                                                          y_true=func_targets_val,
                                                          y_pred=func_pred_label_val)
    
                    print('Fold\t{}\tValEpoch\t{}\tFUNC\tAUC\t{:.6f}\t'
                          'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch,
                                                                                func_results_val['auc'],
                                                                                func_results_val['auprc'], func_results_val['tn'],
                                                                                func_results_val['fp'], func_results_val['fn'],
                                                                                func_results_val['tp']))
    
            net.eval()
            with torch.no_grad():
                for batch_idx, (x, y, cov) in enumerate(test_generator):
                    net.zero_grad()
                    net.eval()  # 20190619
                    optimizer.zero_grad()
                    preds = net(x.float().to(device))
                    assert len(preds) == 5
                    assert len(y) == len(x)
                    loss=0
                    y = Variable(y.type(torch.LongTensor), requires_grad=False).to(device)
                    for i in range(5):
                        if i == 4:
                            loss += criterion(preds[i], y[:, i])
                        else:
                            loss += 0.283789623517092*criterion(preds[i], y[:, i])

                    loss_accum_test += loss.item() * (len(y_surgery) + len(y_reflux) + len(y_func))

                    #loss.backward()

                    #optimizer.step()
                    pred_surgery, pred_reflux, pred_func = [], [], []
                    surg_batch_cov, reflux_batch_cov, func_batch_cov = [], [], []
                    y_surgery, y_reflux, y_func = [], [], []

                    for i in range(len(y)):
                        if y[i][0] != -100:
                            y_surgery.append(y[i][0].cpu().detach().numpy()); pred_surgery.append(preds[0][i]); surg_batch_cov.append(cov[i])
                        if y[i][1] != -100:
                            y_surgery.append(y[i][1].cpu().detach().numpy()); pred_surgery.append(preds[1][i]); surg_batch_cov.append(cov[i])
                        if y[i][2] != -100:
                            y_reflux.append(y[i][2].cpu().detach().numpy()); pred_reflux.append(preds[2][i]); reflux_batch_cov.append(cov[i])
                        if y[i][3] != -100:
                            y_reflux.append(y[i][3].cpu().detach().numpy()); pred_reflux.append(preds[3][i]); reflux_batch_cov.append(cov[i])
                        if y[i][4] != -100:
                            y_func.append(y[i][4].cpu().detach().numpy()); pred_func.append(preds[4][i]); func_batch_cov.append(cov[i])
                    counter_test += len(pred_surgery) + len(pred_reflux) + len(pred_func)


                    surgery_pred_prob, surgery_pred_label = get_proba(pred_surgery)
                    reflux_pred_prob, reflux_pred_label = get_proba(pred_reflux)
                    func_pred_prob, func_pred_label = get_proba(pred_func)

                    surgery_pred_prob_test.append(surgery_pred_prob.cpu().detach().numpy())
                    surgery_targets_test.append(y_surgery)
                    surgery_pred_label_test.append(surgery_pred_label.cpu().detach().numpy())
                    surgery_patient_ID_test.extend(surg_batch_cov)

                    reflux_pred_prob_test.append(reflux_pred_prob.cpu().detach().numpy())
                    reflux_targets_test.append(y_reflux)
                    reflux_pred_label_test.append(reflux_pred_label.cpu().detach().numpy())
                    reflux_patient_ID_test.extend(reflux_batch_cov)

                    func_pred_prob_test.append(func_pred_prob.cpu().detach().numpy())
                    func_targets_test.append(y_func)
                    func_pred_label_test.append(func_pred_label.cpu().detach().numpy())
                    func_patient_ID_test.extend(func_batch_cov)
                surgery_pred_prob_test = np.concatenate(surgery_pred_prob_test)
                surgery_targets_test = np.concatenate(surgery_targets_test)
                surgery_pred_label_test = np.concatenate(surgery_pred_label_test)
                reflux_pred_prob_test = np.concatenate(reflux_pred_prob_test)
                reflux_targets_test = np.concatenate(reflux_targets_test)
                reflux_pred_label_test = np.concatenate(reflux_pred_label_test)
                func_pred_prob_test = np.concatenate(func_pred_prob_test)
                func_targets_test = np.concatenate(func_targets_test)
                func_pred_label_test = np.concatenate(func_pred_label_test)
             
                print("Fold\t{}\tTestEpoch\t{}\tLoss\t{}".format(fold, epoch, loss_accum_test/counter_test))

                surgery_results_test = process_results.get_metrics(y_score=surgery_pred_prob_test,
                                                      y_true=surgery_targets_test,
                                                      y_pred=surgery_pred_label_test)

                print('Fold\t{}\tTestEpoch\t{}\tSURGERY\tAUC\t{:.6f}\t'
                      'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch,
                                                                            surgery_results_test['auc'],
                                                                            surgery_results_test['auprc'], surgery_results_test['tn'],
                                                                            surgery_results_test['fp'], surgery_results_test['fn'],
                                                                            surgery_results_test['tp']))
                reflux_results_test = process_results.get_metrics(y_score=reflux_pred_prob_test,
                                                      y_true=reflux_targets_test,
                                                      y_pred=reflux_pred_label_test)

                print('Fold\t{}\tTestEpoch\t{}\tREFLUX\tAUC\t{:.6f}\t'
                      'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch,
                                                                            reflux_results_test['auc'],
                                                                            reflux_results_test['auprc'], reflux_results_test['tn'],
                                                                            reflux_results_test['fp'], reflux_results_test['fn'],
                                                                            reflux_results_test['tp']))
                func_results_test = process_results.get_metrics(y_score=func_pred_prob_test,
                                                      y_true=func_targets_test,
                                                      y_pred=func_pred_label_test)

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
            if args.val:
                checkpoint['loss_val'] = loss_accum_val / counter_val; checkpoint['surgery_results_val'] = surgery_results_val; checkpoint['reflux_results_val'] = reflux_results_val; checkpoint['func_results_val'] = func_results_val; checkpoint['surgery_pred_prob_val'] = surgery_pred_prob_val; checkpoint['reflux_pred_prob_val'] = reflux_pred_prob_val; checkpoint['func_pred_prob_val'] = func_pred_prob_val; checkpoint['surgery_targets_val'] = surgery_targets_val; checkpoint['reflux_targets_val'] = reflux_targets_val; checkpoint['func_targets_val'] = func_targets_val; checkpoint['surgery_patient_ID_val'] = surgery_patient_ID_val; checkpoint['reflux_patient_ID_val'] = reflux_patient_ID_val; checkpoint['func_patient_ID_val'] = func_patient_ID_val
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
    parser.add_argument("--val", action="store_true", help="run validation set")
    args = parser.parse_args()
    args.lr = 0.006150337268069674
    print("ARGS" + '\t' + str(args))

    max_epochs = args.epochs
    
    load_dataset_surgery = importlib.machinery.SourceFileLoader('load_dataset','../../0.Preprocess/load_dataset.py').load_module()
    #import load_dataset as load_dataset_surgery
    train_X_surg, train_y_surg, train_cov_surg, test_X_surg, test_y_surg, test_cov_surg = load_dataset_surgery.load_dataset(views_to_get="siamese", pickle_file=args.datafile, contrast=args.contrast, split=args.split, get_cov=True)
    load_dataset_vur = importlib.machinery.SourceFileLoader('load_dataset','../../0.Preprocess/load_dataset_vur.py').load_module()
    #import load_dataset_vur
    train_X_reflux, train_y_reflux, train_cov_reflux, test_X_reflux, test_y_reflux, test_cov_reflux = load_dataset_vur.load_dataset(views_to_get="siamese", pickle_file="../../0.Preprocess/vur_images_256_20200422.pickle", image_dim=256, contrast=args.contrast, split=args.split, get_cov=True)
    
    load_dataset_func = importlib.machinery.SourceFileLoader('load_dataset','../../0.Preprocess/load_dataset_func.py').load_module()
    #import load_dataset_func
    train_X_func, train_y_func, train_cov_func, test_X_func, test_y_func, test_cov_func = load_dataset_func.load_dataset(views_to_get="siamese", pickle_file="../../0.Preprocess/func_images_256_20200422.pickle", image_dim=256,  contrast=args.contrast, split=args.split, get_cov=True)
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
    train_X = []
    train_y = []
    train_cov = []
    
    for i in range(len(train_cov_surg)):
        train_cov_surg[i] = "_".join(train_cov_surg[i].split("_")[:5])
    for i in range(len(train_cov_reflux)):
        item_ = train_cov_reflux[i].split("_")[:-1]
        item_[0], item_[1] = str(float(item_[0])), str(float(item_[1]))
        item_ = "_".join(item_)
        train_cov_reflux[i] = item_
        train_cov_reflux[i] = "_".join(train_cov_reflux[i].split("_")[:5])
    for i in range(len(train_cov_func)):
        train_cov_func[i] = "_".join(train_cov_func[i].split("_")[:5])
    

    dataset_train = {}
    for i in range(len(train_cov_surg)):
        trunc_id = "_".join(train_cov_surg[i].split("_")[:4]) 
        if trunc_id not in dataset_train:
            dataset_train[trunc_id] = {'left': None, 'right': None, 'labels': [-100, -100, -100, -100, -100]}
        if "Left" in train_cov_surg[i]:
            dataset_train[trunc_id]['left'], dataset_train[trunc_id]['labels'][0] = train_X_surg[i], train_y_surg[i]
        elif "Right" in train_cov_surg[i]:
            dataset_train[trunc_id]['right'], dataset_train[trunc_id]['labels'][1] = train_X_surg[i], train_y_surg[i]
    for i in range(len(train_cov_reflux)):
        trunc_id = "_".join(train_cov_reflux[i].split("_")[:4])
        if trunc_id not in dataset_train:
            dataset_train[trunc_id] = {'left': None, 'right': None, 'labels': [-100, -100, -100, -100, -100]}
        if "Left" in train_cov_reflux[i]:
            if dataset_train[trunc_id]['left'] is None: dataset_train[trunc_id]['left'] = train_X_reflux[i]
            dataset_train[trunc_id]['labels'][2] = train_y_reflux[i]
        elif "Right" in train_cov_reflux[i]:
            if dataset_train[trunc_id]['right'] is None: dataset_train[trunc_id]['right'] = train_X_reflux[i]
            dataset_train[trunc_id]['labels'][3] = train_y_reflux[i]
    for i in range(len(train_cov_func)):
        trunc_id = "_".join(train_cov_func[i].split("_")[:4])
        if trunc_id not in dataset_train:
            dataset_train[trunc_id] = {'left': None, 'right': None, 'labels': [-100, -100, -100, -100, -100]}
            dataset_train[trunc_id]['left'], dataset_train[trunc_id]['right'] = train_X_func[i][:2], train_X_func[i][2:]
        dataset_train[trunc_id]['labels'][4] = train_y_func[i]


    test_X = []
    test_y = []
    test_cov = []
 
    for i in range(len(test_cov_surg)):
        test_cov_surg[i] = "_".join(test_cov_surg[i].split("_")[:5])
    for i in range(len(test_cov_reflux)):
        item_ = test_cov_reflux[i].split("_")[:-1]
        item_[0], item_[1] = str(float(item_[0])), str(float(item_[1]))
        item_ = "_".join(item_)
        test_cov_reflux[i] = item_
        test_cov_reflux[i] = "_".join(test_cov_reflux[i].split("_")[:5])
    for i in range(len(test_cov_func)):
        test_cov_func[i] = "_".join(test_cov_func[i].split("_")[:5])


    dataset_test = {}
    for i in range(len(test_cov_surg)):
        trunc_id = "_".join(test_cov_surg[i].split("_")[:4])
        if trunc_id not in dataset_test:
            dataset_test[trunc_id] = {'left': None, 'right': None, 'labels': [-100, -100, -100, -100, -100]}
        if "Left" in test_cov_surg[i]:
            dataset_test[trunc_id]['left'], dataset_test[trunc_id]['labels'][0] = test_X_surg[i], test_y_surg[i]
        elif "Right" in test_cov_surg[i]:
            dataset_test[trunc_id]['right'], dataset_test[trunc_id]['labels'][1] = test_X_surg[i], test_y_surg[i]
    for i in range(len(test_cov_reflux)):
        trunc_id = "_".join(test_cov_reflux[i].split("_")[:4])
        if trunc_id not in dataset_test:
            dataset_test[trunc_id] = {'left': None, 'right': None, 'labels': [-100, -100, -100, -100, -100]}
        if "Left" in test_cov_reflux[i]:
            if dataset_test[trunc_id]['left'] is None: dataset_test[trunc_id]['left'] = test_X_reflux[i]
            dataset_test[trunc_id]['labels'][2] = test_y_reflux[i]
        elif "Right" in test_cov_reflux[i]:
            if dataset_test[trunc_id]['right'] is None: dataset_test[trunc_id]['right'] = test_X_reflux[i]
            dataset_test[trunc_id]['labels'][3] = test_y_reflux[i]
    for i in range(len(test_cov_func)):
        trunc_id = "_".join(test_cov_func[i].split("_")[:4])
        if trunc_id not in dataset_test:
            dataset_test[trunc_id] = {'left': None, 'right': None, 'labels': [-100, -100, -100, -100, -100]}
            dataset_test[trunc_id]['left'], dataset_test[trunc_id]['right'] = test_X_func[i][:2,:,:], test_X_func[i][2:,:,:]
        dataset_test[trunc_id]['labels'][4] = test_y_func[i]



    train(args, dataset_train, dataset_test, max_epochs)


if __name__ == '__main__':
    main()
