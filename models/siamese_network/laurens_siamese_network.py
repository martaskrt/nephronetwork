# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:36:04 2019

@author: larun
"""

####
###         IMPORT LIBRARIES
####

import torch
import torch.nn as nn
import numpy as np
import argparse
import time
import sys



####
###         HELPER FUNCTIONS
####    

    ##
    ##        Model and training functions
    ##    

## SPLIT BATCHES
def split_batches(training_data, training_label, validation_data, validation_label,opt):
    
    n_samples = training_data.shape[0]
    perm = np.random.permutation(n_samples)
    
    ts_perm = training_data[perm,:,:,:]
    tl_perm = np.array(training_label)[perm]
    
    ts_batch = np.empty((0,opt.batch_size,training_data.shape[1],training_data.shape[2],training_data.shape[3]))
    tl_batch = np.empty((0,opt.batch_size))
    for k in range((n_samples)//opt.batch_size):
        ts_batch = np.append(ts_batch,[ts_perm[k:(k+opt.batch_size),:,:,:]],axis=0)
        tl_batch = np.append(tl_batch,[tl_perm[k:(k+opt.batch_size)]],axis=0)

    training_data = torch.from_numpy(ts_batch)
    training_label = torch.from_numpy(tl_batch)

    validation_data = torch.from_numpy(validation_data)    
    validation_label = torch.from_numpy(np.array(validation_label))
    
    if opt.gpu: 
        training_data = training_data.cuda()
        training_label = training_label.cuda()
    
        validation_data = validation_data.cuda()
        validation_label = validation_label.cuda()
    
    return training_data,training_label,validation_data,validation_label   


## CNN MODEL CLASS
class cnn(nn.Module):
    def __init__(self,opt):
        super(cnn,self).__init__()
        
        ### input data shape: batch, channels, img width, img height
        self.batch_size = opt.batch_size
        
        self.downconv1 = nn.Sequential(
                nn.Conv2d(opt.num_in_channels,opt.num_filters,opt.kernel_size,padding=2),
                nn.BatchNorm2d(opt.num_filters),
                nn.ReLU(),
                nn.MaxPool2d(2)) 
        self.downconv2 = nn.Sequential(
                nn.Conv2d(opt.num_filters,opt.num_filters*2,opt.kernel_size,padding=2),
                nn.BatchNorm2d(opt.num_filters*2),
                nn.ReLU(),
                nn.MaxPool2d(2))
        self.downconv3 = nn.Sequential(
                nn.Conv2d(opt.num_filters*2,opt.num_filters*4,opt.kernel_size,padding=2),
                nn.BatchNorm2d(opt.num_filters*4),
                nn.ReLU(),
                nn.MaxPool2d(2))
        self.linear1 = nn.Sequential(
                nn.Linear(32*opt.num_filters*4,32*opt.num_filters),
                nn.ReLU())
        self.linear2 = nn.Sequential(
                nn.Linear(32*opt.num_filters,32),
                nn.ReLU())
        self.linear3 = nn.Sequential(
                nn.Linear(32*opt.num_filters,1),
                nn.Softmax())        

    def forward(self,x):        
        self.out1 = self.downconv1(x)
        self.out2 = self.downconv2(self.out1)
        self.out3 = self.downconv3(self.out2)
        
        self.flat_out3 = self.out3.reshape(self.batch_size,-1,1,1)
        self.flat_out3 = torch.squeeze(self.flat_out3)
        
        self.out4 = self.linear1(self.flat_out3)
        self.out5 = self.linear2(self.out4)
        self.out_final = self.linear3(self.out5)
        
        return self.out_final

## SIAMESE CNN MODEL CLASS
class siamese_cnn(nn.Module):
    def __init__(self,opt):
        super(siamese_cnn,self).__init__()
        
        ### input data shape: batch, channels, img width, img height
        self.batch_size = opt.batch_size
        self.img_dim = opt.input_dim
        
        self.downconv1 = nn.Sequential(
                nn.Conv2d(opt.num_in_channels,opt.num_filters,opt.kernel_size,padding=2),
                nn.BatchNorm2d(opt.num_filters),
                nn.ReLU(),
                nn.MaxPool2d(2)) ## 128
        self.downconv2 = nn.Sequential(
                nn.Conv2d(opt.num_filters,opt.num_filters*2,opt.kernel_size,padding=2),
                nn.BatchNorm2d(opt.num_filters*2),
                nn.ReLU(),
                nn.MaxPool2d(2)) ## 64
        self.downconv3 = nn.Sequential(
                nn.Conv2d(opt.num_filters*2,opt.num_filters*4,opt.kernel_size,padding=2),
                nn.BatchNorm2d(opt.num_filters*4),
                nn.ReLU(),
                nn.MaxPool2d(2)) ## batch, 32, 
        
        in_dim = int((self.img_dim/8)*(self.img_dim/8)*opt.num_filters*4)
        self.linear1 = nn.Sequential( ## add batch norm here? 
                nn.Linear(in_dim,16*opt.num_filters), ## change these dimensions
                nn.ReLU())
        self.linear2 = nn.Sequential(
                nn.Linear(2*16*opt.num_filters,4*opt.num_filters),
                nn.ReLU())
        self.linear3 = nn.Sequential(
                nn.Linear(4*opt.num_filters,2),
                nn.Softmax())        

    def forward(self,data):
        
        for k in range(2):
            # print(self.batch_size)
            # input("That's your batch size. Now click enter")
            
            # print(data.shape)
            # input("Waiting...")
            
            x = data[:,k,:,:]
            x = torch.unsqueeze(x,1)
            # print(x.shape)
            # input("Waiting...")
            
            self.out1 = self.downconv1(x)
            self.out2 = self.downconv2(self.out1)
            self.out3 = self.downconv3(self.out2)
            
            #print("self.out3.shape: " + str(self.out3.shape))
            # print(self.out3.shape)
            # input("Waiting...")
            #flat_out3 = self.out3.view(-1,opt.kernel_size*opt.kernel_size*opt.num_filters*4)
            linear_out_dim = int((self.img_dim/8)*(self.img_dim/8)*opt.num_filters*4)
            flat_out3 = self.out3.reshape(self.batch_size,linear_out_dim,1,1)
            flat_out3 = torch.squeeze(flat_out3)
            
            out4 = self.linear1(flat_out3)
            # print(out4.shape)
            # input("out4 shape. Press enter.")
            
            if k == 0:                 
                combined_output = out4 
            else:
                combined_output = torch.cat((combined_output,out4),1)

            # print(combined_output.shape)
            # input("combined_output shape. Press enter.")

            
        out5 = self.linear2(combined_output)    
        out6 = self.linear3(out5)
        
        return out6
        
        
        
def train_model(training_dat,training_lab,valid_dat,valid_lab,opt):
    softmax = torch.nn.Softmax(dim=1)
    
    sys.path.insert(0,opt.process_results_dir)
    import process_results

    start = time.time()
    criterion = nn.CrossEntropyLoss()
    
    train_loss_list = []
    valid_loss_list = []
    
    if opt.siamese: ## 1 channel NN iterated 3 times 
        cnn_mod = siamese_cnn(opt)
        optimizer = torch.optim.Adam(cnn_mod.parameters(), lr=opt.learning_rate)
        
# =============================================================================
#         all_pred_prob = []
#         all_targets = []
#         all_pred_label = []
#         
# =============================================================================
        for epoch in range(opt.epochs):
            training_data, training_labels, validation_data, validation_labels = split_batches(training_dat,training_lab,valid_dat,valid_lab,opt)
            n_batches = training_data.shape[0]    

            cnn_mod.train()
            
            accurate_labels_val = 0
            all_labels = 0
            loss_accum = 0
            val_loss_accum = 0
            counter = 0 
            for  batch in range(n_batches):
                ## select batch input
                inputs = training_data[batch].float().to(opt.device)
                labels = training_labels[batch].type(torch.LongTensor).to(opt.device)
                
                ## forward pass
                output = cnn_mod(inputs)               
                loss = criterion(output,labels)
                train_loss_list.append(loss.item())

                loss_accum += loss.item() * len(labels)
                counter += len(labels)

                #load_dataset.view_images(inputs)
                #print("output: ")
                #print(output)
                #print("labels: ")
                #print(labels)
                
                softmax_output = softmax(output)
                pred_prob = softmax_output[:,1]
                pred_label = torch.argmax(softmax_output, dim=1)
                #print("pred labels: ")
                #print(pred_label)
    
                #pred_prob = pred_prob.squeeze()
                assert len(pred_prob) == len(labels)
                assert len(pred_label) == len(labels)
                
                if batch == 0:                    
                    all_pred_prob = pred_prob
                    all_targets = labels
                    all_pred_label = pred_label
                else:
                    all_pred_prob = torch.cat((all_pred_prob,pred_prob),0)
                    all_targets = torch.cat((all_targets,labels),0)
                    all_pred_label = torch.cat((all_pred_label,pred_label),0)

                accurate_labels_val += torch.sum(torch.argmax(output, dim=1) ==  labels).cpu()
                all_labels += len(labels)
              
                ## backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                    
 
            ## evaluate model in validation set 
            cnn_mod.eval() ## get validation output 
            val_inputs = validation_data.float().to(opt.device)
            val_labels = validation_labels.type(torch.LongTensor).to(opt.device)

            val_output = cnn_mod(val_inputs)
            val_target = val_labels.type(torch.LongTensor).to(opt.device)

            loss = criterion(val_output, val_target)
            val_loss_accum += loss.item() * len(val_target)

            val_loss = criterion(val_output,val_target)
            
            if epoch > 0:
                valid_loss_list = torch.cat((valid_loss_list,val_loss),0)
            else:
                valid_loss_list = val_loss

            time_elapsed = time.time() - start        
            print('Epoch [%d/%d], Train Loss: %.4f, Val loss: %.1f, Time(s): %d' % (
                    epoch+1, opt.epochs, loss, val_loss, time_elapsed))
            
            #print(all_pred_prob)
            #print(all_targets)
            #print(all_pred_label)
            
            assert len(all_pred_prob) == len(all_targets)
            assert len(all_pred_label) == len(all_targets)
            results = process_results.get_metrics(y_score=all_pred_prob.cpu().detach().numpy(),
                                                  y_true=all_targets.cpu().detach().numpy(),
                                                  y_pred=all_pred_label.cpu().detach().numpy())
            print('TrainEpoch\t{}\tACC\t{:.0f}%\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(epoch, 100.*accurate_labels_val/all_labels,
                                                                        loss_accum/counter, results['auc'],results['auprc'],
                                                                        results['tn'], results['fp'], results['fn'],
                                                                        results['tp']))
            # print("TRAIN" + '\t' + "AUC" + '\t' + str(results['auc']) + '\t' + "AUPRC" + '\t' + str(results['auprc']))
    
# =============================================================================
#             if ((epoch+1) % 10) == 0:
#                 checkpoint = {'epoch': epoch,
#                               'loss': loss,
#                               'hyperparams': hyperparams,
#                               'model_state_dict': cnn_mod.state_dict(),
#                               'optimizer': optimizer.state_dict()}
#                 if not os.path.isdir(args.dir):
#                     os.makedirs(args.dir)
#                 path_to_checkpoint = args.dir + '/' + "checkpoint_" + str(epoch) + '.pth'
#                 torch.save(checkpoint, path_to_checkpoint)
#         
# =============================================================================
    #else: ## 3 channel NN

    return cnn_mod
    
##
##        MAIN FUNCTION 
##    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pickle_file', default='/home/lauren/hydronephrosis/data/preprocessed_images_20190315.pickle', help="input directory of training triads")
    parser.add_argument('-load_dataset_dir', default='/home/lauren/hydronephrosis/preprocess/', help="input directory of test triads")
    parser.add_argument('-process_results_dir', default='/home/lauren/hydronephrosis/models/', help="input directory of test triads")
    parser.add_argument('-process_results_dir', default='/home/lauren/hydronephrosis/models/laurens-siamese-mod.pt', help="input directory of test triads")
    parser.add_argument('-device', default='cuda', help="cpu or cuda")
    parser.add_argument('-batch_size', default=32, help="Batch size")
    parser.add_argument('-learning_rate', default=0.0005, help="Learning rate")
    parser.add_argument('-siamese', default=True, help="Train siamese model vs not")
    parser.add_argument('-gpu', default=True, help="Train model using a gpu=True or no=False")
    parser.add_argument('-num_in_channels', default=1, help="Number of channels input, must be 1 if siamese, 3 if not siamese")
    parser.add_argument('-num_filters', default=6, help="Number of filters per convolutional layer")
    parser.add_argument('-kernel_size', default=5, help="Kernel size")
    parser.add_argument('-epochs', default=3, help="Kernel size")
    parser.add_argument('-input_dim', default=256, help="Kernel size")
    opt = parser.parse_args()
    
    ## Import custom modules
    sys.path.insert(0, opt.load_dataset_dir)
    import load_dataset

    ## Load data
    training_data, training_labels, val_data, val_labels = load_dataset.load_dataset(pickle_file=opt.pickle_file,contrast = 2,views_to_get="siamese")
    
    training_data, training_labels, validation_data, validation_labels = split_batches(training_data,training_labels,val_data, val_labels,opt)
    
    ## Run model
    out_mod = train_model(training_data, training_labels, validation_data, validation_labels,opt)

    torch.save(out_mod, PATH)


if __name__ == "__main__":
    main()
    
## 
##      DEBUGGING MAIN FUNCTION 
##

class options(object):
    pass
opt = options()
opt.pickle_file = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Image-analysis/preprocessed_images_20190315.pickle'
opt.load_dataset_dir = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/github/nephronetwork/preprocess/'
opt.process_results_dir = 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/github/nephronetwork/models/'
opt.batch_size = 64
opt.learning_rate = 0.0005
opt.siamese = True
opt.gpu = False
opt.num_in_channels = 1
opt.num_filters = 6
opt.kernel_size = 5
opt.epochs = 5
opt.input_dim = 256
opt.device = 'cpu' ## vs 'cuda'

## Import custom modules
sys.path.insert(0, opt.load_dataset_dir)
import load_dataset

## Load data
training_data, training_labels, val_data, val_labels = load_dataset.load_dataset(pickle_file=opt.pickle_file,contrast = 2,views_to_get="siamese")

## Splitting data into batches moved within training function

## Running model
out_mod = train_model(training_data, training_labels, val_data, val_labels,opt)

# load_dataset.view_images(training_data[0])
