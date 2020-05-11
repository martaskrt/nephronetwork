from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
import importlib.machinery
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

# load data preprocessing and results postprocessing scripts
process_results = importlib.machinery.SourceFileLoader('process_results','../../2.Results/process_results.py').load_module()

SEED = 42

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
softmax = torch.nn.Softmax(dim=1)

# DATA LOADER
class KidneyDataset(torch.utils.data.Dataset):

    def __init__(self, X, y, cov):
        self.X = torch.from_numpy(X).float()
        self.y = y
        self.cov = cov

    def __getitem__(self, index):
        img, target, cov = self.X[index], self.y[index], self.cov[index]
        return img, target, cov

    def __len__(self):
        return len(self.X)


def train(args, train_X, train_y, train_cov, test_X, test_y, test_cov, best_epoch):
    from CNN_surgery import CNN
      
    hyperparams = {'lr': args.lr,
                   'momentum': args.momentum,
                   'weight_decay': args.weight_decay
                  }
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.num_workers}

    # initialize network
    if args.view == "siamese":
        net = CNN(num_inputs=2, output_dim=args.output_dim).to(device)
    else:
        net = CNN(num_inputs=1, output_dim=args.output_dim).to(device)
    net.zero_grad()
    if args.pretrain != "":
        pretrained_dict = torch.load(args.checkpoint)['model_state_dict']
        model_dict = net.state_dict()
        print("loading checkpoint..............")
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k not in ["fc6c.fc7.weight","fc6c.fc7.bias", "fc7_new.fc7.weight", "fc7_new.fc7.bias", "classifier_new.fc8.weight", "classifier_new.fc8.bias"]}
           
        # overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # load the new state dict
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

    test_set = KidneyDataset(test_X, test_y, test_cov)
    test_generator = DataLoader(test_set, **params)



    for epoch in range(best_epoch):
        # set up variables to store results
        accurate_labels_train, accurate_labels_test = 0, 0
        loss_accum_train, loss_accum_test = 0, 0

        all_targets_train, all_pred_prob_train, all_pred_label_train = [], [], []
        all_targets_test, all_pred_prob_test, all_pred_label_test = [], [], []

        patient_ID_train, patient_ID_test = [], []
        counter_train, counter_test = 0, 0

        # batch training
        net.train()
        for batch_idx, (data, target, cov) in enumerate(training_generator):
            optimizer.zero_grad()
            output = net(data.to(device))
            target = Variable(target.type(torch.LongTensor), requires_grad=False).to(device)
            if len(output.shape) == 1:
                output = output.unsqueeze(0)
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

            patient_ID_train.extend(cov)

        all_pred_prob_train = torch.cat(all_pred_prob_train)
        all_targets_train = torch.cat(all_targets_train)
        all_pred_label_train = torch.cat(all_pred_label_train)

        assert len(all_targets_train) == len(training_set)
        assert len(all_pred_prob_train) == len(training_set)
        assert len(all_pred_label_train) == len(training_set)
        assert len(patient_ID_train) == len(training_set)

        results_train = process_results.get_metrics(y_score=all_pred_prob_train.cpu().detach().numpy(),
                                                    y_true=all_targets_train.cpu().detach().numpy(),
                                                    y_pred=all_pred_label_train.cpu().detach().numpy())

        train_loss = loss_accum_train/counter_train
        print('Fold\t{}\tTrainEpoch\t{}\tLoss\t{:.6f}\t'
              'AUC\t{:.6f}\tAUPRC\t{:.6f}'.format(fold, epoch, train_loss, results_train['auc'], results_train['auprc']))

    # ********************************** #
    # best epoch is reached; run test set
    # ********************************** #
    net.test()
    with torch.no_grad():
        for batch_idx, (data, target, cov) in enumerate(test_generator):
            net.zero_grad()
            optimizer.zero_grad()
            output = net(data)
            target = target.type(torch.LongTensor).to(device)

            loss = F.cross_entropy(output, target)
            loss_accum_test += loss.item() * len(target)
            counter_test += len(target)
            output_softmax = softmax(output)

            accurate_labels_test += torch.sum(torch.argmax(output, dim=1) == target).cpu()

            pred_prob = output_softmax[:,1]
            pred_prob = pred_prob.squeeze()
            pred_label = torch.argmax(output, dim=1)

            assert len(pred_prob) == len(target)
            assert len(pred_label) == len(target)

            all_pred_prob_test.append(pred_prob)
            all_targets_test.append(target)
            all_pred_label_test.append(pred_label)

            patient_ID_test.extend(cov)

    all_pred_prob_test = torch.cat(all_pred_prob_test)
    all_targets_test = torch.cat(all_targets_test)
    all_pred_label_test = torch.cat(all_pred_label_test)

    assert len(all_targets_test) == len(test_set)
    assert len(all_pred_prob_test) == len(test_set)
    assert len(all_pred_label_test) == len(test_set)
    assert len(patient_ID_test) == len(test_set)

    results_test = process_results.get_metrics(y_score=all_pred_prob_test.cpu().detach().numpy(),
                                              y_true=all_targets_test.cpu().detach().numpy(),
                                              y_pred=all_pred_label_test.cpu().detach().numpy())
    test_loss = loss_accum_test / counter_test
    print('TestEpoch\t{}\tLoss\t{:.6f}\t'
          'AUC\t{:.6f}\tAUPRC\t{:.6f}'.format(epoch, test_loss, results_test['auc'], results_test['auprc']))


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
    path_to_checkpoint = args.dir + "/checkpoint_" + str(best_epoch) + '.pth'
    torch.save(checkpoint, path_to_checkpoint)

