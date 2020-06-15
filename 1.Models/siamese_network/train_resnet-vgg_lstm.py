import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Variable
from sklearn.utils import shuffle
import importlib.machinery
from collections import defaultdict
from datetime import datetime
import os

SEED = 42
debug = False
model_name = "Siamese_VGG_LSTM"

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
mainDir = "D:\\Sulagshan\\results\\"
resFile = mainDir+model_name+"_"+timestamp+"/lstmRes_"+timestamp+"_"+model_name+"_info.txt"
resFile2 = mainDir+model_name+"_"+timestamp+"/lstmRes_"+timestamp+"_"+model_name+"_auc.txt"
resFile3 = mainDir+model_name+"_"+timestamp+"/lstmRes_"+timestamp+"_"+model_name+"_summary.txt"
# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
softmax = torch.nn.Softmax(dim=1)

def modifyArgs(args):
    args.lr = 0.001
    args.batch_size = 1
    args.singleView = False
    args.mvcnnSharedWeights = False
    args.mvcnn = False
    args.SiameseCNNLstm = True
    args.BichannelCNNLstmNet = False
    args.vgg_bn = True
    args.resnet18 = False
    args.densenet = False
    args.customnet = False


def train(args, train_X, train_y, train_cov, test_X, test_y, test_cov, val_X, val_y, val_cov):
    # sys.path.insert(0, args.git_dir + '/nephronetwork/1.Models/siamese_network/')

    net = chooseNet(args)
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
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.num_workers}

    # Be careful to not shuffle order of image seq within a patient
    train_X, train_y, train_cov = shuffle(train_X, train_y, train_cov, random_state=SEED)
    if debug:
        train_X, train_y, train_cov = train_X[1:5], train_y[1:5], train_cov[1:5]
        test_X, test_y, test_cov =  test_X[1:5], test_y[1:5], test_cov[1:5]
        if val_X:
            val_X, val_y, val_cov = val_X[1:5], val_y[1:5], val_cov[1:5]

    training_set = KidneyDataset(train_X, train_y, train_cov)
    test_set = KidneyDataset(test_X, test_y, test_cov)
    val_set = KidneyDataset(val_X, val_y, val_cov)
    # val_set_length = int(0.5*len(test_set))
    # test_set_length = len(test_set) - val_set_length
    # val_set, test_set = random_split(test_set, [val_set_length, test_set_length])

    training_generator = DataLoader(training_set, **params)
    test_generator = DataLoader(test_set, **params)
    val_generator = DataLoader(val_set, **params)
    print("Dataset generated")

    if debug:
        # prevents concurrent processing, to allow for stepping debug
        training_generator.num_workers = 0
        test_generator.num_workers = 0
        val_generator.num_workers = 0

    for epoch in range(args.stop_epoch + 1):
        print("Epoch " + str(epoch) + " started.")
        res = Results(epoch, train_y, test_y, val_y)

        net.train()
        for batch_idx, (data, target, cov) in enumerate(training_generator):
            optimizer.zero_grad()
            output = net(data.to(device))
            target = torch.tensor(target)  # change target to: tensor([0,1,0,1]) from: [tensor([0]), tensor([1]), tensor([0]), tensor([1])]
            target = Variable(target.type(torch.LongTensor), requires_grad=False).to(device)
            loss = F.cross_entropy(output, target)
            res.loss_accum_train += loss.item() * len(target)
            loss.backward()
            res.accurate_labels_train += torch.sum(torch.argmax(output, dim=1) == target).cpu()
            optimizer.step()
            res.counter_train += len(target)
            output_softmax = softmax(output)
            pred_prob = output_softmax[:, 1]
            pred_label = torch.argmax(output_softmax, dim=1)
            assert len(pred_prob) == len(target)
            assert len(pred_label) == len(target)
            res.all_pred_prob_train.append(pred_prob)
            res.all_pred_label_train.append(pred_label)
            res.all_targets_train.append(target)
            res.all_patient_ID_train.append(cov)

        net.eval()
        with torch.no_grad():
            for batch_idx, (data, target, cov) in enumerate(val_generator):
                net.zero_grad()
                optimizer.zero_grad()
                output = net(data)
                target = torch.tensor(target)
                target = target.type(torch.LongTensor).to(device)
                loss = F.cross_entropy(output, target)
                res.loss_accum_val += loss.item() * len(target)
                res.counter_val += len(target)
                output_softmax = softmax(output)
                res.accurate_labels_val += torch.sum(torch.argmax(output, dim=1) == target).cpu()
                pred_prob = output_softmax[:, 1]
                pred_label = torch.argmax(output, dim=1)
                assert len(pred_prob) == len(target)
                assert len(pred_label) == len(target)
                res.all_pred_prob_val.append(pred_prob)
                res.all_pred_label_val.append(pred_label)
                res.all_targets_val.append(target)
                res.all_patient_ID_val.append(cov)

            for batch_idx, (data, target, cov) in enumerate(test_generator):
                net.zero_grad()
                optimizer.zero_grad()
                output = net(data)
                target = torch.tensor(target)
                target = target.type(torch.LongTensor).to(device)
                loss = F.cross_entropy(output, target)
                res.loss_accum_test += loss.item() * len(target)
                res.counter_test += len(target)
                output_softmax = softmax(output)
                res.accurate_labels_test += torch.sum(torch.argmax(output, dim=1) == target).cpu()
                pred_prob = output_softmax[:, 1]
                pred_label = torch.argmax(output, dim=1)
                assert len(pred_prob) == len(target)
                assert len(pred_label) == len(target)
                res.all_pred_prob_test.append(pred_prob)
                res.all_pred_label_test.append(pred_label)
                res.all_targets_test.append(target)
                res.all_patient_ID_test.append(cov)

        res.processResults(train_y, test_y, val_y, args, hyperparams, net, optimizer)

def main():
    print(timestamp)
    args = parseArgs()
    modifyArgs(args)

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

    makeResultFiles()
    train_X, train_y, train_cov, test_X, test_y, test_cov = organizeDataForLstm(train_X, train_y, train_cov, test_X, test_y, test_cov)
    train_X, train_y, train_cov, val_X, val_y, val_cov = makeValSet(train_X, train_y, train_cov)
    train(args, train_X, train_y, train_cov, test_X, test_y, test_cov, val_X, val_y, val_cov)

def makeValSet(train_X, train_y, train_cov):
    def split(l):
        # return (l[int(len(l)*0.21):], l[:int(len(l)*0.21)])
        return (l[:-1], l[-1:])
    train_X, val_X = split(train_X)
    train_y, val_y = split(train_y)
    train_cov, val_cov = split(train_cov)
    return train_X, train_y, train_cov, val_X, val_y, val_cov


# def makeValSet(test_X, test_y, test_cov):
#     def split(l):
#         return (l[:len(l)//2], l[len(l)//2:])
#     test_X, val_X = split(test_X)
#     test_y, val_y = split(test_y)
#     test_cov, val_cov = split(test_cov)
#     return test_X, test_y, test_cov, val_X, val_y, val_cov

class Results:
    def __init__(self, epoch, train, test, val):
        self.epoch = epoch
        self.accurate_labels_train = 0
        self.accurate_labels_test = 0
        self.accurate_labels_val = 0

        self.loss_accum_train = 0
        self.loss_accum_test = 0
        self.loss_accum_val = 0

        self.all_targets_train = []
        self.all_pred_prob_train = []
        self.all_pred_label_train = []
        self.all_patient_ID_train = []

        self.all_targets_test = []
        self.all_pred_prob_test = []
        self.all_pred_label_test = []
        self.all_patient_ID_test = []

        self.all_targets_val = []
        self.all_pred_prob_val = []
        self.all_pred_label_val = []
        self.all_patient_ID_val = []

        self.counter_train = 0
        self.counter_test = 0
        self.counter_val = 0

        self.totalTrainItems = sum(len(e) for e in train)
        self.totalTestItems = sum(len(e) for e in test) # todo: fix so we pass in test_y; need to move val test slppit into data loader
        self.totalValItems = sum(len(e) for e in val)

    def processResults(self, train_y, test_y, val_y, args, hyperparams, net, optimizer):
        self.concatResults()
        self.verifyLength(train_y, test_y, val_y)
        self.store(args, hyperparams, net, optimizer)

    def concatResults(self):
        self.all_pred_prob_train_tensor = torch.cat(self.all_pred_prob_train)
        self.all_targets_train_tensor = torch.cat(self.all_targets_train)
        self.all_pred_label_train_tensor = torch.cat(self.all_pred_label_train)
        self.all_pred_prob_test_tensor = torch.cat(self.all_pred_prob_test)
        self.all_targets_test_tensor = torch.cat(self.all_targets_test)
        self.all_pred_label_test_tensor = torch.cat(self.all_pred_label_test)
        self.all_pred_prob_val_tensor = torch.cat(self.all_pred_prob_val)
        self.all_targets_val_tensor = torch.cat(self.all_targets_val)
        self.all_pred_label_val_tensor = torch.cat(self.all_pred_label_val)

    def verifyLength(self, train, test, val):
        # todo: clean this up
        assert len(self.all_pred_prob_train_tensor) == self.totalTrainItems
        assert len(self.all_pred_label_train_tensor) == self.totalTrainItems
        assert len(self.all_targets_train_tensor) == self.totalTrainItems
        # assert len(self.all_patient_ID_train) == len(train_y)  # todo: correct this
        assert len(self.all_pred_prob_test_tensor) == self.totalTestItems
        assert len(self.all_pred_label_test_tensor) == self.totalTestItems
        assert len(self.all_targets_test_tensor) == self.totalTestItems
        # assert len(self.all_patient_ID_test) == test_set_length  # todo: corect this
        assert len(self.all_pred_prob_val_tensor) == self.totalValItems
        assert len(self.all_pred_label_val_tensor) == self.totalValItems
        assert len(self.all_targets_val_tensor) == self.totalValItems
        # assert len(self.all_patient_ID_val) == val_set_length  # todo: correct this

    def store(self, args, hyperparams, net, optimizer):
        process_results = importlib.machinery.SourceFileLoader('process_results',
                                                               '../../2.Results/process_results.py').load_module()
        results_train = process_results.get_metrics(y_score=self.all_pred_prob_train_tensor.cpu().detach().numpy(),
                                                    y_true=self.all_targets_train_tensor.cpu().detach().numpy(),
                                                    y_pred=self.all_pred_label_train_tensor.cpu().detach().numpy())
        resTrainStr = 'TrainEpoch,{},ACC,{:.6f},Loss,{:.6f},AUC,{:.6f},AUPRC,{:.6f},TN,{},FP,{},FN,{},TP,{}'.format(
            self.epoch,
            int(self.accurate_labels_train) / self.counter_train,
            self.loss_accum_train / self.counter_train,
            results_train['auc'],
            results_train['auprc'],
            results_train['tn'],
            results_train['fp'],
            results_train['fn'],
            results_train['tp'])
        results_test = process_results.get_metrics(y_score=self.all_pred_prob_test_tensor.cpu().detach().numpy(),
                                                   y_true=self.all_targets_test_tensor.cpu().detach().numpy(),
                                                   y_pred=self.all_pred_label_test_tensor.cpu().detach().numpy())
        resTestStr = 'TestEpoch ,{},ACC,{:.6f},Loss,{:.6f},AUC,{:.6f},AUPRC,{:.6f},TN,{},FP,{},FN,{},TP,{}'.format(
            self.epoch,
            int(self.accurate_labels_test) / self.counter_test,
            self.loss_accum_test / self.counter_test,
            results_test['auc'],
            results_test['auprc'],
            results_test['tn'],
            results_test['fp'],
            results_test['fn'],
            results_test['tp'])
        results_val = process_results.get_metrics(y_score=self.all_pred_prob_val_tensor.cpu().detach().numpy(),
                                                  y_true=self.all_targets_val_tensor.cpu().detach().numpy(),
                                                  y_pred=self.all_pred_label_val_tensor.cpu().detach().numpy())
        if not self.counter_val:
            self.counter_val = float('inf')
        resValStr = 'ValEpoch  ,{},ACC,{:.6f},Loss,{:.6f},AUC,{:.6f},AUPRC,{:.6f},TN,{},FP,{},FN,{},TP,{}'.format(
            self.epoch,
            int(self.accurate_labels_val) / self.counter_val,
            self.loss_accum_val / self.counter_val,
            results_val['auc'],
            results_val['auprc'],
            results_val['tn'],
            results_val['fp'],
            results_val['fn'],
            results_val['tp'])

        # if (epoch <= 40 and (epoch % 5) == 0) or (epoch > 40 and epoch % 10 == 0) or epoch == args.stop_epoch:
        # loss, hyperparams, args,
        checkpoint = {'epoch': self.epoch,
                      # 'loss': loss,
                      'hyperparams': hyperparams,
                      'args': args,
                      # 'model_state_dict': net.state_dict(),
                      # 'optimizer': optimizer.state_dict(),
                      'loss_train': self.loss_accum_train / self.counter_train,
                      'loss_test': self.loss_accum_test / self.counter_test,
                      'loss_val': self.loss_accum_val / self.counter_val,
                      'accuracy_train': int(self.accurate_labels_train) / self.counter_train,
                      'accuracy_test': int(self.accurate_labels_test) / self.counter_test,
                      'accuracy_val': int(self.accurate_labels_val) / self.counter_val,
                      'results_train': results_train,
                      'results_test': results_test,
                      'results_val': results_val,
                      'all_pred_label_train': [e.tolist() for e in self.all_pred_label_train],
                      'all_pred_prob_train': [e.tolist() for e in self.all_pred_prob_train],
                      'all_targets_train': [e.tolist() for e in self.all_targets_train],
                      'all_pred_label_test': [e.tolist() for e in self.all_pred_label_test],
                      'all_pred_prob_test': [e.tolist() for e in self.all_pred_prob_test],
                      'all_targets_test': [e.tolist() for e in self.all_targets_test],
                      'all_pred_label_val': [e.tolist() for e in self.all_pred_label_val],
                      'all_pred_prob_val': [e.tolist() for e in self.all_pred_prob_val],
                      'all_targets_val': [e.tolist() for e in self.all_targets_val],
                      'all_patient_ID_train': self.all_patient_ID_train,
                      'all_patient_ID_test': self.all_patient_ID_test,
                      'all_patient_ID_val': self.all_patient_ID_val,
                      }
        auc_info = {
            'train_tpr': results_train['tpr'],
            'train_fpr': results_train['fpr'],
            'train_auroc_thresholds': results_train['auroc_thresholds'],
            'test_tpr': results_test['tpr'],
            'test_fpr': results_test['fpr'],
            'test_auroc_thresholds': results_test['auroc_thresholds'],
            'val_tpr': results_val['tpr'],
            'val_fpr': results_val['fpr'],
            'val_auroc_thresholds': results_val['auroc_thresholds'],
            'train_recall': results_train['recall'],
            'train_precision': results_train['precision'],
            'train_auprc_thresholds': results_train['auprc_thresholds'],
            'test_recall': results_test['recall'],
            'test_precision': results_test['precision'],
            'test_auprc_thresholds': results_test['auprc_thresholds'],
            'val_recall': results_val['recall'],
            'val_precision': results_val['precision'],
            'val_auprc_thresholds': results_val['auprc_thresholds'],
        }

        f = open(resFile, "a")
        f.write("checkpoint" + "\n" + str(self.epoch) + "\n")
        f.close()

        f = open(resFile2, "a")
        f.write("checkpoint" + "\n" + str(self.epoch) + "\n")
        f.write(str(auc_info) + "\n") # todo: fix
        f.close()

        f = open(resFile3, "a")
        f.write("EpochNumber " + str(self.epoch) + "\n")
        f.write(resTrainStr + "\n")
        f.write(resTestStr + "\n")
        f.write(resValStr + "\n")
        f.close()

        modelPath = mainDir + model_name + "_" + timestamp + "/model" + timestamp + "_" + model_name + "_epoch" + str(
            self.epoch) + ".pt"
        optimizerPath = mainDir + model_name + "_" + timestamp + "/optimizer" + timestamp + "_" + model_name + "_epoch" + str(
            self.epoch) + ".pt"
        torch.save(net.state_dict(), modelPath)
        torch.save(optimizer.state_dict(), optimizerPath)

class KidneyDataset(torch.utils.data.Dataset):

    def __init__(self, X, y, cov):
        # self.X = [torch.tensor(e, requires_grad=True).float() for e in X]
        self.X = [torch.from_numpy(e).float() for e in X]
        # self.X = torch.from_numpy(X).float()
        self.y = y
        self.cov = cov

    def __getitem__(self, index):
        imgs, target, cov = self.X[index], self.y[index], self.cov[index]
        return imgs, target, cov

    def __len__(self):
        return len(self.X)

def makeResultFiles():
    os.mkdir(mainDir + model_name + "_" + timestamp)
    f = open(resFile, "x")  # create the file using "x" arg
    f.write("Description:"+model_name+"\n") # write description to the first line here
    f.close()
    f = open(resFile2, "x")  # create the file using "x" arg
    f.write("Description:"+model_name+"\n") # write description to the first line here
    f.close()
    f = open(resFile3, "x")  # create the file using "x" arg
    f.write("Description:"+model_name+" summarized results"+"\n") # write description to the first line here
    f.close()

def organizeDataForLstm(train_x, train_y, train_cov, test_x, test_y, test_cov):
    def sortData(t_x, t_y, t_cov):
        train_cov, train_x, train_y = zip(*sorted(zip(t_cov, t_x, t_y), key=lambda x: float(x[0].split("_")[0])))
        return train_x, train_y, train_cov

    def group(t_x, t_y, t_cov):
        x, y, cov = defaultdict(list), defaultdict(list), defaultdict(list)
        for i in range(len(t_cov)):
            id = t_cov[i].split("_")[0] + t_cov[i].split("_")[
                4]  # split data per kidney e.g 5.0Left, 5.0Right, 6.0Left, ...
            # id = t_cov[i].split("_")[0] # split only on id e.g. 5.0, 6.0, 7.0, ...
            x[id].append(t_x[i])
            y[id].append(t_y[i])
            cov[id].append(t_cov[i])
        # convert to np array
        organized_train_x = np.asarray([np.asarray(e) for e in list(x.values())])
        return organized_train_x, np.asarray(list(y.values())), np.asarray(list(cov.values()))

    def allowOnlyNVisits(t_x, t_y, t_cov, n=4):
        x, y, cov = [], [], []
        for i, e in enumerate(t_x):
            if len(e) == n:
                x.append(e)
                y.append(t_y[i])
                cov.append(t_cov[i])
        return np.asarray(x, dtype=np.float64), y, cov

    train_x, train_y, train_cov = sortData(train_x, train_y, train_cov)
    train_x, train_y, train_cov = group(train_x, train_y, train_cov)
    # train_x, train_y, train_cov = allowOnlyNVisits(train_x, train_y, train_cov)
    test_x, test_y, test_cov = sortData(test_x, test_y, test_cov)
    test_x, test_y, test_cov = group(test_x, test_y, test_cov)
    # test_x, test_y, test_cov = allowOnlyNVisits(test_x, test_y, test_cov)

    return train_x, train_y, train_cov, test_x, test_y, test_cov

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=100, type=int, help="Number of epochs")
    parser.add_argument('--batch_size', default=16, type=int, help="Batch size")
    parser.add_argument('--lr', default=0.005, type=float, help="Learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, help="Momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay")
    parser.add_argument("--num_workers", default=1, type=int, help="Number of CPU workers")
    parser.add_argument("--contrast", default=1, type=int, help="Image contrast to train on")
    parser.add_argument("--split", default=0.7, type=float, help="proportion of dataset to use as training")
    parser.add_argument("--bottom_cut", default=0.0, type=float, help="proportion of dataset to cut from bottom")
    parser.add_argument("--crop", default=0, type=int, help="Crop setting (0=big, 1=tight)")
    parser.add_argument("--output_dim", default=128, type=int, help="output dim for last linear layer")
    parser.add_argument("--git_dir", default="C:/Users/Lauren/Desktop/DS Core/Projects/Urology/")
    parser.add_argument("--stop_epoch", default=100, type=int,
                        help="If not running cross validation, which epoch to finish with")
    parser.add_argument("--cv_stop_epoch", default=18, type=int, help="get a pth file from a specific epoch")
    parser.add_argument("--view", default="siamese", help="siamese, sag, trans")
    parser.add_argument("--dir", default="./", help="Directory to save model checkpoints to")
    parser.add_argument("--checkpoint", default="", help="Path to load pretrained model checkpoint from")
    parser.add_argument("--etiology", default="B", help="O (obstruction), R (reflux), B (both)")
    parser.add_argument('--cv', action='store_true', help="Flag to run cross validation")
    parser.add_argument('--adam', action="store_true", help="Use Adam optimizer instead of SGD")
    parser.add_argument('--vgg', action='store_true', help="Run VGG16 architecture, not using this flag runs ResNet")
    parser.add_argument('--vgg_bn', action='store_true', help="Run VGG16 batch norm architecture")
    parser.add_argument('--densenet', action='store_true', help="Run DenseNet")
    parser.add_argument('--resnet18', action='store_true',
                        help="Run ResNet18 architecture, not using this flag runs ResNet16")
    parser.add_argument('--resnet50', action='store_true',
                        help="Run ResNet50 architecture, not using this flag runs ResNet50")
    parser.add_argument('--pretrained', action="store_true",
                        help="Use pretrained model with cross validation if cv requested")
    # parser.add_argument('--unet',         action="store_true", help="UNet architecthure")
    return parser.parse_args()

def chooseNet(args):
    num_inputs = 1 if args.view != "siamese" else 2
    model_pretrain = args.pretrained if args.cv else False
    if args.singleView:
        # from VGGResNetSiameseLSTM import MVCNNLstmNet1
        from VGGResNetSiameseLSTM import SingleCNNLSTM
        if args.densenet:
            print("importing SingleCNNLSTM densenet")
            return SingleCNNLSTM("densenet").to(device)
        elif args.resnet18:
            print("importing SingleCNNLSTM resnet18")
            return SingleCNNLSTM("resnet18").to(device)
        elif args.resnet50:
            print("importing SingleCNNLSTM resnet50")
            return SingleCNNLSTM("resnet50").to(device)
        elif args.vgg:
            print("importing SingleCNNLSTM vgg")
            return SingleCNNLSTM("vgg").to(device)
        elif args.vgg_bn:
            print("importing SingleCNNLSTM vgg_bn")
            return SingleCNNLSTM("vgg_bn").to(device)
        elif args.customnet:
            print("importing SingleCNNLSTM customNet")
            return SingleCNNLSTM("custom").to(device)
    elif args.mvcnn:
        # from VGGResNetSiameseLSTM import MVCNNLstmNet1
        from VGGResNetSiameseLSTM import MVCNNLstmNet2
        if args.densenet:
            print("importing MVCNNLstmNet2 densenet")
            return MVCNNLstmNet2("densenet", args.mvcnnSharedWeights).to(device)
        elif args.resnet18:
            print("importing MVCNNLstmNet2 resnet18")
            return MVCNNLstmNet2("resnet18", args.mvcnnSharedWeights).to(device)
        elif args.resnet50:
            print("importing MVCNNLstmNet2 resnet50")
            return MVCNNLstmNet2("resnet50", args.mvcnnSharedWeights).to(device)
        elif args.vgg:
            print("importing MVCNNLstmNet2 vgg")
            return MVCNNLstmNet2("vgg", args.mvcnnSharedWeights).to(device)
        elif args.vgg_bn:
            print("importing MVCNNLstmNet2 vgg_bn")
            return MVCNNLstmNet2("vgg_bn", args.mvcnnSharedWeights).to(device)
        elif args.customnet:
            print("importing MVCNNLstmNet2 customNet")
            return MVCNNLstmNet2("custom").to(device)
    elif args.BichannelCNNLstmNet:
        from VGGResNetSiameseLSTM import BichannelCNNLstmNet
        if args.densenet:
            print("importing BichannelCNNLstmNet densenet")
            return BichannelCNNLstmNet("densenet").to(device)
        elif args.resnet18:
            print("importing BichannelCNNLstmNet resnet18")
            return BichannelCNNLstmNet("resnet18").to(device)
        elif args.resnet50:
            print("importing BichannelCNNLstmNet resnet50")
            return BichannelCNNLstmNet("resnet50").to(device)
        elif args.vgg:
            print("importing BichannelCNNLstmNet vgg")
            return BichannelCNNLstmNet("vgg").to(device)
        elif args.vgg_bn:
            print("importing BichannelCNNLstmNet vgg_bn")
            return BichannelCNNLstmNet("vgg_bn").to(device)
        elif args.customnet:
            print("importing BichannelCNNLstmNet customNet")
            return BichannelCNNLstmNet("custom").to(device)
    elif args.SiameseCNNLstm:
        from VGGResNetSiameseLSTM import SiameseCNNLstm
        if args.densenet:
            print("importing SiameseCNNLstm densenet")
            return SiameseCNNLstm("densenet").to(device)
        elif args.resnet18:
            print("importing SiameseCNNLstm resnet18")
            return SiameseCNNLstm("resnet18").to(device)
        elif args.resnet50:
            print("importing SiameseCNNLstm resnet50")
            return SiameseCNNLstm("resnet50").to(device)
        elif args.vgg:
            print("importing SiameseCNNLstm vgg")
            return SiameseCNNLstm("vgg").to(device)
        elif args.vgg_bn:
            print("importing SiameseCNNLstm vgg_bn")
            return SiameseCNNLstm("vgg_bn").to(device)
        elif args.customnet:
            print("importing SiameseCNNLstm customNet")
            return SiameseCNNLstm("custom").to(device)

# Additional helper functions (can be used if needed)
def getId(cov):
    return cov.split("_")[0]

def getHandedness(cov):
    return cov.split("_")[4]

if __name__ == '__main__':
    main()