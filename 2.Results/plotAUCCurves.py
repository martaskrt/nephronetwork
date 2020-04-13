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
local = True
model_name = "DebugSiameseCNNLstmDenseNet"

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
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

def test(args, train_X, train_y, train_cov, test_X, test_y, test_cov):
    if not local:
        process_results = importlib.machinery.SourceFileLoader('process_results',
                                                               '../../2.Results/process_results.py').load_module()
    else:
        process_results = importlib.machinery.SourceFileLoader('process_results',
                                                               '/Users/sulagshan/Documents/Thesis/nephronetwork/2.Results/process_results.py').load_module()
        sys.path.insert(0, '/Users/sulagshan/Documents/Thesis/nephronetwork/1.Models/siamese_network/')

    net = chooseNet(args)
    net.load_state_dict(torch.load(args.modelPath, map_location=device))

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

    test_set = KidneyDataset(test_X, test_y, test_cov)
    val_set_length = int(0.5*len(test_set))
    test_set_length = len(test_set) - val_set_length
    val_set, test_set = random_split(test_set, [val_set_length, test_set_length])

    test_generator = DataLoader(test_set, **params)
    val_generator = DataLoader(val_set, **params)
    print("Dataset generated")

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    accurate_labels_test = 0
    accurate_labels_val = 0

    loss_accum_test = 0
    loss_accum_val = 0

    all_targets_test = []
    all_pred_prob_test = []
    all_pred_label_test = []
    all_patient_ID_test = []

    all_targets_val = []
    all_pred_prob_val = []
    all_pred_label_val = []
    all_patient_ID_val = []

    counter_test = 0
    counter_val = 0

    net.eval()
    with torch.no_grad():
        for batch_idx, (data, target, cov) in enumerate(val_generator):
            net.zero_grad()
            optimizer.zero_grad()
            output = net(data)
            target = torch.tensor(target)
            target = target.type(torch.LongTensor).to(device)

            loss = F.cross_entropy(output, target)
            loss_accum_val += loss.item() * len(target)
            counter_val += len(target)
            output_softmax = softmax(output)
            accurate_labels_val += torch.sum(torch.argmax(output, dim=1) == target).cpu()

            pred_prob = output_softmax[:, 1]
            pred_label = torch.argmax(output, dim=1)

            assert len(pred_prob) == len(target)
            assert len(pred_label) == len(target)

            all_pred_prob_val.append(pred_prob)
            all_pred_label_val.append(pred_label)
            all_targets_val.append(target)
            all_patient_ID_val.append(cov)

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
            pred_label = torch.argmax(output, dim=1)

            assert len(pred_prob) == len(target)
            assert len(pred_label) == len(target)

            all_pred_prob_test.append(pred_prob)
            all_pred_label_test.append(pred_label)
            all_targets_test.append(target)
            all_patient_ID_test.append(cov)

    all_pred_prob_test_tensor = torch.cat(all_pred_prob_test)
    all_targets_test_tensor = torch.cat(all_targets_test)
    all_pred_label_test_tensor = torch.cat(all_pred_label_test)
    totalTestItems = sum(len(e[1]) for e in test_set)  # index 1 to count number of labels hence len(seq)
    assert len(all_pred_prob_test_tensor) == totalTestItems
    assert len(all_pred_label_test_tensor) == totalTestItems
    assert len(all_targets_test_tensor) == totalTestItems
    assert len(all_patient_ID_test) == test_set_length
    all_pred_prob_val_tensor = torch.cat(all_pred_prob_val)
    all_targets_val_tensor = torch.cat(all_targets_val)
    all_pred_label_val_tensor = torch.cat(all_pred_label_val)
    totalValItems = sum(len(e[1]) for e in val_set)
    assert len(all_pred_prob_val_tensor) == totalValItems
    assert len(all_pred_label_val_tensor) == totalValItems
    assert len(all_targets_val_tensor) == totalValItems
    assert len(all_patient_ID_val) == val_set_length

    results_test = process_results.get_metrics(y_score=all_pred_prob_test_tensor.cpu().detach().numpy(),
                                               y_true=all_targets_test_tensor.cpu().detach().numpy(),
                                               y_pred=all_pred_label_test_tensor.cpu().detach().numpy())
    results_val = process_results.get_metrics(y_score=all_pred_prob_val_tensor.cpu().detach().numpy(),
                                               y_true=all_targets_val_tensor.cpu().detach().numpy(),
                                               y_pred=all_pred_label_val_tensor.cpu().detach().numpy())

    # if (epoch <= 40 and (epoch % 5) == 0) or (epoch > 40 and epoch % 10 == 0) or epoch == args.stop_epoch:
    checkpoint = {'epoch': epoch,
                  'loss': loss,
                  'hyperparams': hyperparams,
                  'args': args,
                  'loss_test': loss_accum_test / counter_test,
                  'loss_val': loss_accum_val / counter_val,
                  'accuracy_test': int(accurate_labels_test) / counter_test,
                  'accuracy_val': int(accurate_labels_val) / counter_val,
                  'results_test': results_test,
                  'results_val': results_val,
                  'all_pred_label_test': [e.tolist() for e in all_pred_label_test],
                  'all_pred_prob_test': [e.tolist() for e in all_pred_prob_test],
                  'all_targets_test': [e.tolist() for e in all_targets_test],
                  'all_pred_label_val': [e.tolist() for e in all_pred_label_val],
                  'all_pred_prob_val': [e.tolist() for e in all_pred_prob_val],
                  'all_targets_val': [e.tolist() for e in all_targets_val],
                  'all_patient_ID_test': all_patient_ID_test,
                  'all_patient_ID_val': all_patient_ID_val,
                  }
    auc_info = {
        'test_tpr': results_test['tpr'],
        'test_fpr': results_test['fpr'],
        'test_auroc_thresholds': results_test['auroc_thresholds'],
        'val_tpr': results_val['tpr'],
        'val_fpr': results_val['fpr'],
        'val_auroc_thresholds': results_val['auroc_thresholds'],
        'test_recall': results_test['recall'],
        'test_precision': results_test['precision'],
        'test_auprc_thresholds': results_test['auprc_thresholds'],
        'val_recall': results_val['recall'],
        'val_precision': results_val['precision'],
        'val_auprc_thresholds': results_val['auprc_thresholds'],
    }

    plotAucCurves(auc_info)

def plotAucCurves(aucInfo):
    def plotAUROC(fpr, tpr):
        plt.plot(fpr, tpr)
        plt.show()

    def plotAUPRC(recall, precision):
        plt.plot(recall, precision)
        plt.show()

    plotAUROC(aucInfo['test_fpr'], aucInfo['test_tpr'])
    plotAUPRC(aucInfo['test_recall'], aucInfo['test_precision'])

    plotAUROC(aucInfo['val_fpr'], aucInfo['val_tpr'])
    plotAUPRC(aucInfo['val_recall'], aucInfo['val_precision'])


def chooseNet(args):
    if args.mvcnn:
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


def organizeDataForLstm(train_x, train_y, train_cov, test_x, test_y, test_cov):
    def sortData(t_x, t_y, t_cov):
        cov, x, y = zip(*sorted(zip(t_cov, t_x, t_y), key=lambda x: float(x[0].split("_")[0])))
        return x, y, cov

    def group(t_x, t_y, t_cov):
        x, y, cov = defaultdict(list), defaultdict(list), defaultdict(list)
        for i in range(len(t_cov)):
            id = t_cov[i].split("_")[0] + t_cov[i].split("_")[4]  # split data per kidney e.g 5.0Left, 5.0Right, 6.0Left, ...
            # id = t_cov[i].split("_")[0] # split only on id e.g. 5.0, 6.0, 7.0, ...
            x[id].append(t_x[i])
            y[id].append(t_y[i])
            cov[id].append(t_cov[i])
        # convert to np array
        organized_x = np.asarray([np.asarray(e) for e in list(x.values())])
        return organized_x, np.asarray(list(y.values())), np.asarray(list(cov.values()))

    def allowOnlyNVisits(t_x, t_y, t_cov, n=4):
        x, y, cov = [], [], []
        for i, e in enumerate(t_x):
            if len(e) == n:
                x.append(e)
                y.append(t_y[i])
                cov.append(t_cov[i])
        return np.asarray(x, dtype=np.float64), y, cov

    test_x, test_y, test_cov = sortData(test_x, test_y, test_cov)
    test_x, test_y, test_cov = group(test_x, test_y, test_cov)
    # test_x, test_y, test_cov = allowOnlyNVisits(test_x, test_y, test_cov)

    return None, None, None, test_x, test_y, test_cov


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
    return parser.parse_args()


def modifyArgs(args):
    args.batch_size = 1
    args.mvcnnSharedWeights = False
    args.SiameseCNNLstm = True
    args.mvcnn = False
    args.BichannelCNNLstmNet = False
    args.vgg_bn = True
    args.resnet18 = False
    args.densenet = False
    respath = "/Users/sulagshan/Documents/Thesis/results/"
    args.modelPath = respath + "model2020_04_12_22_16_02_siameseCNNLstm_vggbn_lr001_epoch17.pt"

def main():
    print(timestamp)
    args = parseArgs()
    modifyArgs(args)

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
        # load dummy data locally to test logic
        data_loader = importlib.machinery.SourceFileLoader("loadData", "/Users/sulagshan/Documents/Thesis/logs/loadData.py").load_module()
        train_X, train_y, train_cov, test_X, test_y, test_cov = data_loader.load()

    train_X, train_y, train_cov, test_X, test_y, test_cov = organizeDataForLstm(train_X, train_y, train_cov, test_X, test_y, test_cov)
    test(args, train_X, train_y, train_cov, test_X, test_y, test_cov)

if __name__ == '__main__':
    main()

# def parseData(datapath, wantEpoch):
#     f = open(datapath).read()
#     parsed = f.split("{")
#     dataStr = "{" + parsed[wantEpoch+1].split("}")[0] + "}"
#     replace = {"'": '"',
#         "0.": "0.0", "1.": "1.0",
#         "\n": "",
#         " ": "",
#         "array(": "",
#         ")": "",
#         ",dtype=float32": "",
#         ",...":""}
#     for k,v in replace.items():
#         dataStr = dataStr.replace(k,v)
#     data = json.loads(dataStr)
#     return data
#
# def plotAUROC(fpr, tpr, thresholds):
#     plt.plot(fpr, tpr)
#     plt.show()
#
# def plotAUPRC(precision, recall, thresholds):
#     plt.plot(recall, precision)
#     plt.show()
#
# def main():
#     wantEpoch = int(sys.argv[1])  # pass the epoch u want here
#     path = sys.argv[2]  # pass in auc data file
#     d = sys.argv[3] # pass in either test, or val; train does not currently work
#     # d == "train", "test", or "val"
#
#     data = parseData(path, wantEpoch)  # note data has thresholds if needed
#     fpr, tpr = data[d+"_"+"fpr"], data[d+"_"+"tpr"]
#     precision, recall = data[d+"_"+"precision"], data[d+"_"+"recall"]
#     print("Displaying AUROC plot")
#     plotAUROC(fpr, tpr, None)
#     print("Displaying AUPRC plot")
#     plotAUROC(precision, recall, None)
#
# if __name__ == '__main__':
#     main()