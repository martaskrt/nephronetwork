import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import importlib.machinery
import numpy as np
from SiameseNetwork import SiamNet
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

process_results = importlib.machinery.SourceFileLoader('process_results','../process_results.py').load_module()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# device = torch.device('cpu')
print(device)
softmax = torch.nn.Softmax(dim=1)

# Set the random seed manually for reproducibility.
SEED = 2516
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

def train(args, train_dataset, val_dataset, test_dataset):
    net = SiamNet(classes=10, num_inputs = 1)
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     net = nn.DataParallel(net)
    net.to(device)
    if args.checkpoint != "":
        pretrained_dict = torch.load(args.checkpoint)
        model_dict = net.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        #pretrained_dict['fc6.fc6_s1.weight'] = pretrained_dict['fc6.fc6_s1.weight'].view(1024, 256, 3, 3)
        pretrained_dict['fc6.fc6_s1.weight'] = pretrained_dict['fc6.fc6_s1.weight'].view(1024, 256, 2, 2)
        for k,v in model_dict.items():
            if k not in pretrained_dict:
                pretrained_dict[k] = model_dict[k]
        # pretrained_dict['fc6b.conv6b_s1.weight'] = model_dict['fc6b.conv6b_s1.weight']
        # pretrained_dict['fc6b.conv6b_s1.bias'] = model_dict['fc6b.conv6b_s1.bias']
        # pretrained_dict['fc6c.fc7.weight'] = model_dict['fc6c.fc7.weight']
        # pretrained_dict['fc6c.fc7.bias'] = model_dict['fc6c.fc7.bias']
        # pretrained_dict['fc7_new.fc7.weight'] = model_dict['fc7_new.fc7.weight']
        # pretrained_dict['fc7_new.fc7.bias'] = model_dict['fc7_new.fc7.bias']
        # pretrained_dict['classifier_new.fc8.weight'] = model_dict['classifier_new.fc8.weight']
        # pretrained_dict['classifier_new.fc8.bias'] = model_dict['classifier_new.fc8.bias']
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
    # optimizer = torch.optim.SGD(net.parameters(), lr=hyperparams['lr'], momentum=hyperparams['momentum'],
    #                             weight_decay=hyperparams['weight_decay'])

    optimizer = torch.optim.Adam(net.parameters(), lr=hyperparams['lr'], weight_decay=hyperparams['weight_decay'])

    for epoch in range(args.epochs):
        all_labels = 0
        accurate_labels = 0
        loss_accum = 0
        counter =0
        all_targets = []
        all_pred_prob = []
        all_pred_label = []
        for batch_idx, (data, target) in enumerate(tqdm(train_dataset)):
            optimizer.zero_grad()
            inp = data.to(device)
            output = net(inp)
            # print("Outside: input size", inp.size(),
            #       "output_size", output.size())
            target = Variable(target.type(torch.LongTensor), requires_grad=False).to(device)

            loss = F.cross_entropy(output, target)
            loss_accum += loss.item() * len(target)
            counter += len(target)
            loss.backward()

            accurate_labels += torch.sum(torch.argmax(output, dim=1) == target).cpu()
            optimizer.step()
            all_labels += len(target)

            output_softmax = softmax(output)
            pred_prob = output_softmax[:, 1]
            pred_label = torch.argmax(output_softmax, dim=1)

            #pred_prob = pred_prob.squeeze()
            assert len(pred_prob) == len(target)
            assert len(pred_label) == len(target)
            all_pred_prob.append(pred_prob)
            all_targets.append(target)
            all_pred_label.append(pred_label)
            # results = process_results.get_metrics(y_score=pred_prob.cpu().detach().numpy(),
            #                                       y_true=target.cpu().detach().numpy())

        all_pred_prob = torch.cat(all_pred_prob)
        all_targets = torch.cat(all_targets)
        all_pred_label = torch.cat(all_pred_label)
        assert len(all_pred_prob) == len(all_targets)
        assert len(all_pred_label) == len(all_targets)
        # results = process_results.get_metrics(y_score=all_pred_prob.cpu().detach().numpy(),
        #                                       y_true=all_targets.cpu().detach().numpy(),
        #                                       y_pred=all_pred_label.cpu().detach().numpy())
        print('TrainEpoch\t{}\tACC\t{:.0f}%\tLoss\t{:.6f}\t'.format(epoch, 100.*accurate_labels/all_labels,
                                                                    loss_accum/counter))
        # print("TRAIN" + '\t' + "AUC" + '\t' + str(results['auc']) + '\t' + "AUPRC" + '\t' + str(results['auprc']))

        if ((epoch+1) % 2) == 0:
            checkpoint = {'epoch': epoch,
                          'loss': loss,
                          'hyperparams': hyperparams,
                          'model_state_dict': net.state_dict(),
                          'optimizer': optimizer.state_dict()}
            if not os.path.isdir(args.dir):
                os.makedirs(args.dir)
            path_to_checkpoint = args.dir + '/' + "checkpoint_mnist_" + str(epoch) + '.pth'
            torch.save(checkpoint, path_to_checkpoint)


        with torch.set_grad_enabled(False):
            accurate_labels_val = 0
            all_labels_val = 0
            loss_accum = 0
            counter = 0
            all_targets = []
            all_pred_prob = []
            all_pred_label = []
            accurate_labels = 0
            for batch_idx, (data, target) in enumerate(val_dataset):
                net.zero_grad()
                optimizer.zero_grad()
                output = net(data)
                target = target.type(torch.LongTensor).to(device)

                loss = F.cross_entropy(output, target)
                loss_accum += loss.item() * len(target)
                counter += len(target)
                output_softmax = softmax(output)

                accurate_labels_val += torch.sum(torch.argmax(output, dim=1) == target).cpu()
                all_labels_val += len(target)

                pred_prob = output_softmax[:,1]
                pred_prob = pred_prob.squeeze()
                pred_label = torch.argmax(output, dim=1)

                all_pred_prob.append(pred_prob)
                all_targets.append(target)
                all_pred_label.append(pred_label)

                assert pred_prob.shape == target.shape
            # results = process_results.get_metrics(y_score=pred_prob.cpu().numpy(), y_true=target.cpu().numpy())
            # print("VAL.............AUC: " + str(results['auc']) + " | AUPRC: " + str(results['auprc']))
            #
            #
            # accuracy = 100. * accurate_labels_val / all_labels_val
            # print('Test accuracy: {}/{} ({:.3f}%)\tLoss: {:.6f}'.format(accurate_labels_val, all_labels_val, accuracy, loss_accum/counter))

            all_pred_prob = torch.cat(all_pred_prob)
            all_targets = torch.cat(all_targets)
            all_pred_label = torch.cat(all_pred_label)
            # results = process_results.get_metrics(y_score=all_pred_prob.cpu().detach().numpy(),
            #                                       y_true=all_targets.cpu().detach().numpy(),
            #                                       y_pred=all_pred_label.cpu().detach().numpy())
            print('ValEpoch\t{}\tACC\t{:.0f}%\tLoss\t{:.6f}'.format(epoch, 100. * accurate_labels_val / all_labels_val,
                                                                        loss_accum / counter))


def load_data(args):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor()
        ])
    mnist_trainset = datasets.MNIST(root='../../data', train=True, download=True, transform=transform)
    mnist_testset = datasets.MNIST(root='../../data', train=False, download=True, transform=transform)

    train_dl = torch.utils.data.DataLoader(
        mnist_trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
        )
    test_dl = torch.utils.data.DataLoader(
        mnist_testset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
        )

    print("Training Set Size:   %d" % len(mnist_trainset))
    print("Test Set Size:       %d" % len(mnist_testset))

    return train_dl, test_dl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=50, type=int, help="Number of epochs")
    parser.add_argument('--batch_size', default=128, type=int, help="Batch size")
    parser.add_argument('--lr', default=0.001, type=float, help="Learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, help="Momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay")
    parser.add_argument("--num_workers", default=6, type=int, help="Number of CPU workers")
    parser.add_argument("--dir", default="./", help="Directory to save model checkpoints to")
    parser.add_argument("--contrast", default=0, type=int, help="Image contrast to train on")
    parser.add_argument("--checkpoint", default="", help="Path to load pretrained model checkpoint from")
    parser.add_argument("--data_dir", default="../../data/kermany2018")


    args = parser.parse_args()

    train_gen, test_gen = load_data(args)

    train(args, train_gen, test_gen, test_gen)


if __name__ == '__main__':
    main()