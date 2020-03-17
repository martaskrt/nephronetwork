import logging
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable

def run_model(model, loader, optimizer, criterion, metrics_every_iter = False, train = True):
    """
    Executes a full pass over the training or validation data, ie. an epoch, calculating the metrics of interest

    :param model: (torch.nn.module) - network architecture to train
    :param loader: (DataLoader) - torch DataLoader
    :param optimizer: (optimizer) - an optimizer to be passed in
    :param metrics_every_iter: (int) - evaluate metrics every ith iteration
    :param train: (bool) - if True then training mode, else evaluation mode (no gradient evaluation, backprop, weight updates, etc)
    :param criterion (torch.nn.modules.loss) - loss function
    :returns: epoch loss, auc, list of predictions and labels
    """

    # use cpu or cuda depending on availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(model)
    print('There are {} GPUs available for use. '.format(torch.cuda.device_count()))
    if train:
        model.train().to(device)
    else:
        model.eval().to(device)

    # initialize lists for keeping track of predictions and labels
    preds_list = []
    labels_list = []
    # total_epoch_loss = 0
    ith_loss = 0            # keeps track of loss every ith iterations for metrics
                            # resets to 0 every ith iteration
    total_running_loss = 0.0 # keeps track of total loss throughout the epoch
                           # divided by the # of batches at the end of an epoch to obtain an epoch's avg loss
    num_batches = 1        # total n will be 1130 in MRNet's case, must start at 1 for tensorboard logging else the first step will repeat 2x

    for image, label in loader:

        # send X,Y and weights (if applicable) to device - only really used for gpu
        image = image.to(device)
        labels = label.to(device)#.squeeze(1)
        # wts = wts.float().to(device)

        #  ----- a single iteration of forward and backward pass ----

        if train:
            optimizer.zero_grad()                       # zero gradients

        outputs = model.forward(image)                # forward pass
        # print(labels)print(outputs)
        # outputs = torch.sigmoid(outputs)
        loss = criterion(outputs, labels)               # compute loss
        total_running_loss += loss.item()
        ith_loss += loss.item()

        if train:
            loss.backward()                             # backprop
            optimizer.step()                            # update parameters

        # extract data from torch Variable, move to cpu, converting to numpy arrays
        # https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/train.py

        preds_numpy = outputs.detach().cpu().numpy()[0][0]
        labels_numpy = labels.detach().cpu().numpy()[0]#[0]
        preds_list.append(preds_numpy)
        labels_list.append(labels_numpy)
        # print(preds_list); print(labels_list)
        num_batches += 1

        # if indicated, will print and (eventually) output running auc and average batch loss
        if metrics_every_iter:
            if num_batches % metrics_every_iter == 0:
                # fpr, tpr, threshold = metrics.balanced_accuracy_score(labels_list, preds_list)
                # auc = metrics.auc(fpr, tpr)
                # logging.info('\t{} Batch(es)\tAUC: {:.3f}\t{} Average Batch Loss:{:.3f}'.\
                # # print('{} Batch(es)\n\tAUC: {:.3f}\n\tRunning Average Loss:{:.3f}\n\tAccuracy:{:.3f}'.\
                #       format(str(num_batches), (auc), (metrics_every_iter),  (ith_loss/metrics_every_iter)))

                # bacc = metrics.balanced_accuracy_score(labels_list, preds_list)
                # logging.info('\t{} Batch(es)\tBalanced Accuracy: {:.3f}\t{} Average Batch Loss:{:.3f}'.\
                # # print('{} Batch(es)\n\tAUC: {:.3f}\n\tRunning Average Loss:{:.3f}\n\tAccuracy:{:.3f}'.\
                #       format(str(num_batches), (bacc), (metrics_every_iter),  (ith_loss/metrics_every_iter)))

                logging.info('\t{} Batch(es)\t\t{} Average Batch Loss:{:.3f}'.\
                # print('{} Batch(es)\n\tAUC: {:.3f}\n\tRunning Average Loss:{:.3f}\n\tAccuracy:{:.3f}'.\
                      format(str(num_batches), (metrics_every_iter),  (ith_loss/metrics_every_iter)))

                # tensorflow
                # with writer.as_default():
                #     tf.summary.scalar('Iter_Average_Loss/train_iter', ith_loss/metrics_every_iter, num_batches * epoch)
                #     tf.summary.scalar('Running_AUC/train_iter', auc, num_batches * epoch)
                ith_loss = 0 # reset loss every ith iterations since this keeps track of the ith iteration

    # epoch-level metrics
    epoch_mean_loss = total_running_loss/len(loader)
    # fpr, tpr, threshold = metrics.roc_curve(labels_list, preds_list)
    # epoch_auc = metrics.auc(fpr, tpr)
    # epoch_acc = metrics.balanced_accuracy_score(y_true=labels_list, y_pred=preds_list)

    # return epoch_mean_loss, epoch_acc, preds_list, labels_list
    return epoch_mean_loss, preds_list, labels_list

