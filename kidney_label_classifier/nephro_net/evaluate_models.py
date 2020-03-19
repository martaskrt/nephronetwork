# TODO: make this more robust. currently needs a lot of manual argument specifications to get what I want!


import pandas as pd
import torch
import numpy as np
from net import Net
from matplotlib import pyplot as plt
import os
from data_loader import Dataset, get_dataloader
from sklearn import metrics
import seaborn as sns


# def conf_mat_helper(preds, labels):
#     cm = confusion_matrix(preds, labels)
#     df_cm = pd.DataFrame(cm, index = [i for i in classes],
#                          columns = [i for i in classes])
def auprc_helper(labels, preds):
    
    pr, rc, threshold = metrics.precision_recall_curve(y_true = labels, probas_pred= preds)

    ap = metrics.average_precision_score(y_true = labels, y_scores = preds)


def auroc_helper(labels, preds):
    """
    Given a list of ground truth labels and the predicted probabilities from a classifier, calculate the AUROC and plot it.
    :param labels: (list) ground truth labels
    :param preds:  (list) list of probabilities corresponding to y = 1
    :return:
    """
    fpr, tpr, threshold = metrics.roc_curve(labels, preds)
    df = pd.DataFrame({'fpr': fpr,
                      'tpr': tpr,
                      'threshold': threshold})
    # uncomment for the best model to save
    # df.to_csv('/home/delvinso/nephro/output/all_custom_thresholds.csv', index = False)
    epoch_auc = metrics.auc(fpr, tpr)
    print('AUC: {}'.format(epoch_auc))

    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    f = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkred',
             lw=lw, label='ROC curve (area = %0.3f)' % epoch_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity (FPR)')
    plt.ylabel('Sensitivity (TPR)')
    plt.title('ROC - Bladder vs Other')
    plt.legend(loc="lower right")
    return f

def extract_model_details(checkpoint_path):
    chkpt = checkpoint_path
    split_chk = chkpt.split('/')
    task = split_chk[5]
    mod = split_chk[6].split('_')[0]
    wts = split_chk[6].split('_', maxsplit = 1)[-1]
    return task, mod, wts

def evaluate(checkpoint_path):
    list_holder = []
    chkpt = checkpoint_path


    task, mod, wts = extract_model_details(chkpt)

    out = task_d[task]
    classes = class_task[task]
    in_f = model_d[mod]  # the number of fully connected layers prior to the classification layer

    print('Model: {}'.format(mod))
    print('\tCurrent Task: {}'.format(task))
    print('\t# of Classes: {}, {} '.format(out, classes))
    # if fail then error?
    model = Net(task=task, mod=mod)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
    print('\tDevice: {}'.format(device))

    if not mod == 'custom':
        print('\t# of Features input to Classifier Layer: {}'.format(in_f))
        # only matters for vgg and alexnet
        model.clf_on = torch.nn.Linear(in_f, out)

    model.eval().to(device)

    print('\tLoading Checkpoint Path: {}....'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    sets = ['valid']
    batch_size = 16

    dls = get_dataloader(sets=sets,
                         root_dir=root_dir,
                         task=task,
                         manifest_path=manifest_path,
                         batch_size=batch_size)

    # sets = ['valid']
    for set in sets:

        print('\tCurrent Set: {}'.format(set))

        loader = dls[set]
        preds_list = []
        labels_list = []
        preds_auc_list = []

        for i, (image, label) in enumerate(loader):
            if i % 100 == 0:
                print('\t' + str(i))
            image = image.float().to(device)
            labels = label.float().to(device)
            with torch.no_grad():
                outputs_raw = model.forward(image)  # forward pass

                # if type == 'probs':
                # outputs = torch.softmax(outputs_raw, dim = 1)
                # elif type == 'max':
                outputs = torch.argmax(outputs_raw, dim=1)  # for binary classification i'm just taking the max as the pred.
                # print(outputs)
            preds_numpy = outputs.detach().cpu().numpy()
            labels_numpy = labels.detach().cpu().numpy()
            # print(preds_numpy)
            # print(labels_numpy)
            preds_list.append(preds_numpy)
            labels_list.append(labels_numpy)

            if task == 'bladder':
                # compute probabilities to compute AUC if we are dealing with binary classification (bladder vs other)
                outputs_raw = torch.softmax(outputs_raw, dim=1)
                # we want the probability that the observation is in the positive (bladder) class
                # for each training example, output is a 1x2 tensor
                preds_auc_numpy = outputs_raw.detach().cpu().numpy()[:, 1]
                # print(preds_auc_numpy)
                preds_auc_list.append(preds_auc_numpy)

        labels_list = np.concatenate(labels_list)
        preds_list = np.concatenate(preds_list)
        print('Length of Labels: {}\nLength of Predictions: {}'.format(len(labels_list), len(preds_list)))
        list_holder.append({'model': mod, 'task': task,
                            'labels': labels_list, 'wts': wts,
                            'preds': preds_list, 'set': set})

        if task == 'bladder':
            preds_auc_list = np.concatenate(preds_auc_list)
            print('Length of Predictions for AUC: {}'.format(len(preds_auc_list)))
            auc_path = os.path.join(fig_dir, '_'.join([task, mod, wts, set]) + '_auc.png')  # + '.png')
            f = auroc_helper(labels = labels_list, preds = preds_auc_list)
            # print(f)
            plt.savefig(auc_path)
            plt.close(f)
            print('\tAUC Plot saved to: ' + auc_path)

            df_auc = pd.DataFrame({'labels':labels_list, 'preds':preds_auc_list})
            df_auc.to_csv(os.path.join(out_dir, '_'.join([task, mod, wts, set]) + '.csv'))

        df = pd.DataFrame(list_holder)


    return(df)


if __name__== "__main__":

    # ---- initialize all the possible options ----

    # tasks =  (['bladder', 'granular'])#, 'view'])#, 'granular'])#, 'view', 'granular')
    # # models = ('vgg', 'alexnet', 'densenet', 'resnet', 'squeezenet' ,'custom')
    # models = (['custom', 'alexnet'])
    # # weights = ('wts', 'no_wts')
    # weights = (['no_wts'])

    # ---- dictionaries ----

    task_d = {'view': 4, 'granular': 6, 'bladder': 2}
    model_d = {'alexnet': 256, 'vgg': 512, 'resnet': 2048, 'densenet': 1024, 'squeezenet': 512, 'custom': 512}
    class_task = {'view': ['Bladder', 'Other', 'Saggital', 'Transverse'],
                  'granular': ['Bladder', 'Other', 'Sag_L', 'Sag_R', 'Trans_Left', 'Trans_Right'],
                  'bladder': ['Other', 'Bladder']}

    # ---- directories for dataloaders (absolute paths) ----
    # colab
    # root_dir = "nephro_test/"
    # manifest_path = "/content/nephro_test/data/kidney_manifest.csv"
    # out_dir = "/content/drive/My Drive/nephro_test/output/"

    root_dir = '/home/delvinso/nephro/'
    manifest_path = '/home/delvinso/nephro/data/kidney_manifest.csv'
    out_dir = '/home/delvinso/nephro/output'

    fig_dir = os.path.join(out_dir, 'figures')

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # find /home/delvinso/nephro/output/bladder /home/delvinso/nephro/output/granular /home/delvinso/nephro/output/view | grep '_best.path.tar'

    # chkpts = ['/home/delvinso/nephro/output/bladder/custom_no_wts/_best.path.tar',
    #          '/home/delvinso/nephro/output/bladder/alexnet_no_wts/_best.path.tar',
    #          '/home/delvinso/nephro/output/granular/custom_no_wts/_best.path.tar',
    #          '/home/delvinso/nephro/output/granular/alexnet_no_wts/_best.path.tar']
    #
    # chkpts = ['/home/delvinso/nephro/output/granular/custom_wts/_best.path.tar',
    #          '/home/delvinso/nephro/output/granular/alexnet_wts/_best.path.tar']
    #

    chkpts = ['/home/delvinso/nephro/output/view/custom_no_wts/_best.path.tar',
              '/home/delvinso/nephro/output/view/custom_wts/_best.path.tar',
              '/home/delvinso/nephro/output/view/alexnet_wts/_best.path.tar',
              '/home/delvinso/nephro/output/view/alexnet_no_wts/_best.path.tar']

    for checkpoint_path in chkpts:
        df = evaluate(checkpoint_path)

        print(df)
        for index, row in df.iterrows():

            current_model = df.model[index]
            current_task = df.task[index]
            current_wts = df.wts[index]
            current_set = df.set[index]
            classes = class_task[current_task]

            params_id = os.path.join(fig_dir, '_'.join([current_task, current_model, current_wts, current_set]))

            cm = metrics.confusion_matrix(y_pred = df.preds[index], y_true = df.labels[index])
            print(cm)
            print(np.diag(cm)/ np.sum(cm, axis = 1))

            df_cm = pd.DataFrame(cm, index=[i for i in classes],
                                 columns=[i for i in classes])

            title = 'Validation Task: {} Model: {} Weights: {}'.format(current_task, current_model, current_wts,
                                                                       current_set)
            # ---- save confusion matrix to disk ----
            plt.figure(figsize=(12, 9))
            ax = sns.heatmap(df_cm, annot=True, fmt='d', cmap='viridis',
                             annot_kws={"size": 16})
            ax.set(xlabel='Predictions', ylabel='Labels',
                   title=title)
            # plt.show()
            figure = ax.get_figure()
            figure.savefig(params_id + '.png')
            print('\tConfusion Matrix saved to: ' + params_id)

            # ---- generate classification report -----

            # calculate accuracy from the confusion matrix - this is actually the same as the multi-class PPV?
            # df_acc = pd.DataFrame({'class': classes, 'acc':np.diag(cm)/ np.sum(cm, axis = 1)}).set_index('class')
            # df_acc.to_csv(params_id + '_classification_report.csv')
            # print('\tAccuracy Report saved to: ' + params_id + '_accuracy_report.csv')
            
            
            dict_cr = metrics.classification_report(df.preds[index], df.labels[index],
                                            target_names=classes, output_dict=True)

            df_cr = pd.DataFrame(dict_cr).transpose()

            # df_cr = df_cr.join(df_acc) # join the sklearn classification report w/ the multi-class accuracy ? this might be confusing...
            
            
            # ---- save classification report to disk -----
            df_cr.to_csv(params_id + '_classification_report.csv')
            print('\tClassification Report saved to: ' + params_id + '_classification_report.csv')


