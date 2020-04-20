# TODO: make this more robust. currently needs a lot of manual argument specifications to get what I want!


import pandas as pd
import torch
import numpy as np
from net import Net
import os
from data_loader import Dataset, get_dataloader

def extract_model_details(checkpoint_path):
    chkpt = checkpoint_path
    split_chk = chkpt.split('/')
    task = split_chk[5]
    mod = split_chk[6].split('_')[0]
    wts = split_chk[6].split('_', maxsplit=1)[-1]
    return task, mod, wts

def evaluate(checkpoint_path):

    chkpt = checkpoint_path

    task, mod, wts = extract_model_details(chkpt)

    out = task_d[task]  # number of output nodes
    classes = class_task[task] # the classes themselves
    in_f = model_d[mod]  # the number of fully connected layers prior to the classification layer, only used for alexnet and vgg

    print('Checkpoint {}'.format(checkpoint_path))
    print('\tModel: {}'.format(mod))
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
    batch_size = 32

    dls = get_dataloader(sets=sets,
                         root_dir=root_dir,
                         task=task,
                         manifest_path=manifest_path,
                         batch_size=batch_size,
                         return_pid = True)

    for set in sets:

        print('\tCurrent Set: {}'.format(set))

        loader = dls[set]
        preds_list = []
        labels_list = []
        pid_list = []

        for i, (image, label, pid) in enumerate(loader):
            if i % 100 == 0:
                print('\t' + str(i))
            image = image.float().to(device)
            labels = label.float().to(device)
            with torch.no_grad():
                outputs_raw = model.forward(image)  # forward pass

                # if type == 'probs':
                outputs = torch.softmax(outputs_raw, dim=1)
                # elif type == 'max':
                # outputs = torch.argmax(outputs_raw, dim=1)  # for binary classification i'm just taking the max as the pred.
                preds_numpy = outputs.detach().cpu().numpy()
                labels_numpy = labels.detach().cpu().numpy()
                preds_list.append(preds_numpy)
                labels_list.append(labels_numpy)
                pid_list.append(pid)


        # labels_list = np.concatenate(labels_list)
        # preds_list = np.concatenate(preds_list)
        #
        value_vars = np.array(range(0, out)) # class indices corresponding to the views, used for input into pd.melt
        print(value_vars)
        print('Length of Labels: {}\nLength of Predictions: {}'.format(np.concatenate(labels_list).shape, np.concatenate(preds_list).shape))
        df = pd.DataFrame(np.concatenate(preds_list))
        df['pid'] = np.concatenate(pid_list)
        df['labels'] = np.concatenate(labels_list)
        df = pd.melt(df, id_vars = ['pid', 'labels'], value_vars = value_vars, var_name = 'class', value_name = 'probs')
        df = df.sort_values(by = ['pid'])
        df['task'] = task
        df['mod'] = mod
        df['wts'] = wts
        df['set'] = set

        print(df)
        df.to_csv(os.path.join(out_dir, '_'.join([task, mod, wts, set]) + '_softmax.csv'))

    return (df)


if __name__ == "__main__":


    # ---- dictionaries ----

    task_d = {'view': 4, 'granular': 6, 'bladder': 2}
    model_d = {'alexnet': 256, 'vgg': 512, 'resnet': 2048, 'densenet': 1024, 'squeezenet': 512, 'custom': 512}
    class_task = {'view': ['Bladder', 'Other', 'Saggital', 'Transverse'],
                  'granular': ['Bladder', 'Other', 'Sag_R', 'Sag_L', 'Trans_Left', 'Trans_Right'],
                  'bladder': ['Other', 'Bladder']}

    # ---- directories for dataloaders (absolute paths) ----


    root_dir = '/home/delvinso/nephro/'
    manifest_path = '/home/delvinso/nephro/data/kidney_manifest.csv'
    out_dir = '/home/delvinso/nephro/output/preds'

    # find /home/delvinso/nephro/output/bladder /home/delvinso/nephro/output/granular /home/delvinso/nephro/output/view | grep '_best.path.tar'

    chkpts = ['/home/delvinso/nephro/output/bladder/custom_no_wts/_best.path.tar',
             '/home/delvinso/nephro/output/bladder/alexnet_no_wts/_best.path.tar',
             '/home/delvinso/nephro/output/granular/custom_no_wts/_best.path.tar',
             '/home/delvinso/nephro/output/granular/alexnet_no_wts/_best.path.tar',
                '/home/delvinso/nephro/output/granular/custom_wts/_best.path.tar',
             '/home/delvinso/nephro/output/granular/alexnet_wts/_best.path.tar']
    #

    # chkpts = ['/home/delvinso/nephro/output/view/custom_no_wts/_best.path.tar',]
              # '/home/delvinso/nephro/output/view/custom_wts/_best.path.tar',
              # '/home/delvinso/nephro/output/view/alexnet_wts/_best.path.tar',
              # '/home/delvinso/nephro/output/view/alexnet_no_wts/_best.path.tar']

    for checkpoint_path in chkpts:
        df = evaluate(checkpoint_path)


