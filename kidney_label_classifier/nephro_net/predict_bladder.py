import pandas as pd
import torch
import numpy as np
from net import Net
import os
from data_loader import Dataset, get_dataloader


def pred_bladder(dataloaders, sets):
    for set in sets:
        loader = dls[set]
        # preds_list = []
        labels_list = []
        preds_auc_list = []
        img_dir_list = []

        for i, (image, label, img_dir) in enumerate(loader):
            if i % 100 == 0:
                print('\t' + str(i))
            # if i == 50:
                # break
            image = image.float().to(device)
            labels = label.float().to(device)  # .squeeze(1)
            # print(labels)
            with torch.no_grad():
                outputs_raw = model.forward(image)  # forward pass
                outputs_raw = torch.softmax(outputs_raw, dim=1)
            # we only want the predicted probabilities corresponding to the POS class
            preds_auc_numpy = outputs_raw.detach().cpu().numpy()[:, 1]
            # print(preds_auc_numpy)
            preds_auc_list.append(preds_auc_numpy)
            labels_numpy = labels.detach().cpu().numpy()
            labels_list.append(labels_numpy)
            # print(labels_numpy)
            # print(img_dir)
            img_dir_list.append(img_dir)
            # print(img_dir_list)
    # need to concatenate because these arrays are of size mb..
    df = pd.DataFrame({
        'img_dir': np.concatenate(img_dir_list),
        # 'labels': np.concatenate(labels_list),
        'preds' : np.concatenate(preds_auc_list)
    })
    return (df)

if __name__== "__main__":

    sets = ['all_images']
    task = 'bladder'
    mod = 'custom'
    batch_size = 64
    root_dir = '/home/delvinso/nephro/'
    manifest_path = os.path.join(root_dir, 'all_data', 'all_kidney_manifest.csv')
    out_dir = os.path.join(root_dir, 'output')

    model = Net(task=task, mod=mod)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
    print('Device: {}'.format(device))

    checkpoint_path = '/home/delvinso/nephro/output/bladder/custom_no_wts/_best.path.tar'
    model.eval().to(device)

    print('Loading Checkpoint Path: {}....'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    # initialize dataloader to retrieve all the images in the manifest (which was created by concatenating them together using R)
    print('Retrieving dataloader...')
    dls = get_dataloader(sets=sets,
                         root_dir=root_dir,
                         task=task,
                         manifest_path=manifest_path,
                         batch_size=batch_size,
                         return_pid = True)
    print('Making predictions...')
    df = pred_bladder(dataloaders=dls, sets=sets)

    out_path = os.path.join(out_dir, 'bladder_probs2.csv')
    df.to_csv(out_path)
    print('Done, results saved to {}'.format(out_path))