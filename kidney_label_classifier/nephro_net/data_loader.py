import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2
# NEED TO CLEAN UP


class Dataset(Dataset):
    def __init__(self, set, task, root_dir, manifest_path, transform=None, return_pid = False):
        """
        A pytorch DataSet for reading in the ultrasounds.

        :param task: (str) - one of : granular, view, or bladder
        :param set: (str) - one of: train, valid (and maybe set in the future)
        :param root_dir:  (str) - root directory, */nephro/
        :param manifest_path: (str) - path to manifest (absolute path preferred)
        :param transform: (torchvision transforms) - a list of transforms to apply to the DataSet when being fetched
        :param return_pid: (bool) to return the patient id associated with an image and label, or not
        """

        self.transform = transform
        self.set = set
        self.root_dir = root_dir
        self.manifest_path = manifest_path
        self.task = task
        self.image_dir = os.path.join(root_dir, 'data', 'imgs')
        self.return_pid = return_pid
        # should be ./nephro + data/imgs

        self.task_manifest = pd.read_csv(manifest_path)

        if self.set == "train":
            self.task_manifest = self.task_manifest[self.task_manifest['set'] == 'train']
            # print(self.task_manifest.shape)
        elif self.set == "valid":
            self.task_manifest = self.task_manifest[self.task_manifest['set'] == 'valid']
            # print(self.task_manifest.shape)
        elif self.set == 'all_images':
            print('MAKE SURE YOU ARE USING THE RIGHT MANIFEST FOR THIS!!!')
            self.task_manifest = self.task_manifest
        else:
            print('Set needs to be one of train, valid')

        # print(self.set)

    def __len__(self):
        return((self.task_manifest.shape[0]))

    def get_label(self, pid):
        # get label from manifest file
        pid = str(pid)

        # task = self.task
        if self.task == 'granular':
            label = self.task_manifest.numeric_image_label[self.task_manifest.image_ids == pid].squeeze(0)
        elif self.task == 'view':
            label = self.task_manifest.numeric_view_label[self.task_manifest.image_ids == pid].squeeze(0)
        elif self.task == 'bladder':
            label = self.task_manifest.numeric_bladder_label[self.task_manifest.image_ids == pid].squeeze(0)
        else:
            print('bad')
            label = torch.tensor(np.NaN)
        #     print('task must be one of granular, view or bladder pls')
        # print('Label: {}'.format(label))
        label = np.array(label)
        return(torch.tensor(label))
    # CELoss takes an integer for multi-class classification. Integer corresponds to the indice of the current class.

    def __getitem__(self, index):
        # might be easier to go from the image folder -> manifest
        # image folder index -> get pid -> get label from manifest -> read in
        # alternatively get pid from manifest -> check folder -> create absolute file directory -> read in

        pid = str(self.task_manifest.image_ids.iloc[index])
        # print(pid)
        y = self.get_label(pid)
        # print(y)
        if self.set == 'all_images':
            img_dir = os.path.join(self.root_dir, 'all_data', 'all_label_img', pid + '.jpg')
        else:
            img_dir = os.path.join(self.root_dir, 'data', 'imgs', pid + '.jpg')
        # print(img_dir)
        # read in as unsigned integer
        img = cv2.imread(img_dir).astype(np.uint8)
        # normalize the image
        # img = img / 255
        img = cv2.GaussianBlur(img, (5, 5), 0)
        if self.transform:
            img = self.transform(img)
        if not self.return_pid:
            return img, y
        else:
            pid = pid + '.jpg'
            return img, y, pid

def get_dataloader(sets, root_dir, manifest_path, task, batch_size=1, return_pid = False):
    """
    Takes a list of training sets (ie. any of 'train', 'valid', or 'test) and returns a dictionary of dataloaders.
    :param sets: (list) - list of any of  'train', 'valid', or 'test
    :param data_dir: (str) - where the nephro data is located
    :param task: (str) - one of : granular, view, bladder
    :param batch_size: (int) - mini batch size
    :return: (dict) of dataloaders
    """
    data_loaders = {}

    for set in ['train', 'valid', 'test', 'all_images']:  # test doesn't apply to MRNet but will keep in
        if set in sets:
            if set == 'train':
                ds = Dataset(set='train',  task = task, root_dir=root_dir, manifest_path = manifest_path, return_pid = return_pid,
                             transform=transforms.Compose([transforms.ToPILImage(),
                                                           #transforms.RandomHorizontalFlip(),  # default is 50%
                                                           #transforms.RandomAffine(25,  # rotation
                                                           #                        translate=(0.1, 0.1),
                                                           #                        shear = (-15, 15)),
                                                           transforms.ToTensor(),
                                                           ]))
                loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
            elif set == 'valid':
                ds = Dataset(set='valid', task = task, root_dir=root_dir,manifest_path = manifest_path, return_pid = return_pid,
                             transform=transforms.Compose([transforms.ToPILImage(),
                                                           transforms.ToTensor(),
                                                           ]))

                loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
            elif set == 'all_images':
                ds = Dataset(set='all_images', task = task, root_dir=root_dir,manifest_path = manifest_path, return_pid = return_pid,
                             transform=transforms.Compose([transforms.ToPILImage(),
                                                           transforms.ToTensor(),
                                                           ]))
                loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
            data_loaders[set] = loader
    return (data_loaders)
