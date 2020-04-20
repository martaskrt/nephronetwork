import logging
import shutil
import os
import torch
import pandas as pd
import numpy as np
# utilities for main, such as logging and checkpointing

# heavily adapted from https://github.com/cs230-stanford/cs230-code-examples/blob/96ac6fd7dd72831a42f534e65182471230007bee/pytorch/vision/utils.py#L63

def save_checkpoint(state, is_best, checkpoint_dir):
    """
    Saves model parameters at checkpoint directory (the model_outdir).

    :param state:  (dict)
    :param is_best:  (bool)
    :param checkpoint_dir: (str)
    :return:
    """
    # if len(str(epoch)) < 2:
    #     epoch = '0' + str(epoch)
    # fn = epoch + '_last.pth.tar'
    filepath = os.path.join(checkpoint_dir, 'last.pth.tar')

    if not os.path.exists(checkpoint_dir):
        print("Checkpoint Directory does not exist, Making directory {}".format(checkpoint_dir))
        os.mkdir(checkpoint_dir)
    # else:
        # print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        # fn = epoch + '_best.path.tar'
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, '_best.path.tar'))

def load_checkpoint(checkpoint, model,  optimizer = None):
    """
    Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    # defaults to whatever is on the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))

    checkpoint = torch.load(checkpoint,
                            map_location = torch.device(device))
    # load model state..
    model.load_state_dict(checkpoint['state_dict'])

    # load optimizer...
    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])
    # if scheduler:
    #     scheduler.load_state_dict(checkpoint['scheduler_dict'])
    logging.info(print('States for Epoch {} loaded...'.format(checkpoint['epoch'])))
    # epochs = epochs - checkpoint['epoch']

# did not edit from CS230
def set_logger(log_path):
    """
    Initialize logger so it will log in both console and `log_path`. Will be output into `model_dir/train.log`
    :param log_path: where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path) # overwrite existing
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def get_class_weights(manifest_dir, task='view', method = 'max'):
    """
    Given nephro root directory, uses the manifest to calculate positive class weights.
    Weights are calculated as either the ratio of each class to the smallest class or the larger class.
    Thus the loss is weighted more heavily towards classes that are larger in the former and weighted less towards smaller classes in the latter.

    :param manifest_dir: (str) - directory of where manifest is located
    :return: Class weights scaled by the smallest class
    """
    mn = pd.read_csv(manifest_dir)
    print(task) # for debugging
    if task == 'view':
        class_cts = mn[mn.set == 'train'].numeric_view_label.value_counts().sort_index(axis = 0)
    elif task == 'granular':
        class_cts = mn[mn.set == 'train'].numeric_image_label.value_counts().sort_index(axis = 0)
    elif task == 'bladder':
        class_cts = mn[mn.set == 'train'].numeric_bladder_label.value_counts().sort_index(axis = 0)
    else:
        print('Must be one of view, granular or bladder pls!')

    if method == 'min':
        class_props = class_cts / (np.min(class_cts))
    elif method == 'max':
        class_props = class_cts / (np.max(class_cts))

    return(torch.tensor(class_props))


# get_weights(root_dir = '/Users/delvin/Downloads/MRNet-v1.0/data/', task = 'abnormal', view = 'axial')
# get_class_weights("nephro_test\\data\\kidney_manifest.csv", 'granular')

