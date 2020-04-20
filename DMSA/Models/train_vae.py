from argparse import ArgumentParser
from importlib import import_module
from math import ceil
from os import replace
from os.path import exists, join
from shutil import copy
from sys import stderr

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

from datasets import load_dataset
from train_utils import extend_batch, get_validation_iwae
from VAE import VAE
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
import random
from PIL import Image
from dmsa_model_v0 import *

import wandb


# class MyDataParallel(torch.nn.DataParallel):
# 	def __init__(self, model, device_ids):
# 		super(MyDataParallel, self).__init__(model, device_ids)

# 	def __getattr__(self, name):
# 		try:
# 			return super(MyDataParallel, self).__getattr__(name)
# 		except AttributeError:
# 			return getattr(self.module, name)



parser = ArgumentParser(description='Train VAE')

parser.add_argument('--model_dir', type=str, action='store', required=True,
					help='Directory with model.py. ' +
						 'It must be a directory in the root ' +
						 'of this repository. ' +
						 'The checkpoints are saved ' +
						 'in this directory as well. ' +
						 'If there are already checkpoints ' +
						 'in the directory, the training procedure ' +
						 'is resumed from the last checkpoint ' +
						 '(last_checkpoint.tar).')

parser.add_argument('--epochs', type=int, action='store', required=True,
					help='Number epochs to train VAEAC.')
parser.add_argument('--lr', type=float, action='store')
parser.add_argument('--beta', type=float, action='store')

parser.add_argument('--exp', type=str, action='store')



# parser.add_argument('--train_dataset', type=str, action='store',
#                     required=True,
#                     help='Dataset of images for training VAEAC to inpaint ' +
#                          '(see load_datasets function in datasets.py).')

# parser.add_argument('--validation_dataset', type=str, action='store',
#                     required=True,
#                     help='Dataset of validation images for VAEAC ' +
#                          'log-likelihood IWAE estimate ' +
#                          '(see load_datasets function in datasets.py).')

parser.add_argument('--validation_iwae_num_samples', type=int, action='store',
					default=10,
					help='Number of samples per object to estimate IWAE ' +
						 'on the validation set. Default: 25.')

parser.add_argument('--validations_per_epoch', type=int, action='store',
					default=1,
					help='Number of IWAE estimations on the validation set ' +
						 'per one epoch on the training set. Default: 5.')

args = parser.parse_args()

wandb.init(project='Hydro VAE', dir="/data/alexchang", name=args.exp)

# Default parameters which are not supposed to be changed from user interface
use_cuda = torch.cuda.is_available()
print("cuda available:", use_cuda)
verbose = True
# Non-zero number of workers cause nasty warnings because of some bug in
# multiprocess library. It might be fixed now, so maybe it is time to set it
# to the number of CPU cores in the system.
num_workers = 32

# import the module with the model networks definitions,
# optimization settings, and a mask generator
model_module = import_module(args.model_dir + '.model')


# import mask generator
# mask_generator = model_module.mask_generator


# build VAEAC on top of the imported networks
model = VAE(
	model_module.reconstruction_log_prob,
	model_module.prior_network,
	model_module.generative_network,
)

# wandb.watch(model.proposal_network)
wandb.watch(model.generative_network)
# wandb.watch(model.prior_network)

if use_cuda:
	model = model.cuda()

# build optimizer and import its parameters
optimizer = model_module.optimizer(model.parameters(), lr=args.lr)
batch_size = model_module.batch_size
vlb_scale_factor = getattr(model_module, 'vlb_scale_factor', 1)


# load train and validation datasets
# train_dataset = load_dataset(args.train_dataset)
# validation_dataset = load_dataset(args.validation_dataset)

# translate, scale, shear, rotate = (0.1, 0.1), None, None, 0
# translate, scale, shear, rotate = None, None, None, 0

# transform = transforms.Compose([
# 		transforms.RandomAffine(rotate, translate, scale, shear),
# 		transforms.ToTensor()
# ])


d_args = make_opt(split="train")
train_dataset = DMSADataset(d_args)


d_args = make_opt(split="test")
# val_transform = transforms.Compose([
# 		transforms.ToTensor()
# ])
validation_dataset = DMSADataset(d_args)


# build dataloaders on top of datasets
dataloader = DataLoader(train_dataset, batch_size=batch_size,
						shuffle=True, drop_last=True,
						num_workers=num_workers)
val_dataloader = DataLoader(validation_dataset, batch_size=batch_size * 5,
							shuffle=False, drop_last=False,
							num_workers=num_workers)

# number of batches after which it is time to do validation
validation_batches = ceil(len(dataloader) / args.validations_per_epoch)

# a list of validation IWAE estimates
validation_iwae = []
# a list of running variational lower bounds on the train set
train_vlb = []
# the length of two lists above is the same because the new
# values are inserted into them at the validation checkpoints only

# load the last checkpoint, if it exists
# if exists(join(args.model_dir, 'last_checkpoint.tar')):
# 	location = 'cuda' if use_cuda else 'cpu'
# 	checkpoint = torch.load(join(args.model_dir, 'last_checkpoint.tar'),
# 							map_location=location)
# 	model.load_state_dict(checkpoint['model_state_dict'])
# 	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# 	validation_iwae = checkpoint['validation_iwae']
# 	train_vlb = checkpoint['train_vlb']


# Makes checkpoint of the current state.
# The checkpoint contains current epoch (in the current run),
# VAEAC and optimizer parameters, learning history.
# The function writes checkpoint to a temporary file,
# and then replaces last_checkpoint.tar with it, because
# the replacement operation is much more atomic than
# the writing the state to the disk operation.
# So if the function is interrupted, last checkpoint should be
# consistent.
def make_checkpoint():
	filename = join(args.model_dir, 'last_{}.tar'.format(args.exp))
	torch.save({
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'validation_iwae': validation_iwae,
		'train_vlb': train_vlb,
	}, filename + '.bak')
	replace(filename + '.bak', filename)


# main train loop
for epoch in range(args.epochs):

	iterator = dataloader
	avg_vlb = 0
	if verbose:
		print('Epoch %d...' % (epoch + 1), file=stderr, flush=True)
		iterator = tqdm(iterator)


	# one epoch
	for i, data in enumerate(iterator):
		batch, output, _ = data

		# the time to do a checkpoint is at start and end of the training
		# and after processing validation_batches batches


		# if batch size is less than batch_size, extend it with objects
		# from the beginning of the dataset
		# batch = extend_batch(batch, dataloader, batch_size)
		# generate mask and do an optimizer step over the mask and the batch
		# mask = mask_generator(batch)


		optimizer.zero_grad()
		if use_cuda:
			batch = batch.cuda()
			output = output.cuda()
			# mask = mask.cuda()
		# vlb = model.batch_vlb(batch, mask).mean()
		# rec_params, mask, proposal, prior = model(batch)
		# r = model(batch)
		# print(r)
		# print(len(r))
		# vlb = model.compute_loss(rec_params, mask, proposal, prior).mean()
		rec_loss, kl, prior_reg = nn.parallel.data_parallel(model, (batch, output), device_ids=range(2))

		vlb = (rec_loss - kl * args.beta + prior_reg).mean()

		(-vlb / vlb_scale_factor).backward()
		optimizer.step()

		# update running variational lower bound average
		avg_vlb += (float(vlb) - avg_vlb) / (i + 1)


	if verbose:
		iterator.set_description('Train VLB: %g' % avg_vlb)
	with torch.no_grad():
		val_iwae, recs = get_validation_iwae(val_dataloader,
											 batch_size * 5, model,
											 args.validation_iwae_num_samples,
											 verbose)
		validation_iwae.append(val_iwae)
		train_vlb.append(avg_vlb)

		make_checkpoint()

		# if current model validation IWAE is the best validation IWAE
		# over the history of training, the current checkpoint is copied
		# to best_checkpoint.tar
		# copying is done through a temporary file, i. e. firstly last
		# checkpoint is copied to temporary file, and then temporary file
		# replaces best checkpoint, so even best checkpoint should be
		# consistent even if the script is interrupted
		# in the middle of copying
		if max(validation_iwae[::-1]) <= val_iwae:
			src_filename = join(args.model_dir, 'last_{}.tar'.format(args.exp))
			dst_filename = join(args.model_dir, 'best_{}.tar'.format(args.exp))
			copy(src_filename, dst_filename + '.bak')
			replace(dst_filename + '.bak', dst_filename)

		if verbose:
			print(file=stderr)
			print(file=stderr)

            

	wandb.log({
		"train vlb": avg_vlb,
		"val iwae": val_iwae,
        "reconstructions":[wandb.Image(recs[i][0], mode="F") for i in range(8)]

	})
