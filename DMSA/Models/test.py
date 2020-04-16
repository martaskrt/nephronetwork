import numpy as np

import matplotlib.pyplot as plt
from skimage import exposure, color

import random
from tqdm import tqdm
import pandas as pd
from sklearn.linear_model import LogisticRegression


import argparse
import math
import os

import torch
from torch import optim
from torch.nn import functional as F
print(torch.__version__)

from torchvision import transforms

print("-")
from PIL import Image
print("-")
from tqdm import tqdm
print("-")
print("-")
print("-")
import numpy as np
# from models2 import VariationalAutoEncoderLite
# import cv2
from skimage import exposure
import scipy.misc

from sklearn.metrics import roc_curve, auc, roc_auc_score



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
from VAEAC import VAEAC
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
import random
from PIL import Image
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str)

parser.add_argument('--ckpt', type=str, required=True)

# parser.add_argument('--run', type=int, required=True)
# parser.add_argument('--mse', type=float, required=True)

# parser.add_argument('--healthy', type=str, required=True)




args = parser.parse_args()

print(args.ckpt, "-------------------------------------------------")

# args.ckpt = "test_diseased_0.0_10.0_2_vgg_reswgangp_lambda20.0_lrG0.0001_lrD0.0002_iter21000.pt"



metadata = pd.read_csv("../mri_gan_cancer/data/preproc_chest_metadata.csv")

# latents_healthy = np.load("latents/test_diseased_0.0_0_vgg_reg_chest_031275.pt_latents.npy")
# noises_healthy = np.load("latents/test_diseased_0.0_0_vgg_reg_chest_031275.pt_latents.npy")

# healthy_dist = np.sum(latents_healthy ** 2, axis=-1)


device = "cuda"


class FairDataset(Dataset):
	# def __init__(self, path, transform, resolution=256):

	# # def __init__(self,csv_file,root_dir,transform=None):
	#     self.annotations = pd.read_csv("../annotations_slices_medium.csv", engine='python')
	#     self.root_dir = path 
	#     self.transform = transform
	
	# def __len__(self):
	#     return (len(self.annotations))

	# def __getitem__(self,index):
	#     volume_name = os.path.join(self.root_dir,
	#     self.annotations.iloc[index,0])
	#     np_volume = np.load(volume_name)
	#     volume = Image.fromarray(np_volume)
	#     # annotations = self.annotations.iloc[index,0].as_matrix()
	#     # annotations = annotations.astype('float').reshape(-1,2)
	#     sample = volume#[np.newaxis, ...]

	#     if self.transform:
	#         sample = self.transform(sample)
		
	#     return sample
	def __init__(self, path, transform, reg, resolution=512, split="train", run=0, intensity=1, size=10, nodule_mask=0):
		self.intensity = intensity

		self.nodule_mask = abs(nodule_mask)
		self.metadata = pd.read_csv("../mri_gan_cancer/data/preproc_chest_metadata.csv")
		if split == "train":
			self.metadata = self.metadata[self.metadata["train"] == 1]
		elif split == "test":
			self.metadata = self.metadata[self.metadata["train"] == 0]
		else:
			raise Exception("Invalid data split")



		

		data_mean = 0.175
		data_std = 0.17

		adjusted_intensity = intensity / data_std
		adjusted_max = (1 - data_mean) / data_std
		
		self.data = np.load("../mri_gan_cancer/data/chest_data.npy")
		self.min_val = np.amin(self.data)
		self.max_val = np.amax(self.data)
		# print("mean:", np.mean(self.data.flatten()))
		# print("std:", np.std(self.data.flatten()))
		# print("max:", np.amax(self.data))



		self.masks = np.zeros_like(self.data)

		self.positions = pd.read_csv("../mri_gan_cancer/data/nodule_positions.csv")["run_{}".format(run)]

		for i in range(self.data.shape[0]):
			positions = [int(n) for n in self.positions[i].split(",")]

			self.data[i] = self.insert_nodule(self.data[i], adjusted_intensity, size, positions)
			self.masks[i, positions[1] - self.nodule_mask: positions[1] + self.nodule_mask, positions[0] - self.nodule_mask: positions[0] + self.nodule_mask] = 1

			# print("nodules inserted")
		self.data[self.data > adjusted_max] = adjusted_max
		# print("clipped at", adjusted_max)
		# self.data = self.data[self.metadata["npy_idx"]]
		# cv2.imwrite("test/diseased_{}.png".format(run), (self.data[0] * data_std + data_mean) * 255) 
		# cv2.imwrite("test/diseased_{}_mask.png".format(run), (self.masks[0] * 255) )

		self.transform = transform
		self.reg = reg

	
	def insert_nodule(self, im, intensity, sigma, position):
		
		x, y = np.meshgrid(np.linspace(-25, 25, 50), np.linspace(-25, 25, 50))
		d = np.sqrt(x * x + y * y)
		nodule = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))

		nodule_x, nodule_y = position[0], position[1]

		im[nodule_y - 25: nodule_y + 25, nodule_x - 25: nodule_x + 25] += nodule * intensity



		return im



	def __len__(self):
		if self.reg:
			return self.metadata["patient_n"].unique().shape[0]
		return self.metadata.shape[0]


	def __getitem__(self,index):

		if not self.reg:

			npy_idx = self.metadata["npy_idx"].iloc[index] - 1
			im = self.data[int(npy_idx)]
		else:
			patient_rows = self.metadata[self.metadata["patient_n"] == self.metadata["patient_n"].unique()[index]]


			# print(patient_rows, index)
			npy_idx = random.sample(list(patient_rows["npy_idx"]), 1)[0]

			im = self.data[npy_idx - 1]
			

		volume = Image.fromarray(im)
		# annotations = self.annotations.iloc[index,0].as_matrix()
		# annotations = annotations.astype('float').reshape(-1,2)
		sample = volume#[np.newaxis, ...]

		if self.transform:
			sample = self.transform(sample)

		return sample

	def get_nodule_mask(self, index):

		if not self.reg:

			npy_idx = self.metadata["npy_idx"].iloc[index] - 1
			mask = self.masks[int(npy_idx)]
		else:
			patient_rows = self.metadata[self.metadata["patient_n"] == self.metadata["patient_n"].unique()[index]]


			# print(patient_rows, index)
			npy_idx = random.sample(list(patient_rows["npy_idx"]), 1)[0]

			mask = self.masks[npy_idx - 1]
		return mask

def get_npy_idx(patient_n):
	# print(test_metadata[test_metadata["patient_n"].isin(patient_n)])
	return np.arange(95)[test_metadata["patient_n"].isin(patient_n)]


def get_lr_features(query_images, anomaly_score=False):




	# percept_losses = []
	pixel_losses = []
	latent_losses = []
	vlb_losses = []
	reg_losses = []

	for i in range(len(query_images)):
		with torch.no_grad():

			query= query_images[i]
			query = query.unsqueeze(0).to(device)



			# #         # print(img_gen.shape)
			# im = img_gen.squeeze(0).squeeze(0).cpu().numpy()
			# im = im - np.amin(im.flatten())
			# im = im / np.amax(im.flatten()) * 255
			
			# cv2.imwrite("test/test{}_same.png".format(i), im[0])
					

	




			mask_ar = torch.from_numpy(query_images.get_nodule_mask(i)).unsqueeze(0).unsqueeze(0).to(device).float()

			img_gen = model.generate_samples_params(query, mask_ar, K=1)[0, 0, 0].reshape(1, 1, 512, 512)
			torchvision.utils.save_image(img_gen, "test/test{}_{}.png".format(query_images.intensity, i), normalize=False, nrow=2)


			rec_loss, kl, prior_regularization = model.features(query, mask_ar)

			pixel_losses.append(rec_loss.item())
			latent_losses.append(kl.item())
			reg_losses.append(prior_regularization.item())
			# vlb_losses.append((rec_loss + kl - prior_regularization).item())
			# no_reg_losses.append((rec_loss + kl).item())



			# img_name = "gen_imgs/diseased_{}_{}_{}_{}_{}_{}_{}_project.png".format(args.run, args.nodule_intensity, args.feature_extractor, batch_start + n, args.latent_space, args.mask)
			# cv2.imwrite(img_name, np.concatenate((img_ar[n], gen_ar[n])))




			# percept_losses.append(0)


			# pixel_loss = F.l1_loss(img_gen, query)


			# pixel_loss = F.l1_loss(img_gen * mask_ar, query * mask_ar)
			# pixel_losses.append(pixel_loss.item())

			# plt.imshow(query_images[i].squeeze(0))
			# plt.show()

			# plt.imshow(img_gen.cpu().numpy())
			# plt.show()

	pixel_losses = np.array(pixel_losses)[..., np.newaxis]
	latent_losses = np.array(latent_losses)[..., np.newaxis]
	reg_losses = np.array(reg_losses)[..., np.newaxis]

	features = np.concatenate((pixel_losses, latent_losses, reg_losses), axis=-1).reshape(95, 3)
	return -features


def split(patient_list, healthy_features, diseased_features, train_ratio=1, disease_ratio=0.5):

	# split train/test patients
	train_idx = random.sample(patient_list, int(len(patients) * train_ratio))
	test_idx = list(set(patients) - set(train_idx))

	# train
	healthy_idx_train = random.sample(train_idx, int(len(train_idx) * (1 - disease_ratio)))
	diseased_idx_train = list(set(train_idx) - set(healthy_idx_train))

	healthy_npy_train = get_npy_idx(healthy_idx_train)
	diseased_npy_train = get_npy_idx(diseased_idx_train)

	# test
	healthy_idx_test = random.sample(test_idx, int(len(test_idx) * (1 - disease_ratio)))
	diseased_idx_test = list(set(test_idx) - set(healthy_idx_test))

	healthy_npy_test = get_npy_idx(healthy_idx_test)
	diseased_npy_test = get_npy_idx(diseased_idx_test)


	X_train = np.concatenate((healthy_features[healthy_npy_train], diseased_features[diseased_npy_train]))
	y_train = np.concatenate((np.zeros(len(healthy_npy_train)), np.ones(len(diseased_npy_train))))

	X_test = np.concatenate((healthy_features[healthy_npy_test], diseased_features[diseased_npy_test]))
	y_test = np.concatenate((np.zeros(len(healthy_npy_test)), np.ones(len(diseased_npy_test))))

	return X_train, y_train, X_test, y_test


ascore = False
ckpt_root = "latents/"

coefs = []

# Default parameters which are not supposed to be changed from user interface
use_cuda = torch.cuda.is_available()

print("gpu available:", use_cuda)
verbose = True
# Non-zero number of workers cause nasty warnings because of some bug in
# multiprocess library. It might be fixed now, so maybe it is time to set it
# to the number of CPU cores in the system.
num_workers = 32

# import the module with the model networks definitions,
# optimization settings, and a mask generator
model_module = import_module(args.model_dir + '.model')


# import mask generator
mask_generator = model_module.mask_generator


model = VAEAC(
	model_module.reconstruction_log_prob,
	model_module.proposal_network,
	model_module.prior_network,
	model_module.generative_network,
	mask_generator
)

transform = transforms.Compose([
	   transforms.ToTensor()
	])

# load the last checkpoint, if it exists



if not exists(join(args.model_dir, args.ckpt)):
	raise Exception("model does not exist")


location = 'cuda'
checkpoint = torch.load(join(args.model_dir, args.ckpt),
						map_location=location)
model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# validation_iwae = checkpoint['validation_iwae']
# train_vlb = checkpoint['train_vlb']

model = model.to(device)
model.eval()
print("model loaded")


# size = 10.0
mask = 30
# mask size used during latent optimization
for size in [10.0]:
	for intensity in [0.2, 0.3, 0.4, 0.5, 0.6]:
		for mask_val in ["zero"]:
			for test_mask in [mask]: # [25, 50, 100, 200, 512]:
				for fe in ["vgg"]:

					for mse in [1.0]:

						for run in [7, 8, 9]:
							healthy_query = FairDataset(None, transform=transform, reg=False, resolution=512, split="test", run=run, intensity=0, size=size, nodule_mask=mask) # mask size used for testing
							diseased_query = FairDataset(None, transform=transform, reg=False, resolution=512, split="test", run=run, intensity=intensity, size=size, nodule_mask=mask)    

							with torch.no_grad():
								# batch = batch.cuda()
								# mask = torch.zeros_like(batch)
								# mask[:, :, 241:271, 241:271] = 1

								# samples = model.generate_samples_params(batch, mask, K=10)
								# print(samples.max(), samples.min(), samples.mean())
								# print(batch.max(), batch.min(), batch.mean())
								# print(model.batch_vlb(batch, mask), "-------------")
								# torchvision.utils.save_image(samples[0:5, 0, 0].reshape(5, 1, 512, 512), "test/test{}.png".format(i), normalize=False, nrow=2)
								






								healthy_features =  get_lr_features(healthy_query, anomaly_score=ascore)


								diseased_features = get_lr_features(diseased_query, anomaly_score=ascore)

							# healthy_query = FairDataset(None, transform=transform, reg=False, resolution=512, split="test", run=run, intensity=0, size=size)
							# healthy_features =  get_lr_features(healthy_prefix + "_latents.npy", healthy_prefix + "_noises.npz", healthy_query, anomaly_score=ascore)

							# diseased_query = FairDataset(None, transform=transform, reg=False, resolution=512, split="test", run=run, intensity=intensity, size=size)    
							# diseased_features = get_lr_features(diseased_prefix + "_latents.npy", diseased_prefix + "_noises.npz", diseased_query, anomaly_score=ascore)

							
							


							# BOOTSTRAP FOR LOGISTIC REGRESSION OVER STYLE LATENTS
							# healthy_dist = np.sum((w_healthy - w_mean) ** 2, axis=-1)
							# diseased_dist = np.sum((w_diseased - w_mean) ** 2, axis=-1)
							# diseased_dist = np.sum(w_diseased ** 2, axis=-1)

							scores = []
							anomaly_scores = []
							aucs = []
							lr_aucs = []
							tprs = []
							fprs = []

							pixel_aucs = []
							latent_aucs = []
							reg_aucs = []
							vlb_aucs = []
							no_reg_aucs = []

							for i in range(100):

								test_metadata = metadata[metadata["train"] == 0]

								patients = list(metadata[metadata["train"] == 0]["patient_n"].unique())

								# X_train, y_train, X_test, y_test = split(patients, train_ratio=0.75)
								X_test, y_test, _, _ = split(patients, healthy_features, diseased_features, train_ratio=1, disease_ratio=0.5)


								# clf = LogisticRegression(random_state=0, max_iter=1000, solver="liblinear").fit(X_train, y_train)

								# # print(clf.predict_proba(X_test))

								# lr_y_scores = clf.predict_proba(X_test)[:, 1]
								# lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_y_scores)
								# lr_roc_auc = auc(lr_fpr, lr_tpr)
								

								fpr, tpr, thresh = roc_curve(y_test, X_test[:, 0])
								pixel_auc = auc(fpr, tpr)
								# precision, recall, thresholds = precision_recall_curve(y_test, y_score)
								# prc_auc = auc(precision, recall)
								pixel_aucs.append(pixel_auc)

								fpr, tpr, thresh = roc_curve(y_test, X_test[:, 1])
								latent_auc = auc(fpr, tpr)
								# precision, recall, thresholds = precision_recall_curve(y_test, y_score)
								# prc_auc = auc(precision, recall)
								latent_aucs.append(latent_auc)
								
								fpr, tpr, thresh = roc_curve(y_test, X_test[:, 2])
								reg_auc = auc(fpr, tpr)
								# precision, recall, thresholds = precision_recall_curve(y_test, y_score)
								# prc_auc = auc(precision, recall)
								reg_aucs.append(reg_auc)
								
								fpr, tpr, thresh = roc_curve(y_test, X_test[:, 0] + X_test[:, 1] - X_test[:, 2])
								vlb_auc = auc(fpr, tpr)
								# precision, recall, thresholds = precision_recall_curve(y_test, y_score)
								# prc_auc = auc(precision, recall)
								vlb_aucs.append(vlb_auc)
								
								fpr, tpr, thresh = roc_curve(y_test, X_test[:, 0] + X_test[:, 1])
								no_reg_auc = auc(fpr, tpr)
								# precision, recall, thresholds = precision_recall_curve(y_test, y_score)
								# prc_auc = auc(precision, recall)
								no_reg_aucs.append(no_reg_auc)
								
								# scores.append(clf.score(X_test, y_test))
								# print(clf.predict_proba(X_test))
								# lr_aucs.append(lr_roc_auc)
								# coefs.append(clf.coef_)
								# fprs.append(fpr)
								# tprs.append(tpr)

							# lw = 2
							# plt.plot(np.mean(fpr, axis=-1), np.mean(tpr, axis=-1), color='darkorange',
							#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
							# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
							# plt.xlim([0.0, 1.0])
							# plt.ylim([0.0, 1.05])
							# plt.xlabel('False Positive Rate')
							# plt.ylabel('True Positive Rate')
							# plt.legend(loc="lower right")
							# # print(clf.get_params())
							# # print(X_train, y_train)-
							# plt.savefig("auprc {}.png".format(intensity))

							# fig, axs = plt.subplots(3)
							# fig.suptitle("test_diseased_{2}_{0}_{3}_reg_chest_031275.pt_w_{1}_losses_auc_{4}_{5}".format(run, mse, intensity, fe, np.mean(aucs), np.std(aucs)))
							# axs[0].hist(healthy_losses["p_losses"].reshape(-1), alpha=0.5, label='healthy p losses')
							# axs[0].hist(diseased_losses["p_losses"].reshape(-1), alpha=0.5, label='diseased p losses')
							# axs[1].hist(healthy_losses["mse_losses"].reshape(-1), alpha=0.5, label='healthy mse losses')
							# axs[1].hist(diseased_losses["mse_losses"].reshape(-1), alpha=0.5, label='diseased mse losses')
							# axs[2].hist(healthy_losses["n_losses"].reshape(-1), alpha=0.5, label='healthy n losses')
							# axs[2].hist(diseased_losses["n_losses"].reshape(-1), alpha=0.5, label='diseased n losses')
							

							# axs[0].legend(loc='upper right')
							# axs[1].legend(loc='upper right')
							# axs[2].legend(loc='upper right')

							# fig.savefig("figures/test_diseased_{2}_{0}_{3}_reg_chest_031275.pt_w_{1}_losses.png".format(run, mse, intensity, fe))

							# fig, axs = plt.subplots(1)
							# fig.suptitle("test_diseased_{2}_{0}_{3}_reg_chest_031275.pt_w_{1}_{6}_{7}_{8}_losses_auc_{4}_{5}".format(run, mse, intensity, fe, np.mean(aucs), np.std(aucs), mask, test_mask, mask_val))
							# axs.hist(healthy_features[:, -1].reshape(-1), alpha=0.5, label='healthy p losses')
							# axs.hist(diseased_features[:, -1].reshape(-1), alpha=0.5, label='diseased p losses')
						

							# axs.legend(loc='upper right')

							# fig.savefig("figures/test_diseased_{2}_{5}_{0}_{3}_reg_chest_031275.pt_w_{1}_{4}_{6}.png".format(run, mse, intensity, fe, mask, size, mask_val))
							# fig.clf()
							

							print("intensity", intensity, "size", size, "run", run, "==============") 
							print("pixel auc", np.mean(pixel_aucs), np.std(pixel_aucs)) 
							print("latent auc", np.mean(latent_aucs), np.std(latent_aucs))
							print("reg auc", np.mean(reg_aucs), np.std(reg_aucs)) 
							print("vlb auc", np.mean(vlb_aucs), np.std(vlb_aucs))
							print("no_reg auc", np.mean(no_reg_aucs), np.std(no_reg_aucs))


# class FairDataset(Dataset):
# 	# def __init__(self, path, transform, resolution=256):

# 	# # def __init__(self,csv_file,root_dir,transform=None):
# 	#     self.annotations = pd.read_csv("../annotations_slices_medium.csv", engine='python')
# 	#     self.root_dir = path 
# 	#     self.transform = transform
	
# 	# def __len__(self):
# 	#     return (len(self.annotations))

# 	# def __getitem__(self,index):
# 	#     volume_name = os.path.join(self.root_dir,
# 	#     self.annotations.iloc[index,0])
# 	#     np_volume = np.load(volume_name)
# 	#     volume = Image.fromarray(np_volume)
# 	#     # annotations = self.annotations.iloc[index,0].as_matrix()
# 	#     # annotations = annotations.astype('float').reshape(-1,2)
# 	#     sample = volume#[np.newaxis, ...]

# 	#     if self.transform:
# 	#         sample = self.transform(sample)
		
# 	#     return sample
# 	def __init__(self, path, transform, reg, resolution=512, split="train", run=0, intensity=1, size=10, nodule_mask=0):


# 		self.nodule_mask = abs(nodule_mask)
# 		self.metadata = pd.read_csv("../mri_gan_cancer/data/preproc_chest_metadata.csv")
# 		if split == "train":
# 			self.metadata = self.metadata[self.metadata["train"] == 1]
# 		elif split == "test":
# 			self.metadata = self.metadata[self.metadata["train"] == 0]
# 		else:
# 			raise Exception("Invalid data split")



		

# 		data_mean = 0.175
# 		data_std = 0.17

# 		adjusted_intensity = intensity / data_std
# 		adjusted_max = (1 - data_mean) / data_std
		
# 		self.data = np.load("../mri_gan_cancer/data/chest_data.npy")
# 		# self.min_val = np.amin(self.data)
# 		# self.max_val = np.amax(self.data)
# 		# # print("mean:", np.mean(self.data.flatten()))
# 		# # print("std:", np.std(self.data.flatten()))
# 		# # print("max:", np.amax(self.data))



# 		# self.masks = np.zeros_like(self.data)

# 		# self.positions = pd.read_csv("../mri_gan_cancer/data/nodule_positions.csv")["run_{}".format(run)]

# 		# for i in range(self.data.shape[0]):
# 		#     positions = [int(n) for n in self.positions[i].split(",")]

# 		#     self.data[i] = self.insert_nodule(self.data[i], adjusted_intensity, size, positions)
# 		#     self.masks[i, positions[1] - self.nodule_mask: positions[1] + self.nodule_mask, positions[0] - self.nodule_mask: positions[0] + self.nodule_mask] = 1

# 		#     # print("nodules inserted")
# 		# self.data[self.data > adjusted_max] = adjusted_max
# 		# # print("clipped at", adjusted_max)
# 		# self.data = self.data[self.metadata["npy_idx"]]
# 		# cv2.imwrite("test/diseased_{}.png".format(run), (self.data[0] * data_std + data_mean) * 255) 
# 		# cv2.imwrite("test/diseased_{}_mask.png".format(run), (self.masks[0] * 255) )

# 		self.transform = transform
# 		self.reg = reg

	
# 	def insert_nodule(self, im, intensity, sigma, position):
		
# 		x, y = np.meshgrid(np.linspace(-25, 25, 50), np.linspace(-25, 25, 50))
# 		d = np.sqrt(x * x + y * y)
# 		nodule = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))

# 		nodule_x, nodule_y = position[0], position[1]

# 		im[nodule_y - 25: nodule_y + 25, nodule_x - 25: nodule_x + 25] += nodule * intensity



# 		return im



# 	def __len__(self):
# 		if self.reg:
# 			return self.metadata["patient_n"].unique().shape[0]
# 		return self.metadata.shape[0]


# 	def __getitem__(self,index):

# 		if not self.reg:

# 			npy_idx = self.metadata["npy_idx"].iloc[index] - 1
# 			im = self.data[int(npy_idx)]
# 		else:
# 			patient_rows = self.metadata[self.metadata["patient_n"] == self.metadata["patient_n"].unique()[index]]


# 			# print(patient_rows, index)
# 			npy_idx = random.sample(list(patient_rows["npy_idx"]), 1)[0]

# 			im = self.data[npy_idx - 1]
			

# 		volume = Image.fromarray(im)
# 		# annotations = self.annotations.iloc[index,0].as_matrix()
# 		# annotations = annotations.astype('float').reshape(-1,2)
# 		sample = volume#[np.newaxis, ...]

# 		if self.transform:
# 			sample = self.transform(sample)

# 		return sample

# 	def get_nodule_mask(self, index):

# 		if not self.reg:

# 			npy_idx = self.metadata["npy_idx"].iloc[index] - 1
# 			mask = self.masks[int(npy_idx)]
# 		else:
# 			patient_rows = self.metadata[self.metadata["patient_n"] == self.metadata["patient_n"].unique()[index]]


# 			# print(patient_rows, index)
# 			npy_idx = random.sample(list(patient_rows["npy_idx"]), 1)[0]

# 			mask = self.masks[npy_idx - 1]
# 		return mask

