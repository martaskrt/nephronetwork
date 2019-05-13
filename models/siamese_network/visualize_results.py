
'''
Input is text file with the following format:
# TrainEpoch\tNUM\tACC\tNUM%\tLoss\tNUM\tAUC\tNUM\tAUPRC\tNUM\tTN\tNUM\tFP\tNUM\tFN\tNUM\tTP\tNUM
'''

import argparse
# import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
import os
import importlib.machinery
from PIL import Image, ImageDraw, ImageFont

load_dataset = importlib.machinery.SourceFileLoader('load_dataset','../preprocess/load_dataset.py').load_module()

threshold=0.5


def view_images(args, wrong_samples, pred_prob, groud_truth):
    datafile="../preprocess/preprocessed_images_20190402.pickle"
    _, _, _, test_X, _, test_cov = load_dataset.load_dataset(views_to_get="siamese", sort_by_date=True,
                                                             pickle_file=datafile, contrast=1,
                                                             split=0.9, get_cov=True)

    try:
        os.makedirs(args.outdir)
    except:
        print("Directory exists!")
        import sys
        sys.exit(1)
    for id in wrong_samples:
        idx = test_cov.index(id)  # find index of this iterm in test_cov
        scipy.misc.imsave(args.outdir + '/' + str(id) +'.jpg', test_X[idx])

        base = Image.open(args.outdir + '/' + str(id) +'.jpg').convert('RGBA')

        txt = Image.new('RGBA', base.size, (255, 255, 255, 0))
        # get a font
        fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)
        # get a drawing context
        d = ImageDraw.Draw(txt)
        # draw text, half opacity
        d.text((10, 10), "Hello", font=fnt, fill=(255, 255, 255, 128))
        # draw text, full opacity
        d.text((10, 60), "World", font=fnt, fill=(255, 255, 255, 255))
        out = Image.alpha_composite(base, txt)

        out.show()

def get_wrong_samples(args):
    model = torch.load(args.checkpoint, map_location='cpu')
    patient_ID_test = model['patient_ID_test']
    target_test = model['all_targets_test']
    pred_prob_test = model['all_pred_prob_test'] ## THIS IS PROBABILITY OF CLASS 1
    pred_labels = []
    for sample in pred_prob_test:
        if sample >= threshold:
            pred_labels.append(1)
        else:
            pred_labels.append(0)

    ## list of incorrect patients
    wrong = []
    num_samples = len(pred_labels)
    for i in range(num_samples):
        if pred_labels[i] != target_test[i]:
            wrong.append(patient_ID_test[i])
    return wrong, pred_prob_test, target_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='model to process results from')
    parser.add_argument('--outdir', required=True, help='directory to store images')
    args = parser.parse_args()
    wrong_samples, pred_prob, groud_truth = get_wrong_samples(args)
    view_images(args, wrong_samples, pred_prob, groud_truth)
if __name__ == "__main__":
    main()


