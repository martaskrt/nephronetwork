import argparse
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split

def split(args):
    target_train_dir = "ILSVRC2012_img_train"
    target_val_dir = "ILSVRC2012_img_val"

    if os.path.isdir(target_train_dir):
        print("Target training dir already exists! Remove it before running script.")
        sys.exit(1)
    if os.path.isdir(target_val_dir):
        print("Target val dir already exists! Remove it before running script.")
        sys.exit(1)

    file_names = []
    for subdir, dirs, files in os.walk(args.dir):
        for file in files:
            if file.lower()[-5:] == ".jpeg":
                file_names.append(os.path.join(subdir, file))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='train-jps', help="path to directory containing image .jpeg files")
    parser.add_argument('--ratio', default=0.8, type=float, help="proportion of images to train on")

    args = parser.parse_args()
    split(args)

if __name__ == "__main__":
    main()