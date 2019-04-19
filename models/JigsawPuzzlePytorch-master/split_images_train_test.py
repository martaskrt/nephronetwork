import argparse
import os
import sys
import shutil
from sklearn.model_selection import train_test_split

def split(args):
    target_train_dir = "ILSVRC2012_img_train"
    target_val_dir = "ILSVRC2012_img_val"

    # ****** check if directories exist. if yes, exist; else create them ****** #
    if os.path.isdir(target_train_dir):
        print("Target training dir already exists! Remove it before running script.")
        sys.exit(1)
    if os.path.isdir(target_val_dir):
        print("Target val dir already exists! Remove it before running script.")
        sys.exit(1)

    os.makedirs(target_train_dir)
    os.makedirs(target_val_dir)

    # ****** get file names in src dir, split them by train and val, and move them to new dir ********* #
    file_names = []
    for subdir, dirs, files in os.walk(args.dir):
        for file in files:
            if file.lower()[-4:] == ".jpg":
                file_names.append(os.path.join(subdir, file))

    file_names_train, file_names_val = train_test_split(file_names, test_size=args.ratio, random_state=42)

    f = open(target_train_dir + "/ilsvrc12_train.txt", 'w')
    for fname in file_names_train:
        short_fname = fname.split("/")[-1]
        f.write(short_fname + '\n')
        shutil.copy(fname, target_train_dir)
    f.close()

    g = open(target_val_dir + "/ilsvrc12_val.txt", 'w')
    for fname in file_names_val:
        short_fname = fname.split("/")[-1]
        g.write(short_fname + '\n')
        shutil.copy(fname, target_val_dir)
    g.close()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='train-jpgs', help="path to directory containing image .jpeg files")
    parser.add_argument('--ratio', default=0.2, type=float, help="proportion of images to test on")

    args = parser.parse_args()
    split(args)

if __name__ == "__main__":
    main()