from sklearn.utils import shuffle
# from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
# import codecs
# import errno
# import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./", help="Directory to save jpgs in")
    parser.add_argument("--contrast", default=1, type=int, help="Image contrast to train on")
    parser.add_argument("--view", default="sag", help="siamese, sag, trans")
    parser.add_argument("--split", default=0.7, type=float, help="proportion of dataset to use as training")
    parser.add_argument("--bottom_cut", default=0.0, type=float, help="proportion of dataset to cut from bottom")
    parser.add_argument("--etiology", default="B", help="O (obstruction), R (reflux), B (both)")
    # parser.add_argument('--unet', action="store_true", help="UNet architecthure")
    parser.add_argument("--crop", default=0, type=int, help="Crop setting (0=big, 1=tight)")
    parser.add_argument("--git_dir",default="")

    args = parser.parse_args()

    datafile = args.git_dir + "nephronetwork/0.Preprocess/preprocessed_images_20190617.pickle"

    load_dataset_LE = importlib.machinery.SourceFileLoader('load_dataset_LE', args.git_dir + '/nephronetwork/0.Preprocess/load_dataset_LE.py').load_module()

    train_X, train_y, train_cov, test_X, test_y, test_cov = load_dataset_LE.load_dataset(views_to_get=args.view,
                                                                                         sort_by_date=True,
                                                                                         pickle_file=datafile,
                                                                                         contrast=args.contrast,
                                                                                         split=args.split,
                                                                                         get_cov=True,
                                                                                         bottom_cut=args.bottom_cut,
                                                                                         etiology=args.etiology,
                                                                                         crop=args.crop,
                                                                                         git_dir=args.git_dir)


    for i in range(test_X.shape[1]):
        df_sub = test_X.iloc[i]
        df_sub.head()

#        img_file_name = str(int(sample_id)) + "_" + sample_name[1] + "_" + str(img_num) + ".jpg"
#        scipy.misc.imsave(os.path.join(opt.jpg_dump_dir, img_file_name), resized_image)
#        print('Image file written to' + opt.jpg_dump_dir + img_file_name)
