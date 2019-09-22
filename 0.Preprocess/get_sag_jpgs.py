import numpy as np
import os
import importlib.machinery
import scipy.misc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="/storage/sag-test-jpgs/", help="Directory to save jpgs in")
    parser.add_argument("--contrast", default=1, type=int, help="Image contrast to train on")
    parser.add_argument("--view", default="sag", help="siamese, sag, trans")
    parser.add_argument("--split", default=0.7, type=float, help="proportion of dataset to use as training")
    parser.add_argument("--bottom_cut", default=0.0, type=float, help="proportion of dataset to cut from bottom")
    parser.add_argument("--etiology", default="B", help="O (obstruction), R (reflux), B (both)")
    parser.add_argument("--crop", default=0, type=int, help="Crop setting (0=big, 1=tight)")
    parser.add_argument("--git_dir",default="/home/lauren/")

    args = parser.parse_args()

    datafile = args.git_dir + "nephronetwork/0.Preprocess/preprocessed_images_20190617.pickle"

    load_dataset_LE = importlib.machinery.SourceFileLoader('load_dataset_LE', args.git_dir + '/nephronetwork/0.Preprocess/load_dataset_LE.py').load_module()

    train_X, train_y, train_cov, test_X, test_y, test_cov = load_dataset_LE.load_dataset(views_to_get=args.view,sort_by_date=True,pickle_file=datafile,contrast=args.contrast,split=args.split,get_cov=True,bottom_cut=args.bottom_cut,etiology=args.etiology,crop=args.crop,git_dir=args.git_dir)


    for i in range(test_X.shape[1]):
        img = test_X[i]

        img_details = train_cov[i].split("_")

        img_file_name = str(int(float(img_details[0]))) + "_" + img_details[3] + "_" + img_details[4] + ".jpg"
        scipy.misc.imsave(os.path.join(args.dir, img_file_name), img)
        print('Image file written to' + args.dir + img_file_name)




##### DEBUG

# args_dict = {'dir': '/storage/sag-test-jpgs/',
#         'contrast': 1,
#         'view': 'sag', ## trans, sag
#         'split':0.7,
#         'bottom_cut':0.0,
#         'etiology':'B',
#         'crop':0,
#         'git_dir':"/home/lauren/"
#          }
#
# class myargs():
#     def __init__(self,args_dict):
#         self.dir = args_dict['dir']
#         self.contrast = args_dict['contrast']
#         self.view = args_dict['view']
#         self.split = args_dict['split']
#         self.bottom_cut = args_dict['bottom_cut']
#         self.etiology = args_dict['etiology']
#         self.crop = args_dict['crop']
#         self.git_dir = args_dict['git_dir']
#
# args = myargs(args_dict)
# args.git_dir
