import argparse
import torch
import matplotlib.pyplot as plt
import importlib.machinery
from torch import nn
import numpy as np
import os
import scipy.misc
load_dataset = importlib.machinery.SourceFileLoader('load_dataset','../../preprocess/load_dataset.py').load_module()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from SiameseNetwork import SiamNet

def load(model, checkpoint):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(checkpoint, map_location='cpu')
    pretrained_dict = {k: v for k, v in list(pretrained_dict.items()) if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    # print([k for k, v in list(pretrained_dict.items())])
    return model

def plot_activations(args):
    #train_X, train_y, test_X, test_y = load_dataset.load_dataset(views_to_get="siamese", sort_by_date=True, pickle_file=args.datafile,
                                                                 #contrast=args.contrast)
    model = SiamNet()
    model = load(model, args.checkpoint)
    # model.load_state_dict(pretrained_dict)
    # print(test_X.shape)
    #X = torch.from_numpy(test_X).float()[0].unsqueeze(0)
    model.cpu()
    #outputs = model(X)
    id = 1
    outdir = "activation_outputs_" + str(id)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    def add_border(img):
        return np.pad(img, 1, "constant", constant_values=1.0)

    def draw_activations(path, activation, imgwidth=2):
        print(activation.shape)

        img = np.vstack([
            np.hstack([
                add_border(filter) for filter in
                activation[i * imgwidth:(i + 1) * imgwidth, :, :]])
            for i in range(activation.shape[0] // imgwidth)])
        scipy.misc.imsave(path, img)
# model.conv.conv1_s1,
    for i, tensor in enumerate([model.conv.conv1_s1, model.conv.conv1_s1,model.conv.conv2_s1, model.conv.conv3_s1, model.conv.conv4_s1,
                                model.conv.conv5_s1, model.fc6.fc6_s1, model.fc6b.conv6b_s1]):
        draw_activations(
            os.path.join(outdir, "conv%d_out_%d.png" % (i + 1, id)),
            tensor.weight.cpu().detach().numpy()[0])
    print("visualization results are saved to %s" % outdir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to model checkpoint .pth file")
    parser.add_argument("--contrast", default=0, type=int, help="Image contrast to train on")

    parser.add_argument("--datafile", default="/Volumes/terminator/nephronetwork/preprocess/"
                                              "preprocessed_images_20190315.pickle",
                        help="File containing pandas dataframe with images stored as numpy array")

    args = parser.parse_args()
    plot_activations(args)

if __name__ == "__main__":
    main()